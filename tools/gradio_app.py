try:
    import spaces  # type: ignore

    IN_SPACES = True
except ImportError:
    print("Not running on Zero")
    IN_SPACES = False

import PIL
import PIL.Image
from PIL.Image import Image

from mini_nvs_solver.rr_logging_utils import (
    log_camera,
    create_svd_blueprint,
)

from mini_nvs_solver.pose_utils import generate_camera_parameters
from mini_nvs_solver.camera_parameters import PinholeParameters
from mini_nvs_solver.depth_utils import image_to_depth
from mini_nvs_solver.image_warping import image_depth_warping
from mini_nvs_solver.sigma_utils import load_lambda_ts
from mini_nvs_solver.nerfstudio_data import frames_to_nerfstudio


import gradio as gr
from gradio_rerun import Rerun

import rerun as rr
import rerun.blueprint as rrb

import numpy as np
import PIL
import torch
from pathlib import Path
import trimesh
import subprocess

import mmcv
from uuid import uuid4

from typing import Final, Literal

from jaxtyping import Float64, Float32, UInt8

from monopriors.relative_depth_models.depth_anything_v2 import DepthAnythingV2Predictor

from mini_nvs_solver.custom_diffusers_pipeline.svd import StableVideoDiffusionPipeline
from mini_nvs_solver.custom_diffusers_pipeline.scheduler import EulerDiscreteScheduler
from mini_nvs_solver.threaded_logging_utils import svd_render_threaded
from queue import Queue
import threading

SVD_HEIGHT: Final[int] = 576
SVD_WIDTH: Final[int] = 1024
NEAR: Final[float] = 0.0001
FAR: Final[float] = 500.0

if gr.NO_RELOAD:
    depth_predictor: DepthAnythingV2Predictor = DepthAnythingV2Predictor(
        device="cuda", encoder="vitl"
    )
    SVD_PIPE = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        torch_dtype=torch.float16,
        variant="fp16",
    )
    SVD_PIPE.to("cuda")
    scheduler = EulerDiscreteScheduler.from_config(SVD_PIPE.scheduler.config)
    SVD_PIPE.scheduler = scheduler


def svd_render(
    image_o: PIL.Image.Image,
    masks: Float64[torch.Tensor, "b 72 128"],
    cond_image: PIL.Image.Image,
    lambda_ts: Float64[torch.Tensor, "n b"],
    num_denoise_iters: Literal[2, 5, 25, 50, 100],
    weight_clamp: float,
    svd_pipe: StableVideoDiffusionPipeline,
):
    frames: list[PIL.Image.Image] = svd_pipe(
        [image_o],
        log_queue=None,
        temp_cond=cond_image,
        mask=masks,
        lambda_ts=lambda_ts,
        weight_clamp=weight_clamp,
        num_frames=25,
        decode_chunk_size=8,
        num_inference_steps=num_denoise_iters,
    ).frames[0]
    return frames


if IN_SPACES:
    svd_render = spaces.GPU(svd_render)
    image_to_depth = spaces.GPU(image_to_depth, duration=200)


@rr.thread_local_stream("warped_image")
def gradio_warped_image(
    image_path: str,
    num_denoise_iters: Literal[2, 25, 50, 100],
    direction: Literal["left", "right"],
    degrees_per_frame: int | float,
    major_radius: float = 60.0,
    minor_radius: float = 70.0,
    num_frames: int = 25,  # StableDiffusion Video generates 25 frames
    progress=gr.Progress(track_tqdm=False),
):
    if num_denoise_iters != 2 and IN_SPACES:
        gr.Warning(
            "Running on Zero, anything greater than 2 iterations may cause GPU abort due to long running time"
        )
    # ensure that the degrees per frame is a float
    degrees_per_frame = float(degrees_per_frame)

    image_path: Path = Path(image_path) if isinstance(image_path, str) else image_path
    assert image_path.exists(), f"Image file not found: {image_path}"
    save_path: Path = image_path.parent / f"{image_path.stem}_{uuid4()}"

    # setup rerun logging
    stream = rr.binary_stream()
    parent_log_path = Path("world")
    rr.log(f"{parent_log_path}", rr.ViewCoordinates.LDB, static=True)
    blueprint: rrb.Blueprint = create_svd_blueprint(parent_log_path)
    rr.send_blueprint(blueprint)

    # Load image and resize to SVD dimensions
    rgb_original: Image = PIL.Image.open(image_path)
    rgb_resized: Image = rgb_original.resize(
        (SVD_WIDTH, SVD_HEIGHT), PIL.Image.Resampling.NEAREST
    )
    rgb_np_original: UInt8[np.ndarray, "h w 3"] = np.array(rgb_original)
    rgb_np_hw3: UInt8[np.ndarray, "h w 3"] = np.array(rgb_resized)

    # generate initial camera parameters for video trajectory
    camera_list: list[PinholeParameters] = generate_camera_parameters(
        num_frames=num_frames,
        image_width=SVD_WIDTH,
        image_height=SVD_HEIGHT,
        degrees_per_frame=degrees_per_frame,
        major_radius=major_radius,
        minor_radius=minor_radius,
        direction=direction,
    )

    assert len(camera_list) == num_frames, "Number of camera parameters mismatch"

    # Estimate depth map and pointcloud for the input image
    depth: Float32[np.ndarray, "h w"]
    trimesh_pc: trimesh.PointCloud
    depth_original: Float32[np.ndarray, "original_h original_w"]
    trimesh_pc_original: trimesh.PointCloud

    depth, trimesh_pc, depth_original, trimesh_pc_original = image_to_depth(
        rgb_np_original=rgb_np_original,
        rgb_np_hw3=rgb_np_hw3,
        cam_params=camera_list[0],
        near=NEAR,
        far=FAR,
        depth_predictor=depth_predictor,
    )

    rr.log(
        f"{parent_log_path}/point_cloud",
        rr.Points3D(
            positions=trimesh_pc.vertices,
            colors=trimesh_pc.colors,
        ),
        static=True,
    )

    start_cam: PinholeParameters = camera_list[0]
    cond_image: list[PIL.Image.Image] = []
    masks: list[Float64[torch.Tensor, "1 72 128"]] = []

    # Perform image depth warping to generated camera parameters
    current_cam: PinholeParameters
    for frame_id, current_cam in enumerate(camera_list):
        rr.set_time_sequence("frame_id", frame_id)
        if frame_id == 0:
            cam_log_path: Path = parent_log_path / "warped_camera"
            log_camera(cam_log_path, current_cam, rgb_np_hw3, depth)
        else:
            # clear logged depth from the previous frame
            rr.log(f"{cam_log_path}/pinhole/depth", rr.Clear(recursive=False))
            cam_log_path: Path = parent_log_path / "warped_camera"
            # do image warping
            warped_frame2, mask_erosion_tensor = image_depth_warping(
                image=rgb_np_hw3,
                depth=depth,
                cam_T_world_44_s=start_cam.extrinsics.cam_T_world,
                cam_T_world_44_t=current_cam.extrinsics.cam_T_world,
                K=current_cam.intrinsics.k_matrix,
            )
            cond_image.append(warped_frame2)
            masks.append(mask_erosion_tensor)

            log_camera(cam_log_path, current_cam, np.asarray(warped_frame2))
            yield stream.read(), None, [], "Warping images"

    masks: Float64[torch.Tensor, "b 72 128"] = torch.cat(masks)
    # load sigmas to optimize for timestep
    progress(0.1, desc="Optimizing timesteps for diffusion")
    lambda_ts: Float64[torch.Tensor, "n b"] = load_lambda_ts(num_denoise_iters)
    progress(0.15, desc="Starting diffusion")

    if IN_SPACES:
        frames: list[PIL.Image.Image] = svd_render(
            image_o=rgb_resized,
            masks=masks,
            cond_image=cond_image,
            lambda_ts=lambda_ts,
            num_denoise_iters=num_denoise_iters,
            weight_clamp=0.2,
            svd_pipe=SVD_PIPE,
        )
    else:
        # to allow logging from a separate thread
        log_queue: Queue = Queue()
        handle = threading.Thread(
            target=svd_render_threaded,
            kwargs={
                "image_o": rgb_resized,
                "masks": masks,
                "cond_image": cond_image,
                "lambda_ts": lambda_ts,
                "num_denoise_iters": num_denoise_iters,
                "weight_clamp": 0.2,
                "svd_pipe": SVD_PIPE,
                "log_queue": log_queue,
            },
        )

        handle.start()
        i = 0
        while True:
            msg = log_queue.get()
            match msg:
                case frames if all(
                    isinstance(frame, PIL.Image.Image) for frame in frames
                ):
                    break
                case entity_path, entity, times:
                    i += 1
                    rr.reset_time()
                    for timeline, time in times:
                        if isinstance(time, int):
                            rr.set_time_sequence(timeline, time)
                        else:
                            rr.set_time_seconds(timeline, time)
                    static = False
                    if entity_path == "latents":
                        static = True
                    rr.log(entity_path, entity, static=static)
                    yield stream.read(), None, [], f"{i} out of {num_denoise_iters}"
                case _:
                    assert False
        handle.join()

    # all frames but the first one
    frame: np.ndarray
    for frame_id, (frame, cam_pararms) in enumerate(zip(frames, camera_list)):
        # add one since the first frame is the original image
        rr.set_time_sequence("frame_id", frame_id)
        cam_log_path = parent_log_path / "generated_camera"
        generated_rgb_np: UInt8[np.ndarray, "h w 3"] = np.array(frame)
        print(f"Logging frame {frame_id}")
        log_camera(cam_log_path, cam_pararms, generated_rgb_np, depth=None)
        yield stream.read(), None, [], "Logging generated frames"

    frames_to_nerfstudio(
        rgb_np_original, frames, trimesh_pc_original, camera_list, save_path
    )
    # zip up nerfstudio data
    zip_file_path = save_path / "nerfstudio.zip"
    # progress(0.95, desc="Zipping up camera data in nerfstudio format")
    # Run the zip command
    subprocess.run(["zip", "-r", str(zip_file_path), str(save_path)], check=True)
    video_file_path = save_path / "output.mp4"
    mmcv.frames2video(str(save_path), str(video_file_path), fps=7)
    print(f"Video saved to {video_file_path}")

    yield stream.read(), video_file_path, [str(zip_file_path)], "finished"


title = "# Mini NVS Solver"
description1 = """### Demo showcasing [NVS-Solver: Video Diffusion Model as Zero-Shot Novel View Synthesizer](https://github.com/ZHU-Zhiyu/NVS_Solver)"""
description2 = """### Checkout the Mini NVS Solver Repo [![GitHub Repo stars](https://img.shields.io/github/stars/pablovela5620/mini-nvs-solver)](https://github.com/pablovela5620/mini-nvs-solver)"""
description3 = """### HIGHLY RECOMMNEDED to run this demo locally through the github repo to avoid GPU abort due to long running time"""

with gr.Blocks() as demo:
    gr.Markdown(title)
    gr.Markdown(description1)
    gr.Markdown(description2)
    gr.Markdown(description3)
    with gr.Row():
        img = gr.Image(interactive=True, label="Image", type="filepath")
        with gr.Tab(label="Settings"):
            with gr.Column():
                warp_img_btn = gr.Button("Warp Images")
                num_iters = gr.Radio(
                    choices=[2, 25, 50, 100],
                    value=2,
                    label="Number of iterations",
                    type="value",
                )
                cam_direction = gr.Radio(
                    choices=["left", "right"],
                    value="left",
                    label="Camera direction",
                    type="value",
                )
                degrees_per_frame = gr.Slider(
                    minimum=0.25,
                    maximum=1.0,
                    step=0.05,
                    value=0.3,
                    label="Degrees per frame",
                )
                iteration_num = gr.Textbox(
                    value="",
                    label="Status",
                )
        with gr.Tab(label="Outputs"):
            video_output = gr.Video(interactive=False)
            image_files_output = gr.File(interactive=False, file_count="multiple")

    # Rerun 0.16 has issues when embedded in a Gradio tab, so we share a viewer between all the tabs.
    # In 0.17 we can instead scope each viewer to its own tab to clean up these examples further.
    with gr.Row():
        viewer = Rerun(
            streaming=True,
            height=800,
        )

    warp_img_btn.click(
        gradio_warped_image,
        inputs=[img, num_iters, cam_direction, degrees_per_frame],
        outputs=[viewer, video_output, image_files_output, iteration_num],
    )

    gr.Examples(
        [
            [
                "examples/000001.jpg",
            ],
        ],
        fn=warp_img_btn,
        inputs=[img, num_iters, cam_direction, degrees_per_frame],
        outputs=[viewer, video_output, image_files_output],
    )


if __name__ == "__main__":
    demo.queue().launch()
