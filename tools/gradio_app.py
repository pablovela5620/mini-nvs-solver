import PIL.Image
import cv2
import os

import gradio as gr
from gradio_rerun import Rerun

import rerun as rr
import rerun.blueprint as rrb

import numpy as np
import PIL
import torch
from pathlib import Path
import threading
import mmcv
from queue import SimpleQueue
import calibur
import trimesh

from typing import Final, assert_never, Literal

from jaxtyping import Float64, Float32, Bool
import json
from dataclasses import asdict

from monopriors.relative_depth_models import (
    get_relative_predictor,
    RelativeDepthPrediction,
)

from src.pose_utils import generate_camera_poses_around_ellipse
from src.image_warping import forward_warp
from src.sigma_utils import search_hypers
from src.custom_diffusers_pipeline.svd import StableVideoDiffusionPipeline
from src.custom_diffusers_pipeline.scheduler import EulerDiscreteScheduler
from src.depth_utils import depth_to_points, depth_edges_mask
from src.nerfstudio_data import FrameColmap, NerfStudioColmapData

from einops import rearrange

SVD_HEIGHT: Final[int] = 576
SVD_WIDTH: Final[int] = 1024

if gr.NO_RELOAD:
    DepthAnythingV2Predictor = get_relative_predictor("DepthAnythingV2Predictor")(
        device="cuda"
    )
    SVD_PIPE = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        torch_dtype=torch.float16,
        variant="fp16",
    )
    SVD_PIPE.to("cuda")
    scheduler = EulerDiscreteScheduler.from_config(SVD_PIPE.scheduler.config)
    SVD_PIPE.scheduler = scheduler


def svd_render_threaded(
    image_o: PIL.Image.Image,
    masks: Float64[torch.Tensor, "b 72 128"],
    cond_image: PIL.Image.Image,
    lambda_ts: Float64[torch.Tensor, "n b"],
    num_denoise_iters: Literal[25, 50, 100],
    weight_clamp: float,
    log_queue: SimpleQueue | None = None,
):
    frames: list[PIL.Image.Image] = SVD_PIPE(
        [image_o],
        log_queue=log_queue,
        temp_cond=cond_image,
        mask=masks,
        lambda_ts=lambda_ts,
        weight_clamp=weight_clamp,
        num_frames=25,
        decode_chunk_size=8,
        num_inference_steps=num_denoise_iters,
    ).frames[0]

    log_queue.put(frames)


@rr.thread_local_stream("warped_image")
def gradio_warped_image(
    image_path: str,
    num_denoise_iters: Literal[25, 50, 100],
    direction: Literal["left", "right"],
    degrees_per_frame: int | float,
    save_path: str,
    major_radius: float = 60.0,
    minor_radius: float = 70.0,
    num_frames: int = 25,  # StableDiffusion Video generates 25 frames
):
    # ensure that the degrees per frame is a float
    degrees_per_frame = float(degrees_per_frame)
    # convert save_path from string to Path
    image_path_name = Path(image_path).stem
    save_path: Path = Path(save_path) / image_path_name
    if not save_path.exists():
        save_path.mkdir(parents=True)
    stream = rr.binary_stream()

    parent_log_path = Path("world")

    # create blueprint
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                origin=f"{parent_log_path}",
            ),
            rrb.Vertical(
                rrb.Spatial2DView(
                    origin=f"{parent_log_path}/generated_cam/image/rgb",
                ),
                rrb.Spatial2DView(
                    origin=f"{parent_log_path}/generated_new/image/rgb",
                ),
            ),
            rrb.Vertical(
                rrb.Spatial2DView(
                    origin=f"{parent_log_path}/initial_cam/image",
                    contents=[
                        "+ $origin/**",
                    ],
                ),
                rrb.TensorView(origin="latents"),
            ),
            column_shares=[2, 1, 1],
        ),
        collapse_panels=True,
    )
    rr.send_blueprint(blueprint)
    cam_log_path = parent_log_path / "initial_cam"

    rr.log(f"{parent_log_path}", rr.ViewCoordinates.LDB, timeless=True)
    rr.set_time_sequence("iteration", 0)

    inverse = True if direction == "right" else False
    cam_T_world_list: list[Float64[np.ndarray, "4 4"]] = (
        generate_camera_poses_around_ellipse(
            num_frames, degrees_per_frame, major_radius, minor_radius, inverse=inverse
        )
    )
    near = 0.0001
    far = 500.0
    focal = 260.0
    K = np.eye(3)
    K[0, 0] = focal
    K[1, 1] = focal
    K[0, 2] = SVD_WIDTH / 2
    K[1, 2] = SVD_HEIGHT / 2

    assert os.path.exists(image_path)
    rgb_o = PIL.Image.open(image_path)
    original_width, original_height = rgb_o.size
    rgb_o = rgb_o.resize((SVD_WIDTH, SVD_HEIGHT), PIL.Image.Resampling.NEAREST)

    rr.log(f"{cam_log_path}/image/rgb", rr.Image(rgb_o).compress(jpeg_quality=80))

    relative_pred: RelativeDepthPrediction = DepthAnythingV2Predictor.__call__(
        np.array(PIL.Image.open(image_path)), K_33=K.astype(np.float32)
    )
    image = np.array(rgb_o)
    disparity: Float32[np.ndarray, "h w"] = relative_pred.disparity

    disparity[disparity < 1e-5] = 1e-5
    depth = 10000.0 / disparity
    depth = np.clip(depth, near, far)

    new_width, new_height = SVD_WIDTH, SVD_HEIGHT
    # Resize the depth map to the Stable Diffusion Video dimensions
    depth = cv2.resize(depth, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    rr.log(f"{cam_log_path}/image/depth", rr.DepthImage(depth))

    # remove flying pixels
    # edges_mask: Bool[np.ndarray, "h w"] = depth_edges_mask(depth, threshold=0.1)
    # masked_depth: Float32[np.ndarray, "h w"] = depth * ~edges_mask

    depth_1hw: Float32[np.ndarray, "1 h w"] = rearrange(depth, "h w -> 1 h w")
    pts_3d: Float32[np.ndarray, "h w 3"] = depth_to_points(
        depth_1hw,
        relative_pred.K_33,
        R=cam_T_world_list[0][:3, :3].astype(np.float32),
        t=cam_T_world_list[0][:3, 3].astype(np.float32),
    )

    trimesh_pointcloud = trimesh.PointCloud(
        vertices=pts_3d.reshape(-1, 3), colors=image.reshape(-1, 3)
    )

    rr.log(
        f"{parent_log_path}/point_cloud",
        rr.Points3D(
            positions=trimesh_pointcloud.vertices,
            colors=trimesh_pointcloud.colors,
        ),
    )

    cam_T_world_44_s = cam_T_world_list[0]
    cond_image: list[PIL.Image.Image] = []
    masks: list[Float64[torch.Tensor, "1 72 128"]] = []

    rr.log(
        f"{cam_log_path}/image",
        rr.Pinhole(
            image_from_camera=K,
            width=SVD_WIDTH,
            height=SVD_HEIGHT,
            camera_xyz=rr.ViewCoordinates.RDF,
        ),
    )
    rr.log(
        f"{cam_log_path}",
        rr.Transform3D(
            mat3x3=cam_T_world_44_s[:3, :3],
            translation=cam_T_world_44_s[:3, 3],
            from_parent=True,
        ),
    )

    yield stream.read(), None

    i = 0
    for cam_T_world_44_t in cam_T_world_list[1:]:
        cam_log_path = parent_log_path / "generated_cam"
        rr.set_time_sequence("iteration", i + 1)
        warped_frame2, mask2, _ = forward_warp(
            image, None, depth, cam_T_world_44_s, cam_T_world_44_t, K, None
        )

        rr.log(
            f"{cam_log_path}/image/rgb",
            rr.Image(np.uint8(warped_frame2)).compress(jpeg_quality=10),
        )
        rr.log(
            f"{cam_log_path}/image",
            rr.Pinhole(image_from_camera=K, width=SVD_WIDTH, height=SVD_HEIGHT),
        )
        rr.log(
            f"{cam_log_path}",
            rr.Transform3D(
                mat3x3=cam_T_world_44_t[:3, :3],
                translation=cam_T_world_44_t[:3, 3],
                from_parent=True,
            ),
        )

        mask = 1 - mask2
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask: Float64[np.ndarray, "h w 3"] = np.repeat(
            mask[:, :, np.newaxis] * 255.0, repeats=3, axis=2
        )

        kernel = np.ones((5, 5), np.uint8)
        mask_erosion = cv2.dilate(np.array(mask), kernel, iterations=1)
        mask_erosion = PIL.Image.fromarray(np.uint8(mask_erosion))

        mask_erosion_ = np.array(mask_erosion) / 255.0
        mask_erosion_[mask_erosion_ < 0.5] = 0
        mask_erosion_[mask_erosion_ >= 0.5] = 1
        warped_frame2 = PIL.Image.fromarray(np.uint8(warped_frame2))
        # perform masking on the warped image
        warped_frame2 = PIL.Image.fromarray(
            np.uint8(warped_frame2 * (1 - mask_erosion_))
        )

        cond_image.append(warped_frame2)

        mask_erosion = np.mean(mask_erosion_, axis=-1)
        mask_erosion = (
            mask_erosion.reshape(72, 8, 128, 8)
            .transpose(0, 2, 1, 3)
            .reshape(72, 128, 64)
        )
        mask_erosion = np.mean(mask_erosion, axis=2)
        mask_erosion[mask_erosion < 0.2] = 0
        mask_erosion[mask_erosion >= 0.2] = 1
        mask_erosion_tensor = torch.from_numpy(mask_erosion)

        masks.append(rearrange(mask_erosion_tensor, "h w -> 1 h w"))
        i += 1
        yield stream.read(), None
    masks: Float64[torch.Tensor, "b 72 128"] = torch.cat(masks)

    # create a simple queue to log the latents during diffusion
    log_queue: SimpleQueue = SimpleQueue()
    # load sigmas
    if num_denoise_iters == 25:
        sigma_list: list[float] = np.load("data/sigmas/sigmas_25.npy").tolist()
    elif num_denoise_iters == 50:
        sigma_list: list[float] = np.load("data/sigmas/sigmas_50.npy").tolist()
    elif num_denoise_iters == 100:
        sigma_list: list[float] = np.load("data/sigmas/sigmas_100.npy").tolist()
    lambda_ts: Float64[torch.Tensor, "n b"] = search_hypers(sigma_list)

    handle = threading.Thread(
        target=svd_render_threaded,
        kwargs={
            "image_o": rgb_o,
            "masks": masks,
            "cond_image": cond_image,
            "lambda_ts": lambda_ts,
            "num_denoise_iters": num_denoise_iters,
            "weight_clamp": 0.2,
            "log_queue": log_queue,
        },
    )

    handle.start()
    while True:
        msg = log_queue.get()
        match msg:
            case frames if all(isinstance(frame, PIL.Image.Image) for frame in frames):
                break
            case entity_path, entity, times:
                rr.reset_time()
                for timeline, time in times:
                    if isinstance(time, int):
                        rr.set_time_sequence(timeline, time)
                    else:
                        rr.set_time_seconds(timeline, time)
                rr.log(entity_path, entity, static=True)
                yield stream.read(), None
            case _:
                assert_never(msg)

    handle.join()

    # restart to generate set of frames in the same timeline
    i = 0
    # setup things for saving the json file
    frames_list: list[FrameColmap] = []
    for frame, cam_T_world_44_cv in zip(frames, cam_T_world_list):
        cam_log_path = parent_log_path / "generated_new"
        generated_rgb_np = np.array(frame)

        world_T_cam_44_cv = np.linalg.inv(cam_T_world_44_cv)
        world_T_cam_44_gl = calibur.convert_pose(
            world_T_cam_44_cv,
            src_convention=calibur.CC.CV,
            dst_convention=calibur.CC.GL,
        )

        rr.set_time_sequence("iteration", i + 1)
        rr.log(
            f"{cam_log_path}/image/rgb",
            rr.Image(generated_rgb_np).compress(jpeg_quality=10),
        )
        rr.log(
            f"{cam_log_path}/image",
            rr.Pinhole(
                image_from_camera=K,
                width=SVD_WIDTH,
                height=SVD_HEIGHT,
                camera_xyz=rr.ViewCoordinates.RUB,
            ),
        )
        rr.log(
            f"{cam_log_path}",
            rr.Transform3D(
                mat3x3=world_T_cam_44_gl[:3, :3],
                translation=world_T_cam_44_gl[:3, 3],
                from_parent=False,
            ),
        )

        # save the generated frames
        gframe_save_path = f"{save_path}/{i:06}.jpg"
        save_bgr_img = mmcv.rgb2bgr(generated_rgb_np)
        save_bgr_img = mmcv.imresize(save_bgr_img, (original_width, original_height))
        mmcv.imwrite(save_bgr_img, str(gframe_save_path))

        frames_list.append(
            FrameColmap(
                file_path=f"{i:06}.jpg",
                transform_matrix=world_T_cam_44_gl.tolist(),
                colmap_im_id=i,
            )
        )

        i += 1
        yield stream.read(), None

    # save video file
    mmcv.frames2video(str(save_path), str(save_path / "output.mp4"), fps=7)
    # redundant log so that yield stream.read does not throw an error
    rr.log(
        f"{cam_log_path}/image/rgb",
        rr.Image(generated_rgb_np).compress(jpeg_quality=10),
    )

    # save point cloud
    ply_data = trimesh_pointcloud.export(file_type="ply")
    ply_file_path = f"{save_path}/pointcloud.ply"
    with open(ply_file_path, "wb") as ply_file:
        ply_file.write(ply_data)

    # save the json file
    nerf_data = NerfStudioColmapData(
        w=original_width,
        h=original_height,
        fl_x=focal,
        fl_y=focal,
        cx=original_width / 2,
        cy=original_height / 2,
        k1=0.0,
        k2=0.0,
        p1=0.0,
        p2=0.0,
        camera_model="OPENCV",
        ply_file_path="pointcloud.ply",
        frames=frames_list,
    )
    # save to json
    # Convert the data to a dictionary
    nerf_data_dict = asdict(nerf_data)
    # Convert the frames list to a list of dictionaries
    nerf_data_dict["frames"] = [asdict(frame) for frame in nerf_data.frames]

    with open(save_path / "transforms.json", "w") as f:
        json.dump(nerf_data_dict, f)
    yield stream.read(), str(save_path / "output.mp4")


with gr.Blocks() as demo:
    with gr.Tab("Streaming"):
        with gr.Row():
            img = gr.Image(interactive=True, label="Image", type="filepath")
            with gr.Tab(label="Settings"):
                with gr.Column():
                    warp_img_btn = gr.Button("Warp Images")
                    save_path = gr.Textbox(
                        label="Data Save Path",
                        value="data/gradio_outputs",
                    )
                    num_iters = gr.Radio(
                        choices=[25, 50, 100],
                        value=25,
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
            with gr.Tab(label="Outputs"):
                video_output = gr.Video(interactive=False)

    # Rerun 0.16 has issues when embedded in a Gradio tab, so we share a viewer between all the tabs.
    # In 0.17 we can instead scope each viewer to its own tab to clean up these examples further.
    with gr.Row():
        viewer = Rerun(
            streaming=True,
        )

    warp_img_btn.click(
        gradio_warped_image,
        inputs=[img, num_iters, cam_direction, degrees_per_frame, save_path],
        outputs=[viewer, video_output],
    )

    gr.Examples(
        [
            [
                "/home/pablo/0Dev/docker/.per/repos/NVS_Solver/example_imgs/single/000001.jpg"
            ]
        ],
        fn=warp_img_btn,
        inputs=[img, num_iters, cam_direction, degrees_per_frame],
        outputs=[viewer, video_output],
    )


if __name__ == "__main__":
    demo.launch()
