import torch
import numpy as np
from pathlib import Path
from typing import Literal, Final
from jaxtyping import Float64, Float32, UInt8
from mini_nvs_solver.pose_utils import generate_camera_parameters
from mini_nvs_solver.image_warping import image_depth_warping
from mini_nvs_solver.sigma_utils import load_lambda_ts
from mini_nvs_solver.camera_parameters import PinholeParameters
from mini_nvs_solver.nerfstudio_data import frames_to_nerfstudio

from mini_nvs_solver.depth_utils import image_to_depth
from mini_nvs_solver.rr_logging_utils import (
    log_camera,
    create_svd_blueprint,
)
import rerun as rr
import rerun.blueprint as rrb
import PIL
import PIL.Image
from PIL.Image import Image

from monopriors.relative_depth_models.depth_anything_v2 import DepthAnythingV2Predictor
from mini_nvs_solver.custom_diffusers_pipeline.svd import StableVideoDiffusionPipeline
from mini_nvs_solver.custom_diffusers_pipeline.scheduler import EulerDiscreteScheduler

import trimesh

SVD_HEIGHT: Final[int] = 576
SVD_WIDTH: Final[int] = 1024
NEAR: Final[float] = 0.0001
FAR: Final[float] = 500.0


def nvs_solver_inference(
    image_path: str | Path,
    num_denoise_iters: Literal[2, 25, 50, 100],
    direction: Literal["left", "right"],
    degrees_per_frame: int | float,
    major_radius: float = 60.0,
    minor_radius: float = 70.0,
    weight_clamp: float = 0.2,
    num_frames: int = 25,  # StableDiffusion Video generates 25 frames
    save_path: Path | None = None,
) -> None:
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

    image_path: Path = Path(image_path) if isinstance(image_path, str) else image_path
    assert image_path.exists(), f"Image file not found: {image_path}"

    # setup rerun logging
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

    masks: Float64[torch.Tensor, "b 72 128"] = torch.cat(masks)
    # load sigmas to optimize for timestep
    lambda_ts: Float64[torch.Tensor, "n b"] = load_lambda_ts(num_denoise_iters)

    frames: list[PIL.Image.Image] = SVD_PIPE(
        [rgb_resized],
        log_queue=None,
        temp_cond=cond_image,
        mask=masks,
        lambda_ts=lambda_ts,
        weight_clamp=weight_clamp,
        num_frames=25,
        decode_chunk_size=8,
        num_inference_steps=num_denoise_iters,
    ).frames[0]

    # all frames but the first one
    frame: np.ndarray
    for frame_id, (frame, cam_pararms) in enumerate(zip(frames, camera_list)):
        # add one since the first frame is the original image
        rr.set_time_sequence("frame_id", frame_id)
        cam_log_path = parent_log_path / "generated_camera"
        generated_rgb_np: UInt8[np.ndarray, "h w 3"] = np.array(frame)
        log_camera(cam_log_path, cam_pararms, generated_rgb_np, depth=None)

    frames_to_nerfstudio(
        rgb_np_original, frames, trimesh_pc_original, camera_list, save_path
    )
