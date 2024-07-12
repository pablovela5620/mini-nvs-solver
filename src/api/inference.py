import cv2
import torch
import PIL.Image
import numpy as np
from pathlib import Path
from typing import Literal, Final
from jaxtyping import Float64, Float32, UInt8
from src.pose_utils import generate_camera_poses_around_ellipse
from src.image_warping import forward_warp
from src.sigma_utils import load_lambda_ts
from src.camera_parameters import Intrinsics, Extrinsics, PinholeParameters
from src.depth_utils import depth_to_points
import rerun as rr
import PIL
from PIL.Image import Image
from monopriors.relative_depth_models import (
    get_relative_predictor,
    RelativeDepthPrediction,
    BaseRelativePredictor,
)

from queue import SimpleQueue
from src.custom_diffusers_pipeline.svd import StableVideoDiffusionPipeline
from src.custom_diffusers_pipeline.scheduler import EulerDiscreteScheduler

import trimesh
from einops import rearrange

DepthAnythingV2Predictor: BaseRelativePredictor = get_relative_predictor(
    "DepthAnythingV2Predictor"
)(device="cuda")
SVD_PIPE = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    torch_dtype=torch.float16,
    variant="fp16",
)
SVD_PIPE.to("cuda")
scheduler = EulerDiscreteScheduler.from_config(SVD_PIPE.scheduler.config)
SVD_PIPE.scheduler = scheduler

SVD_HEIGHT: Final[int] = 576
SVD_WIDTH: Final[int] = 1024


def generate_camera_parameters(
    num_frames: int,
    image_width: int,
    image_height: int,
    degrees_per_frame: int | float,
    major_radius: float,
    minor_radius: float,
    direction: Literal["left", "right"] = "right",
):
    inverse = True if direction == "right" else False
    cam_T_world_list: list[Float64[np.ndarray, "4 4"]] = (
        generate_camera_poses_around_ellipse(
            num_frames, degrees_per_frame, major_radius, minor_radius, inverse=inverse
        )
    )
    near = 0.0001
    far = 500.0

    intri = Intrinsics(
        camera_conventions="RDF",
        fl_x=260.0,
        fl_y=260.0,
        cx=image_width / 2,
        cy=image_height / 2,
        height=image_height,
        width=image_width,
    )

    camera_list: list[PinholeParameters] = []
    for id, cam_T_world_44 in enumerate(cam_T_world_list):
        cam_R_world = cam_T_world_44[:3, :3]
        cam_t_world = cam_T_world_44[:3, 3]
        extri = Extrinsics(cam_R_world=cam_R_world, cam_t_world=cam_t_world)
        pinhole_params = PinholeParameters(
            name=f"camera_{id}", extrinsics=extri, intrinsics=intri
        )
        camera_list.append(pinhole_params)

    return camera_list, near, far


def rescale_intri(
    camera_intrinsics: Intrinsics, new_width: int, new_height: int
) -> Intrinsics:
    """
    Rescales the input image and intrinsic matrix by a given scale factor.

    Args:
        cam (PinholeCameraParameter): The pinhole camera parameter.

    Returns:
        : The rescaled image frame and intrinsic matrix.
    """
    x_scale: float = new_width / camera_intrinsics.width
    y_scale: float = new_height / camera_intrinsics.height
    rescaled_intri = Intrinsics(
        camera_conventions=camera_intrinsics.camera_conventions,
        fl_x=camera_intrinsics.fl_x * x_scale,
        fl_y=camera_intrinsics.fl_y * y_scale,
        cx=camera_intrinsics.cx * x_scale,
        cy=camera_intrinsics.cy * y_scale,
        height=new_height,
        width=new_width,
    )

    return rescaled_intri


def image_to_depth(
    rgb_np_original: UInt8[np.ndarray, "original_h original_w 3"],
    rgb_np_hw3: UInt8[np.ndarray, "h w 3"],
    cam_params: PinholeParameters,
    near: float,
    far: float,
):
    original_width, original_height, _ = rgb_np_original.shape
    # resize the K matrix from SVD to original image dimensions
    resized_intrinsics: Intrinsics = rescale_intri(
        cam_params.intrinsics, original_width, original_height
    )
    relative_pred: RelativeDepthPrediction = DepthAnythingV2Predictor.__call__(
        rgb_np_original, K_33=resized_intrinsics.k_matrix.astype(np.float32)
    )
    disparity: Float32[np.ndarray, "original_h original_w"] = relative_pred.disparity
    disparity[disparity < 1e-5] = 1e-5
    depth: Float32[np.ndarray, "original_h original_w"] = 10000.0 / disparity
    depth: Float32[np.ndarray, "original_h original_w"] = np.clip(depth, near, far)

    # Resize the depth map to the Stable Diffusion Video dimensions
    depth: Float32[np.ndarray, "h w"] = cv2.resize(
        depth, (SVD_WIDTH, SVD_HEIGHT), interpolation=cv2.INTER_NEAREST
    )

    depth_1hw: Float32[np.ndarray, "1 h w"] = rearrange(depth, "h w -> 1 h w")
    pts_3d: Float32[np.ndarray, "h w 3"] = depth_to_points(
        depth_1hw,
        cam_params.intrinsics.k_matrix.astype(np.float32),
        R=cam_params.extrinsics.cam_R_world.astype(np.float32),
        t=cam_params.extrinsics.cam_t_world.astype(np.float32),
    )

    trimesh_pointcloud = trimesh.PointCloud(
        vertices=pts_3d.reshape(-1, 3), colors=rgb_np_hw3.reshape(-1, 3)
    )

    return depth, trimesh_pointcloud


def image_depth_warping(image, depth, cam_T_world_44_s, cam_T_world_44_t, K):
    warped_frame2, mask2, _ = forward_warp(
        image, None, depth, cam_T_world_44_s, cam_T_world_44_t, K, None
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
    warped_frame2 = PIL.Image.fromarray(np.uint8(warped_frame2 * (1 - mask_erosion_)))

    mask_erosion = np.mean(mask_erosion_, axis=-1)
    mask_erosion = (
        mask_erosion.reshape(72, 8, 128, 8).transpose(0, 2, 1, 3).reshape(72, 128, 64)
    )
    mask_erosion = np.mean(mask_erosion, axis=2)
    mask_erosion[mask_erosion < 0.2] = 0
    mask_erosion[mask_erosion >= 0.2] = 1
    mask_erosion_tensor = torch.from_numpy(mask_erosion)
    mask_erosion_tensor = rearrange(mask_erosion_tensor, "h w -> 1 h w")

    return warped_frame2, mask_erosion_tensor


def nvs_solver_inference(
    image_path: str | Path,
    num_denoise_iters: Literal[2, 25, 50, 100],
    direction: Literal["left", "right"],
    degrees_per_frame: int | float,
    major_radius: float = 60.0,
    minor_radius: float = 70.0,
    weight_clamp: float = 0.2,
    num_frames: int = 25,  # StableDiffusion Video generates 25 frames
):
    image_path: Path = Path(image_path) if isinstance(image_path, str) else image_path
    assert image_path.exists(), f"Image file not found: {image_path}"

    # setup rerun logging
    parent_log_path = Path("world")
    rr.log(f"{parent_log_path}", rr.ViewCoordinates.LDB, static=True)

    # generate initial camera parameters for video trajectory
    camera_list, near, far = generate_camera_parameters(
        num_frames=num_frames,
        image_width=SVD_WIDTH,
        image_height=SVD_HEIGHT,
        degrees_per_frame=degrees_per_frame,
        major_radius=major_radius,
        minor_radius=minor_radius,
        direction=direction,
    )
    assert len(camera_list) == num_frames, "Number of camera parameters mismatch"

    # Load image and resize to SVD dimensions
    rgb_original: Image = PIL.Image.open(image_path)
    rgb_resized: Image = rgb_original.resize(
        (SVD_WIDTH, SVD_HEIGHT), PIL.Image.Resampling.NEAREST
    )
    rgb_np_original: UInt8[np.ndarray, "h w 3"] = np.array(rgb_original)
    rgb_np_hw3: UInt8[np.ndarray, "h w 3"] = np.array(rgb_resized)

    # Estimate depth map and pointcloud for the input image
    depth: Float32[np.ndarray, "h w"]
    trimesh_pointcloud: trimesh.PointCloud

    depth, trimesh_pointcloud = image_to_depth(
        rgb_np_original=rgb_np_original,
        rgb_np_hw3=rgb_np_hw3,
        cam_params=camera_list[0],
        near=near,
        far=far,
    )

    rr.log(
        f"{parent_log_path}/point_cloud",
        rr.Points3D(
            positions=trimesh_pointcloud.vertices,
            colors=trimesh_pointcloud.colors,
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
            cam_log_path: Path = parent_log_path / "original_camera"
            log_camera(cam_log_path, current_cam, rgb_np_hw3, depth)
        else:
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
    frame: PIL.Image.Image
    for frame_id, (frame, cam_pararms) in enumerate(zip(frames, camera_list[1:])):
        # add one since the first frame is the original image
        rr.set_time_sequence("frame_id", frame_id + 1)
        cam_log_path = parent_log_path / "generated_camera"
        generated_rgb_np: UInt8[np.ndarray, "h w 3"] = np.array(frame)
        log_camera(cam_log_path, cam_pararms, generated_rgb_np, depth=None)


def log_camera(
    cam_log_path: Path,
    cam: PinholeParameters,
    rgb_np_hw3: UInt8[np.ndarray, "h w 3"],
    depth: Float32[np.ndarray, "h w"] | None = None,
) -> None:
    pinhole_log_path: Path = cam_log_path / "pinhole"
    rr.log(
        f"{cam_log_path}",
        rr.Transform3D(
            mat3x3=cam.extrinsics.cam_R_world,
            translation=cam.extrinsics.cam_t_world,
            from_parent=True,
        ),
    )
    rr.log(
        f"{pinhole_log_path}",
        rr.Pinhole(
            image_from_camera=cam.intrinsics.k_matrix,
            width=cam.intrinsics.width,
            height=cam.intrinsics.height,
            camera_xyz=rr.ViewCoordinates.RDF,
        ),
    )
    rr.log(f"{pinhole_log_path}/rgb", rr.Image(rgb_np_hw3).compress(jpeg_quality=80))
    if depth is not None:
        rr.log(
            f"{pinhole_log_path}/depth",
            rr.DepthImage(depth),
        )
