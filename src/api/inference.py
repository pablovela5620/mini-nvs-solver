import torch
import mmcv
import numpy as np
from pathlib import Path
from typing import Literal, Final
from jaxtyping import Float64, Float32, UInt8
from src.pose_utils import generate_camera_parameters
from src.image_warping import image_depth_warping
from src.sigma_utils import load_lambda_ts
from src.camera_parameters import (
    PinholeParameters,
    rescale_intri,
    Intrinsics,
)

from src.depth_utils import image_to_depth
from src.rr_logging_utils import (
    log_camera,
    create_svd_blueprint,
)
from src.nerfstudio_data import FrameColmap, NerfStudioColmapData
import rerun as rr
import rerun.blueprint as rrb
import PIL
import PIL.Image
from PIL.Image import Image
from monopriors.relative_depth_models import (
    get_relative_predictor,
    BaseRelativePredictor,
)

from src.custom_diffusers_pipeline.svd import StableVideoDiffusionPipeline
from src.custom_diffusers_pipeline.scheduler import EulerDiscreteScheduler

import trimesh
import calibur
import json
from dataclasses import asdict

from icecream import ic

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
NEAR: Final[float] = 0.0001
FAR: Final[float] = 500.0


def to_nerfstudio_camera(
    rgb_np_original: UInt8[np.ndarray, "original_h original_w 3"],
    frames: list[PIL.Image.Image],
    trimesh_pc_original: trimesh.PointCloud,
    camera_list: list[PinholeParameters],
    save_path: Path | None = None,
) -> None:
    assert len(frames) == len(camera_list) == 25, "Not 25 frames"
    if save_path is not None:
        save_path.mkdir(parents=True, exist_ok=True)
    # send a new blueprint for nerf studio
    parent_log_path = Path("nerfstudio")
    rr.log(f"{parent_log_path}", rr.ViewCoordinates.LDB, static=True)
    rr.log(
        f"{parent_log_path}/point_cloud",
        rr.Points3D(
            positions=trimesh_pc_original.vertices,
            colors=trimesh_pc_original.colors,
        ),
        static=True,
    )

    # resize all frames to original size
    original_height, original_width, _ = rgb_np_original.shape
    frames: list[PIL.Image.Image] = [
        frame.resize(
            (original_width, original_height),
            PIL.Image.Resampling.BILINEAR,
        )
        for frame in frames
    ]

    cam_log_path = parent_log_path / "original_resolution_camera"
    frames_colmap_list: list[FrameColmap] = []
    for frame_id, (frame, cam_params) in enumerate(zip(frames, camera_list)):
        rr.set_time_sequence("frame_id", frame_id)
        original_intri: Intrinsics = rescale_intri(
            cam_params.intrinsics,
            target_width=original_width,
            target_height=original_height,
        )
        generated_rgb_np: UInt8[np.ndarray, "h w 3"] = np.array(frame)
        # nerfstudio uses GL convention and cam to world
        assert cam_params.intrinsics.camera_conventions == "RDF"
        world_T_cam_44_cv = cam_params.extrinsics.world_T_cam
        world_T_cam_44_gl = calibur.convert_pose(
            world_T_cam_44_cv,
            src_convention=calibur.CC.CV,
            dst_convention=calibur.CC.GL,
        )

        # save the generated frames
        if save_path is not None:
            gframe_save_path = f"{save_path}/{frame_id:06}.jpg"
            save_bgr_img = mmcv.rgb2bgr(generated_rgb_np)
            mmcv.imwrite(save_bgr_img, str(gframe_save_path))

            frames_colmap_list.append(
                FrameColmap(
                    file_path=f"{frame_id:06}.jpg",
                    transform_matrix=world_T_cam_44_gl.tolist(),
                    colmap_im_id=frame_id,
                )
            )

        pinhole_log_path: Path = cam_log_path / "pinhole"
        rr.log(
            f"{cam_log_path}",
            rr.Transform3D(
                mat3x3=world_T_cam_44_gl[:3, :3],
                translation=world_T_cam_44_gl[:3, 3],
                from_parent=False,
            ),
        )
        rr.log(
            f"{pinhole_log_path}",
            rr.Pinhole(
                image_from_camera=original_intri.k_matrix,
                width=original_intri.width,
                height=original_intri.height,
                camera_xyz=rr.ViewCoordinates.RUB,
            ),
        )
        rr.log(
            f"{pinhole_log_path}/rgb",
            rr.Image(generated_rgb_np).compress(jpeg_quality=80),
        )

    if save_path is not None:
        # save point cloud
        ply_data = trimesh_pc_original.export(file_type="ply")
        ply_file_path = save_path / "pointcloud.ply"
        with open(ply_file_path, "wb") as ply_file:
            ply_file.write(ply_data)

        # save the json file
        nerf_data = NerfStudioColmapData(
            w=original_intri.width,
            h=original_intri.height,
            fl_x=original_intri.fl_x,
            fl_y=original_intri.fl_y,
            cx=original_intri.cx,
            cy=original_intri.cy,
            k1=0.0,
            k2=0.0,
            p1=0.0,
            p2=0.0,
            camera_model="OPENCV",
            ply_file_path=ply_file_path.name,
            frames=frames_colmap_list,
        )
        # save to json
        # Convert the data to a dictionary
        nerf_data_dict = asdict(nerf_data)
        # Convert the frames list to a list of dictionaries
        nerf_data_dict["frames"] = [asdict(frame) for frame in nerf_data.frames]

        with open(save_path / "transforms.json", "w") as f:
            json.dump(nerf_data_dict, f)


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
        depth_predictor=DepthAnythingV2Predictor,
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

    to_nerfstudio_camera(
        rgb_np_original, frames, trimesh_pc_original, camera_list, save_path
    )
