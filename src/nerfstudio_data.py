import numpy as np
from dataclasses import dataclass, field, asdict
import json
from pathlib import Path
import PIL
import PIL.Image
import trimesh
import calibur
import mmcv
from jaxtyping import UInt8
from src.camera_parameters import (
    PinholeParameters,
    Intrinsics,
    Extrinsics,
    rescale_intri,
)
import rerun as rr
from src.rr_logging_utils import log_camera


@dataclass
class FrameColmap:
    file_path: str
    transform_matrix: list[list[float]]
    colmap_im_id: int


@dataclass
class NerfStudioColmapData:
    w: int
    h: int
    fl_x: float
    fl_y: float
    cx: float
    cy: float
    k1: float
    k2: float
    p1: float
    p2: float
    camera_model: str
    ply_file_path: str
    frames: list[FrameColmap] = field(default_factory=list)
    applied_transform: list[list[float]] = field(
        default_factory=lambda: [[0.0] * 4 for _ in range(3)]
    )


def frames_to_nerfstudio(
    rgb_np_original: UInt8[np.ndarray, "original_h original_w 3"],
    frames: list[PIL.Image.Image],
    trimesh_pc_original: trimesh.PointCloud,
    camera_list: list[PinholeParameters],
    save_path: Path | None = None,
) -> None:
    assert len(frames) == len(camera_list) == 25, "Not 25 frames"
    if save_path is not None:
        save_path.mkdir(parents=True, exist_ok=True)
    # create a new path for the final images/cameras
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
        original_intri.camera_conventions = "RUB"
        extrinsics_gl = Extrinsics(
            world_R_cam=world_T_cam_44_gl[:3, :3],
            world_t_cam=world_T_cam_44_gl[:3, 3],
        )

        log_camera(
            cam_log_path,
            PinholeParameters(
                name=str(frame_id), extrinsics=extrinsics_gl, intrinsics=original_intri
            ),
            generated_rgb_np,
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
        # Convert the data to a dictionary
        nerf_data_dict = asdict(nerf_data)
        # Convert the frames list to a list of dictionaries
        nerf_data_dict["frames"] = [asdict(frame) for frame in nerf_data.frames]

        with open(save_path / "transforms.json", "w") as f:
            json.dump(nerf_data_dict, f)
