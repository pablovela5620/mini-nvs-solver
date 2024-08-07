import numpy as np
from jaxtyping import UInt8, Float32
from pathlib import Path
from mini_nvs_solver.camera_parameters import PinholeParameters
import rerun as rr
import rerun.blueprint as rrb


def log_camera(
    cam_log_path: Path,
    cam: PinholeParameters,
    rgb_np_hw3: UInt8[np.ndarray, "h w 3"],
    depth: Float32[np.ndarray, "h w"] | None = None,
    jpeg_quality: int = 50,
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
            camera_xyz=getattr(rr.ViewCoordinates, cam.intrinsics.camera_conventions),
            image_plane_distance=5.0,
        ),
    )
    rr.log(
        f"{pinhole_log_path}/rgb",
        rr.Image(rgb_np_hw3).compress(jpeg_quality=jpeg_quality),
    )
    if depth is not None:
        rr.log(
            f"{pinhole_log_path}/depth",
            rr.DepthImage(depth),
        )


def create_svd_blueprint(parent_log_path: Path) -> rrb.Blueprint:
    nerfstudio_path = Path("nerfstudio")
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Tabs(
                rrb.Spatial3DView(
                    name="3D Projection Original",
                    origin=f"{parent_log_path}",
                ),
                rrb.Spatial3DView(
                    name="3D Projection Nerfstudio",
                    origin=f"{nerfstudio_path}",
                ),
                active_tab="3D Projection Original",
            ),
            rrb.Vertical(
                rrb.Spatial2DView(
                    origin=f"{parent_log_path}/warped_camera/pinhole",
                    contents=[
                        "+ $origin/**",
                    ],
                ),
                rrb.Spatial2DView(
                    origin=f"{nerfstudio_path}/original_resolution_camera/pinhole/rgb",
                ),
            ),
            rrb.Vertical(
                rrb.Spatial2DView(
                    origin=f"{parent_log_path}/generated_camera/pinhole/rgb",
                ),
                rrb.TensorView(origin="latents"),
            ),
            column_shares=[5, 3, 2],
        ),
        collapse_panels=True,
    )
    return blueprint
