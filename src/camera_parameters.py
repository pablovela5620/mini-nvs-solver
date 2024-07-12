from dataclasses import dataclass, field
import numpy as np
from jaxtyping import Float
from typing import Literal


@dataclass
class Distortion:
    k1: float
    k2: float
    p1: float
    p2: float
    k3: float


@dataclass
class Extrinsics:
    # Rotation and translation can be provided for both world-to-camera and camera-to-world transformations
    world_R_cam: Float[np.ndarray, "3 3"] | None = None
    world_t_cam: Float[np.ndarray, "3"] | None = None
    cam_R_world: Float[np.ndarray, "3 3"] | None = None
    cam_t_world: Float[np.ndarray, "3"] | None = None
    # The projection matrix and transformation matrices will be computed in post-init
    world_T_cam: Float[np.ndarray, "4 4"] = field(init=False)
    cam_T_world: Float[np.ndarray, "4 4"] = field(init=False)

    def __post_init__(self):
        self.compute_transformation_matrices()

    def compute_transformation_matrices(self) -> None:
        # If world-to-camera is provided, compute the transformation matrix and its inverse
        if self.world_R_cam is not None and self.world_t_cam is not None:
            self.world_T_cam = self.compose_transformation_matrix(
                self.world_R_cam, self.world_t_cam
            )
            self.cam_T_world = np.linalg.inv(self.world_T_cam)
            # Extract camera-to-world rotation and translation from the inverse matrix
            self.cam_R_world, self.cam_t_world = self.decompose_transformation_matrix(
                self.cam_T_world
            )
        # If camera-to-world is provided, compute the transformation matrix and its inverse
        elif self.cam_R_world is not None and self.cam_t_world is not None:
            self.cam_T_world = self.compose_transformation_matrix(
                self.cam_R_world, self.cam_t_world
            )
            self.world_T_cam = np.linalg.inv(self.cam_T_world)
            # Extract world-to-camera rotation and translation from the inverse matrix
            self.world_R_cam, self.world_t_cam = self.decompose_transformation_matrix(
                self.world_T_cam
            )
        else:
            raise ValueError(
                "Either world-to-camera or camera-to-world rotation and translation must be provided."
            )

    def compose_transformation_matrix(
        self, R: Float[np.ndarray, "3 3"], t: Float[np.ndarray, "3"]
    ) -> Float[np.ndarray, "4 4"]:
        return np.vstack([np.hstack([R, t.reshape(-1, 1)]), np.array([0, 0, 0, 1])])

    def decompose_transformation_matrix(
        self, T: Float[np.ndarray, "4 4"]
    ) -> tuple[Float[np.ndarray, "3 3"], Float[np.ndarray, "3"]]:
        R: Float[np.ndarray, "3 3"] = T[:3, :3]
        t: Float[np.ndarray, "..."] = T[:3, 3]
        return R, t


@dataclass
class Intrinsics:
    camera_conventions: Literal["RDF", "RUB"]
    """RDF(OpenCV): X Right - Y Down - Z Front | RUB (OpenGL): X Right- Y Up - Z Back"""
    fl_x: float
    fl_y: float
    cx: float
    cy: float
    height: int
    width: int
    k_matrix: Float[np.ndarray, "3 3"] = field(init=False)

    def __post_init__(self):
        self.compute_k_matrix()

    def compute_k_matrix(self):
        # Compute the camera matrix using the focal length and principal point
        self.k_matrix = np.array(
            [[self.fl_x, 0, self.cx], [0, self.fl_y, self.cy], [0, 0, 1]]
        )

    def __repr__(self):
        return (
            f"Intrinsics(camera_conventions={self.camera_conventions}, "
            f"fl_x={self.fl_x}, fl_y={self.fl_y}, cx={self.cx}, cy={self.cy}, "
            f"height={self.height}, width={self.width})"
        )


@dataclass
class PinholeParameters:
    name: str
    extrinsics: Extrinsics
    intrinsics: Intrinsics
    projection_matrix: Float[np.ndarray, "3 4"] = field(init=False)
    distortion: Distortion | None = None

    def __post_init__(self):
        self.compute_projection_matrix()

    def compute_projection_matrix(self):
        # Compute the projection matrix using k_matrix and world_T_cam
        self.projection_matrix = (
            self.intrinsics.k_matrix @ self.extrinsics.cam_T_world[:3, :]
        )


def rescale_intri(
    camera_intrinsics: Intrinsics, *, target_width: int, target_height: int
) -> Intrinsics:
    """
    Rescales the input image and intrinsic matrix by a given scale factor.

    Args:
        cam (PinholeCameraParameter): The pinhole camera parameter.

    Returns:
        : The rescaled image frame and intrinsic matrix.
    """
    x_scale: float = target_width / camera_intrinsics.width
    y_scale: float = target_height / camera_intrinsics.height

    # assume the focal length is the same for x and y
    focal: float = camera_intrinsics.fl_y * y_scale

    rescaled_intri = Intrinsics(
        camera_conventions=camera_intrinsics.camera_conventions,
        fl_x=focal,
        fl_y=focal,
        cx=camera_intrinsics.cx * x_scale,
        cy=camera_intrinsics.cy * y_scale,
        height=target_height,
        width=target_width,
    )

    return rescaled_intri
