from dataclasses import dataclass, field
import numpy as np
from typing import Optional
from jaxtyping import Float


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
    world_R_cam: Optional[Float[np.ndarray, "3 3"]] = None
    world_t_cam: Optional[Float[np.ndarray, "3"]] = None
    cam_R_world: Optional[Float[np.ndarray, "3 3"]] = None
    cam_t_world: Optional[Float[np.ndarray, "3"]] = None
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
        R = T[:3, :3]
        t = T[:3, 3]
        return R, t


@dataclass
class Intrinsics:
    camera_conventions: str
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
