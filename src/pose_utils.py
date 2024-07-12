import numpy as np
from jaxtyping import Float64
from src.camera_parameters import Intrinsics, Extrinsics, PinholeParameters
from typing import Literal


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def look_at_matrix(camera_position, target, up):
    # Camera's forward vector (z-axis)
    forward = normalize(target - camera_position)
    # Camera's right vector (x-axis)
    right = normalize(np.cross(up, forward))
    # Camera's up vector (y-axis), ensure it is orthogonal to the other two axes
    up = np.cross(forward, right)

    # Create the rotation matrix by combining the camera axes to form a basis
    rotation = np.array(
        [
            [right[0], up[0], forward[0], 0],
            [right[1], up[1], forward[1], 0],
            [right[2], up[2], forward[2], 0],
            [0, 0, 0, 1],
        ]
    )

    # Create the translation matrix
    translation = np.array(
        [
            [1, 0, 0, -camera_position[0]],
            [0, 1, 0, -camera_position[1]],
            [0, 0, 1, -camera_position[2]],
            [0, 0, 0, 1],
        ]
    )

    # The view matrix is the inverse of the camera's transformation matrix
    # Here we assume the rotation matrix is orthogonal (i.e., rotation.T == rotation^-1)
    view_matrix = rotation.T @ translation

    return view_matrix


def generate_camera_poses_around_ellipse(
    num_poses: int,
    angle_step: float,
    major_radius: float,
    minor_radius: float,
    inverse: bool = False,
) -> list[Float64[np.ndarray, "4 4"]]:
    """
    Generate camera poses rotating around the origin, forming an elliptical trajectory.
    Can choose to rotate around the x-axis, y-axis, or z-axis.
    """
    poses = []
    for i in range(num_poses):
        angle = np.deg2rad(angle_step * i if not inverse else 360 - angle_step * i)

        cam_x = major_radius * np.sin(angle)
        cam_z = minor_radius * np.cos(angle)
        look_at = np.array([0, 0, 0])  # Assume the object is located at the origin
        camera_position = np.array([cam_x, 0, cam_z])
        up_direction = np.array(
            [0, 1, 0]
        )  # Assume the "up" direction is the positive Y-axis

        # Calculate the camera pose matrix
        pose_matrix = look_at_matrix(camera_position, look_at, up_direction)

        poses.append(pose_matrix)
    return poses


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

    return camera_list
