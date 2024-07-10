from jaxtyping import Float32, Int64, Bool
import numpy as np
from einops import rearrange


def depth_to_points(
    depth_1hw: Float32[np.ndarray, "1 h w"],
    K_33: Float32[np.ndarray, "3 3"],
    R=None,
    t=None,
) -> Float32[np.ndarray, "h w 3"]:
    K_33_inv: Float32[np.ndarray, "3 3"] = np.linalg.inv(K_33)
    if R is None:
        R: Float32[np.ndarray, "3 3"] = np.eye(3, dtype=np.float32)
    if t is None:
        t: Float32[np.ndarray, "3"] = np.zeros(3, dtype=np.float32)

    _, height, width = depth_1hw.shape

    # create 3d points grid
    x: Int64[np.ndarray, " h"] = np.arange(width)
    y: Int64[np.ndarray, " w"] = np.arange(height)
    coord: Int64[np.ndarray, "h w 2"] = np.stack(np.meshgrid(x, y), -1)
    z: Int64[np.ndarray, "h w 1"] = np.ones_like(coord)[:, :, [0]]
    coord = np.concatenate((coord, z), -1).astype(np.float32)  # z=1
    coord: Float32[np.ndarray, "1 h w 3"] = rearrange(coord, "h w c -> 1 h w c")

    # from depth to 3D points
    depth_1hw11: Float32[np.ndarray, "1 h w 1 1"] = rearrange(
        depth_1hw, "1 h w -> 1 h w 1 1"
    )

    # back project points from pixels to camera coordinate system
    pts3D_1 = (
        depth_1hw11
        * rearrange(K_33_inv, "h w -> 1 1 1 h w")
        @ rearrange(coord, "1 h w c -> 1 h w c 1")
    )

    # transform from camera to world coordinate system
    pts3D_2: Float32[np.ndarray, "1 h w 3 1"] = rearrange(
        R, "h w -> 1 1 1 h w"
    ) @ pts3D_1 + rearrange(t, "s -> 1 1 1 s 1")

    # rearrange to 3D points
    pointcloud: Float32[np.ndarray, "h w 3"] = rearrange(pts3D_2, "1 h w c 1 -> h w c")
    return pointcloud


def depth_edges_mask(
    depth: Float32[np.ndarray, "h w"], threshold: float = 0.1
) -> Bool[np.ndarray, "h w"]:
    """Returns a mask of edges in the depth map.
    Args:
    depth: 2D numpy array of shape (H, W) with dtype float32.
    Returns:
    mask: 2D numpy array of shape (H, W) with dtype bool.
    """
    # Compute the x and y gradients of the depth map.
    depth_dx, depth_dy = np.gradient(depth)
    # Compute the gradient magnitude.
    depth_grad = np.sqrt(depth_dx**2 + depth_dy**2)
    # Compute the edge mask.
    mask: Bool[np.ndarray, "h w"] = depth_grad > threshold
    return mask
