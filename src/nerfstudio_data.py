import numpy as np
from dataclasses import dataclass, field


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
