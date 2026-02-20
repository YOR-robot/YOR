from PIL import Image
import numpy as np
import cv2
import open3d as o3d
from quaternion import as_rotation_matrix, quaternion
from scipy.spatial.transform import Rotation as R
import os
import rerun as rr
from typing import List

import threading
import time
from typing import List, Tuple, Optional
import rerun as rr

import numpy as np
import rerun as rr

def rerun_init():
    rr.init("rerun_visualizer", spawn=False)
    rr.log(
        "world/axis",
        rr.Transform3D(translation=[0, 0, 0], rotation=rr.Quaternion(xyzw=[0, 0, 0, 1])),
        static=True,
    )
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP)

def log_occupancy_as_voxels(
    grid: np.ndarray,
    origin_xy: tuple[float, float],  # (origin_x, origin_z) in world
    grid_res: float,
    y_level: float,                  # world Y to place the grid (e.g. floor_y_mean)
    thickness: float = 0.02,         # voxel thickness along Y
    log_root: str = "world/occ_vox",
    log_free: bool = False,
    log_unknown: bool = False,
):
    """Render the 2D occupancy grid as a carpet of 3D boxes aligned with the map."""
    H, W = grid.shape
    half = np.array([grid_res/2.0, thickness/2.0, grid_res/2.0], dtype=np.float32)

    def _emit(mask: np.ndarray, color: tuple[int,int,int], name: str):
        rows, cols = np.where(mask)
        if rows.size == 0:
            return
        # centers: X from cols, Z from rows, Y pinned slightly above the floor
        centers = np.column_stack([
            origin_xy[0] + (cols + 0.5) * grid_res,
            np.full(rows.shape, y_level + half[1], dtype=np.float32),
            origin_xy[1] + (rows + 0.5) * grid_res,
        ]).astype(np.float32)
        half_sizes = np.repeat(half[None, :], rows.size, axis=0)
        colors = np.tile(np.array(color, dtype=np.uint8), (rows.size, 1))
        rr.log(f"{log_root}/{name}", rr.Boxes3D(centers=centers, half_sizes=half_sizes, colors=colors, fill_mode="solid"))

    # 1. Obstacles (1.0) – bright
    _emit(grid >= 1.0, (255, 255, 255), "obstacles")
    # 2. Unknown – light gray (optional)
    if log_unknown:
        unk = (grid > 0.0) & (grid < 1.0)
        _emit(unk, (200, 200, 200), "unknown")
    # 3. Free – dark gray (optional; can be a LOT of boxes, so off by default)
    if log_free:
        free = grid == 0.0
        _emit(free, (50, 50, 50), "free")

def log_path_3d(
    path_cells: list[tuple[int,int]],
    origin_xy: tuple[float, float],
    grid_res: float,
    y_level: float,
    log_path: str = "world/path3d",
):
    """Render a cell-path as a 3D polyline aligned to the grid."""
    if not path_cells:
        return
    pts = np.array(
        [[origin_xy[0] + (c + 0.5)*grid_res, y_level + 0.03, origin_xy[1] + (r + 0.5)*grid_res]
         for (r, c) in path_cells],
        dtype=np.float32
    )
    rr.log(log_path, rr.LineStrips3D([pts], colors=[(255, 0, 0)]))


def mark_robot_footprint(grid: np.ndarray, center: tuple[int, int], radius_pixels: int):
    """Mark an expanded robot footprint in the occupancy grid at the given center location."""
    r, c = center
    for dr in range(-radius_pixels, radius_pixels + 1):
        for dc in range(-radius_pixels, radius_pixels + 1):
            if dr**2 + dc**2 <= radius_pixels**2:
                rr_, cc_ = r + dr, c + dc
                if 0 <= rr_ < grid.shape[0] and 0 <= cc_ < grid.shape[1]:
                    if grid[rr_, cc_] < 1.0:
                        grid[rr_, cc_] = 0.5  # mark as robot occupied

def visualize_occupancy(grid: np.ndarray, space="world/Occupancy Grid"):
    h, w = grid.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)

    img[grid == 0.0] = (50, 50, 50)
    img[grid == 1.0] = (255, 255, 255)
    unknown_mask = ~((grid == 0.0) | (grid == 1.0))
    img[unknown_mask] = (200, 200, 200)

    #cv2.imshow(window_name, img)
    cv2.imwrite("2D_MAPPPPP.png",img)
    rr.log("world/occupancy", rr.Image(img))

def visualize_grid_with_path_bgr_with_unknown(grid: np.ndarray,
                                              path: list[tuple[int,int]],
                                              start: tuple[int,int],
                                              goal: tuple[int,int],
                                              space="world/occupancy_path"):
    """
    Display a 3-channel BGR image of occupancy+path, with:
      • free     (0.0) → dark gray
      • obstacle (1.0) → white
      • unknown       → light gray
      • path          → red
      • start         → green
      • goal          → blue
    """
    h, w = grid.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # free → dark gray
    img[grid == 0.0] = (50, 50, 50)
    # obstacle → white
    img[grid == 1.0] = (255, 255, 255)
    # unknown → light gray
    unknown_mask = ~((grid == 0.0) | (grid == 1.0))
    img[unknown_mask] = (200, 200, 200)

    # path → red
    for r, c in path:
        img[r, c] = (0, 0, 255)

    # start → green, goal → blue
    img[start] = (0, 255, 0)
    img[goal]  = (255, 0, 0)

    #cv2.imshow(window_name, img)
    rr.log("world/path", rr.Image(img))
