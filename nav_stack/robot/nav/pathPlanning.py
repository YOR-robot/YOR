# local_grid2d.py
# Build a dynamic 2D occupancy grid (x–z plane) from 3D points at ~10 Hz.
# Assumes RIGHT_HAND_Y_UP (Y is up). Works ego-centrically if T_world_robot is given.

# unknown: dark gray
# free : balck
# occupied: white
# inflated: gray

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Callable
import numpy as np, heapq, cv2
import threading, time
import rerun as rr
import math
from scipy.spatial.transform import Rotation as R
from loop_rate_limiters import RateLimiter

try:
    import cv2  # for fast dilation
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False


# ---- Grid codes (exported) ----------------------------------------------------
UNKNOWN: int  = -1
FREE: int     = 0
OCCUPIED: int = 1
INFLATED: int = 2

__all__ = [
    "Grid2DParams",
    "LocalGrid2D",
    "StaticGridWithLiveOverlayThread",
    "UNKNOWN", "FREE", "OCCUPIED", "INFLATED",
    "render_grid_rgb",
]

try:
    from viserBridge import VISER_LOG_FN  # adjust import path to your project layout
except Exception:
    VISER_LOG_FN = None


def vlog(msg: str) -> None:
    """Print to terminal and, if available, mirror to Viser."""
    print(msg)
    if VISER_LOG_FN is not None:
        try:
            VISER_LOG_FN(msg)
        except Exception:
            # Never let logging crash the robot
            pass



# ---- Parameters ---------------------------------------------------------------
@dataclass
class Grid2DParams:
    # Geometry / resolution
    res_m: float = 0.05          # grid cell size (meters)
    x_half_m: float = 4.0        # left/right half-width (meters)
    z_front_m: float = 6.0       # forward extent (meters)
    z_back_m: float = 2.0        # backward extent (meters)

    # Floor / obstacle segmentation (Y is UP)
    floor_band_m: float = 0.1   # |y - floor_y| ≤ band => floor
    min_obst_h_m: float = 0.2   # dy ≥ min => obstacle (ignore floor speckle)
    max_obst_h_m: float = 1.50   # dy ≤ max => obstacle (ignore ceiling)

    # Safety inflation
    robot_radius_m: float = 0.25 # radius for obstacle inflation # was 0.25

    # Temporal update policy
    decay_per_tick: int = 1      # subtract per tick from obstacle counts
    hit_increase: int = 3        # add per obstacle hit
    occ_threshold: int = 2       # counts ≥ threshold => occupied

    # Frame policy
    ego_centric: bool = True     # if True, grid is in robot frame (x,z)
    min_pts_per_obst_cell: int = 3
    recenter_thresh_m: float = 0.05

    #auto-size global grid from map
    auto_size_from_map: bool = False
    auto_size_margin_m: float = 0.5

    min_world_width_m: float = 4.0
    min_world_height_m: float = 4.0


# ---- Core builder -------------------------------------------------------------
class LocalGrid2D:
    """
    Maintains a local 2D occupancy grid around the robot.

    Inputs:
      - pts_xyz: (N,3) float array, RIGHT_HAND_Y_UP (Y is up), meters.
      - T_world_robot: 4x4 transform mapping robot->world (if ego_centric=True).
                       If provided, points are projected into the robot frame via:
                          p_r = R^T * (p_w - t)
                       If None, a world-aligned grid is produced.

    Output:
      - grid:   (H,W) int8 with codes: UNKNOWN(-1), FREE(0), OCCUPIED(1), INFLATED(2)
      - meta:   dict with resolution, extents, shape, and estimated floor_y
    """
    def __init__(self, params: Grid2DParams):
        self.p = params

        self._dynamic_world = (not self.p.ego_centric) and bool(self.p.auto_size_from_map)
        if self._dynamic_world:
            # Start with a tiny placeholder; real size will be set on first update()
            self.W = 1
            self.H = 1
        else:
            self.W = int(np.ceil((2.0 * self.p.x_half_m) / self.p.res_m))              # cols (x)
            self.H = int(np.ceil((self.p.z_front_m + self.p.z_back_m) / self.p.res_m)) # rows (z)
        # Evidence buffers
        self._counts      = np.zeros((self.H, self.W), dtype=np.int16)   # obstacle evidence
        self._seen_floor  = np.zeros((self.H, self.W), dtype=np.uint8)   # where floor observed
        self._inflated    = np.zeros((self.H, self.W), dtype=np.uint8)   # inflated obstacle mask

        self.floor_y_est: float = 0.0

        self.x0 = None
        self.z_top = None
        self._have_window = False
        self._cx = 0.0
        self._cz = 0.0

        # Precompute inflation kernel
        radius_px = max(1, int(np.ceil(self.p.robot_radius_m / self.p.res_m)))
        self._inflate_radius_px = radius_px

        if _HAS_CV2:
            self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1 + 2*radius_px, 1 + 2*radius_px))
        else:
            # Fallback: simple circular boolean kernel
            yy, xx = np.ogrid[-radius_px:radius_px+1, -radius_px:radius_px+1]
            self._kernel = (xx*xx + yy*yy) <= (radius_px * radius_px)
            self._kernel = self._kernel.astype(np.uint8)

    # ---- Public API -----------------------------------------------------------
    def update(self, pts_xyz: np.ndarray, T_world_robot: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        """
        Update the grid from a fused cloud.

        Returns:
          grid (H,W) int8, meta (dict)
        """
        if pts_xyz is None or len(pts_xyz) == 0:
            return self.as_grid()
        
                # Robot world position if provided
        xw_r, zw_r = None, None
        if T_world_robot is not None and T_world_robot.shape == (4, 4):
            xw_r = float(T_world_robot[0, 3])
            zw_r = float(T_world_robot[2, 3])

        # Initialize or slide the window for WORLD grid
        if not self.p.ego_centric and not self._dynamic_world:
            if not self._have_window:
                # If we don't yet have a pose, center the window at 0,0
                self._center_window_on(xw_r if xw_r is not None else 0.0,
                                    zw_r if zw_r is not None else 0.0)
            else:
                if (xw_r is not None and zw_r is not None):
                    if (abs(xw_r - self._cx) >= self.p.recenter_thresh_m or
                        abs(zw_r - self._cz) >= self.p.recenter_thresh_m):
                        self._slide_window_if_needed(xw_r, zw_r)


        y = pts_xyz[:, 1]
        self.floor_y_est = _mean_floor_y(y, band_m=self.p.floor_band_m, prev_ema=self.floor_y_est, alpha=0.15)

                # --- NEW: dynamic world grid sizing from map bbox ---
        if self._dynamic_world:
            # Work in WORLD coords: pts_xyz is in WORLD already.
            x = pts_xyz[:, 0]
            z = pts_xyz[:, 2]
            if x.size > 0:
                margin = float(self.p.auto_size_margin_m)
                x_min = float(np.min(x)) - margin
                x_max = float(np.max(x)) + margin
                z_min = float(np.min(z)) - margin
                z_max = float(np.max(z)) + margin
                self._ensure_world_grid_covering(x_min, x_max, z_min, z_max)
                # print(
                #         "[Grid2D] world grid size H×W =", self.H, "x", self.W,
                #         "x range =", self.x0, "→", self.x0 + self.W * self.p.res_m,
                #         "z range =", self.z_top - self.H * self.p.res_m, "→", self.z_top,
                #     )

        dy = y - self.floor_y_est
        is_floor = np.abs(dy) <= self.p.floor_band_m
        is_obst  = (dy >= self.p.min_obst_h_m) & (dy <= self.p.max_obst_h_m)

        xz_floor = self._project_xz(pts_xyz[is_floor], T_world_robot)
        xz_obst  = self._project_xz(pts_xyz[is_obst],  T_world_robot)

        # Convert to grid indices
        iz_f, ix_f = self._to_idx(xz_floor)
        iz_o, ix_o = self._to_idx(xz_obst)

        # print(
        #     "[Grid2D.update] pts:",
        #     len(pts_xyz),
        #     "floor pts:", is_floor.sum(),
        #     "obst pts:", is_obst.sum(),
        #     "floor cells:", ix_f.size,
        #     "obst cells:", ix_o.size,
        #     "counts>0:", np.count_nonzero(self._counts),
        # )

        # Temporal decay
        if self.p.decay_per_tick > 0:
            self._counts = np.maximum(0, self._counts - self.p.decay_per_tick)

        # Reset per-tick masks
        self._seen_floor.fill(0)
        self._inflated.fill(0)

        # Mark floor seen
        if ix_f.size:
            self._seen_floor[iz_f, ix_f] = 1

        # Accumulate obstacle hits
        if ix_o.size:
            flat = iz_o.astype(np.int64) * self.W + ix_o.astype(np.int64)
            uniq, counts = np.unique(flat, return_counts=True)
            if self.p.min_pts_per_obst_cell > 1:
                good = uniq[counts >= int(self.p.min_pts_per_obst_cell)]
            else:
                good = uniq  # no threshold: any touch counts
            if good.size:
                gi = (good // self.W).astype(np.int32)
                gj = (good %  self.W).astype(np.int32)
                np.add.at(self._counts, (gi, gj), int(self.p.hit_increase))

        # Occupied
        occ = (self._counts >= self.p.occ_threshold).astype(np.uint8)

        # Inflate
        if np.any(occ):
            self._inflated = _binary_dilate(occ, self._kernel)
        else:
            self._inflated.fill(0)

        return self.as_grid()

    def as_grid(self) -> Tuple[np.ndarray, Dict]:
        """Return (grid_codes, meta). Grid codes: UNKNOWN=-1, FREE=0, OCCUPIED=1, INFLATED=2."""
        grid = np.full(self._counts.shape, UNKNOWN, dtype=np.int8)

        occ = (self._counts >= self.p.occ_threshold)
        grid[occ] = OCCUPIED

        infl = (self._inflated == 1) & (~occ)
        grid[infl] = INFLATED

        free = (self._seen_floor == 1) & (grid == UNKNOWN)
        grid[free] = FREE

        # Compute extents
        cs = float(self.p.res_m)
        if not self.p.ego_centric and self._dynamic_world and self.x0 is not None and self.z_top is not None:
            x_min_m = float(self.x0)
            x_max_m = float(self.x0 + self.W * cs)
            z_min_m = float(self.z_top - self.H * cs)
            z_max_m = float(self.z_top)
        else:
            x_min_m = -self.p.x_half_m
            x_max_m =  self.p.x_half_m
            z_min_m = -self.p.z_back_m
            z_max_m =  self.p.z_front_m

        meta = {
            "res_m": self.p.res_m,
            "cell_size_m": self.p.res_m,
            "shape": grid.shape,
            "floor_y_est": self.floor_y_est,
            "codes": {UNKNOWN: "unknown", FREE: "free",
                      OCCUPIED: "occupied", INFLATED: "inflated"},
            "ego_centric": self.p.ego_centric,
            "x_min_m": x_min_m,
            "x_max_m": x_max_m,
            "z_min_m": z_min_m,
            "z_max_m": z_max_m,
        }

        if not self.p.ego_centric:
            meta.update({
                "x0": 0.0 if self.x0 is None else float(self.x0),
                "z_top": 0.0 if self.z_top is None else float(self.z_top),
            })

        return grid, meta


    # ---- Helpers -------------------------------------------------------------
    def _project_xz(self, pts_xyz: np.ndarray, T_world_robot: Optional[np.ndarray]) -> np.ndarray:
        """
        Return Nx2 [x,z] in robot frame if ego_centric and T_world_robot is provided,
        else world [x,z].
        """
        if not self.p.ego_centric or T_world_robot is None or T_world_robot.shape != (4,4):
            return pts_xyz[:, [0, 2]]  # world X,Z

        # T_world_robot maps p_r -> p_w. To get p_r from p_w: p_r = R^T (p_w - t)
        R = T_world_robot[:3, :3]
        t = T_world_robot[:3, 3]
        pw = pts_xyz - t[None, :]
        pr = (R.T @ pw.T).T
        return pr[:, [0, 2]]  # robot X,Z
    def _ensure_world_grid_covering(self,
                                    x_min: float, x_max: float,
                                    z_min: float, z_max: float) -> None:
        """
        Ensure the WORLD-aligned grid covers [x_min,x_max] x [z_min,z_max].
        Expands _counts / _seen_floor / _inflated if necessary and updates
        x0, z_top accordingly. Resolution is fixed at self.p.res_m.
        """
        cs = float(self.p.res_m)

        if self.x0 is None or self.z_top is None or self._counts.size == 0:
            # raw size from bbox
            width_m  = max(x_max - x_min, 1e-6)
            height_m = max(z_max - z_min, 1e-6)
            # print("first iteration")
            # print(width_m, height_m)

            # enforce minimum physical size
            width_m  = max(width_m,  self.p.min_world_width_m)
            height_m = max(height_m, self.p.min_world_height_m)
            # print("second iteration")
            # print(width_m, height_m)

            self.W = int(np.ceil(width_m  / cs))
            self.H = int(np.ceil(height_m / cs))

            # anchor origin so bbox is inside
            self.x0    = x_min - 0.5 * (width_m  - (x_max - x_min))
            self.z_top = z_max + 0.5 * (height_m - (z_max - z_min))

            self._counts     = np.zeros((self.H, self.W), dtype=np.int16)
            self._seen_floor = np.zeros((self.H, self.W), dtype=np.uint8)
            self._inflated   = np.zeros((self.H, self.W), dtype=np.uint8)
            self._have_window = True
            # print("final count")
            # print(self._counts.shape)
            return

        # --- Expansions (never shrink) ---
        cur_x1 = self.x0
        cur_x2 = self.x0 + self.W * cs
        cur_z1 = self.z_top - self.H * cs
        cur_z2 = self.z_top

        new_x1 = min(cur_x1, x_min)
        new_x2 = max(cur_x2, x_max)
        new_z1 = min(cur_z1, z_min)
        new_z2 = max(cur_z2, z_max)

        # if already covered, nothing to do
        if (new_x1 >= cur_x1 and new_x2 <= cur_x2 and
            new_z1 >= cur_z1 and new_z2 <= cur_z2):
            return

        width_m  = max(new_x2 - new_x1, self.p.min_world_width_m)
        height_m = max(new_z2 - new_z1, self.p.min_world_height_m)

        W_new = int(np.ceil(width_m  / cs))
        H_new = int(np.ceil(height_m / cs))

        new_x0   = new_x1
        new_ztop = new_z1 + height_m

        # Allocate new arrays
        new_counts     = np.zeros((H_new, W_new), dtype=self._counts.dtype)
        new_seen_floor = np.zeros((H_new, W_new), dtype=self._seen_floor.dtype)
        new_inflated   = np.zeros((H_new, W_new), dtype=self._inflated.dtype)

        # Where does the old grid land inside the new one?
        col_off = int(round((cur_x1 - new_x0) / cs))
        row_off = int(round((new_ztop - self.z_top) / cs))

        # Compute overlap region robustly in case of rounding
        old_H, old_W = self._counts.shape

        r0_new = max(0, row_off)
        c0_new = max(0, col_off)
        r1_new = min(H_new, row_off + old_H)
        c1_new = min(W_new, col_off + old_W)

        r0_old = max(0, -row_off)
        c0_old = max(0, -col_off)
        r1_old = r0_old + (r1_new - r0_new)
        c1_old = c0_old + (c1_new - c0_new)

        if r1_new > r0_new and c1_new > c0_new:
            new_counts[r0_new:r1_new, c0_new:c1_new] = self._counts[r0_old:r1_old, c0_old:c1_old]
            new_seen_floor[r0_new:r1_new, c0_new:c1_new] = self._seen_floor[r0_old:r1_old, c0_old:c1_old]
            new_inflated[r0_new:r1_new, c0_new:c1_new]   = self._inflated[r0_old:r1_old, c0_old:c1_old]

        # Swap in
        self._counts = new_counts
        self._seen_floor = new_seen_floor
        self._inflated = new_inflated
        self.W, self.H = W_new, H_new
        self.x0 = new_x0
        self.z_top = new_ztop
        self._have_window = True


    def _to_idx(self, xz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if xz.size == 0:
            return _EMPTY_I, _EMPTY_I

        cs = self.p.res_m
        if self.p.ego_centric:
            # original ego-centric box
            x_min, x_max = -self.p.x_half_m, self.p.x_half_m
            z_min, z_max = -self.p.z_back_m, self.p.z_front_m
            x, z = xz[:, 0], xz[:, 1]
            mask = (x >= x_min) & (x < x_max) & (z >= z_min) & (z < z_max)
            if not np.any(mask):
                return _EMPTY_I, _EMPTY_I
            x = x[mask]; z = z[mask]
            ix = ((x - x_min) / cs).astype(np.int32)
            iz = ((z - z_min) / cs).astype(np.int32)
        else:
            # WORLD-aligned sliding window using (x0, z_top)
            if self.x0 is None or self.z_top is None:
                return _EMPTY_I, _EMPTY_I
            x, z = xz[:, 0], xz[:, 1]
            x1, x2 = self.x0, self.x0 + self.W * cs
            z1, z2 = self.z_top - self.H * cs, self.z_top
            mask = (x >= x1) & (x < x2) & (z > z1) & (z <= z2)
            if not np.any(mask):
                return _EMPTY_I, _EMPTY_I
            xm = x[mask]; zm = z[mask]
            ix = np.floor((xm - self.x0) / cs).astype(np.int32)
            iz = np.floor((self.z_top - zm) / cs).astype(np.int32)

        np.clip(ix, 0, self.W - 1, out=ix)
        np.clip(iz, 0, self.H - 1, out=iz)
        return iz, ix



# ---- Rendering (optional) ----------------------------------------------------
def render_grid_rgb(grid: np.ndarray, flip_vertical: bool = True) -> np.ndarray:
    """
    Convert grid codes to an RGB image for visualization/logging.
    Colors:
      unknown=gray, free=white, occupied=black, inflated=dark gray
    """
    h, w = grid.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)

    img[grid == UNKNOWN]  = (60, 60, 60)
    img[grid == FREE]     = (0, 0, 0)
    img[grid == OCCUPIED] = (255, 255, 255)
    img[grid == INFLATED] = (128,  128,  128)

    if flip_vertical:
        img = np.flipud(img).copy()
    return img

def _center_window_on(self, xw: float, zw: float):
    cs = self.p.res_m
    self.x0    = xw - 0.5 * self.W * cs
    self.z_top = zw + 0.5 * self.H * cs
    self._cx, self._cz = xw, zw
    self._have_window = True

def _slide_window_if_needed(self, xw: float, zw: float):
    cs = self.p.res_m
    # How many whole cells did we move? (rows increase downward as z decreases)
    dc = int(np.round((xw - self._cx) / cs))
    dr = int(np.round((self._cz - zw) / cs))
    if dc == 0 and dr == 0:
        return
    # Roll counts / masks and blank wrapped margins to UNKNOWN
    if dr != 0:
        self._counts = np.roll(self._counts, dr, axis=0)
        self._seen_floor = np.roll(self._seen_floor, dr, axis=0)
        self._inflated   = np.roll(self._inflated,   dr, axis=0)
        (self._counts[:dr, :], self._seen_floor[:dr, :], self._inflated[:dr, :]) = (0, 0, 0) if dr > 0 else (self._counts[dr:, :]*0, self._seen_floor[dr:, :]*0, self._inflated[dr:, :]*0)

    if dc != 0:
        self._counts = np.roll(self._counts, dc, axis=1)
        self._seen_floor = np.roll(self._seen_floor, dc, axis=1)
        self._inflated   = np.roll(self._inflated,   dc, axis=1)
        (self._counts[:, :dc], self._seen_floor[:, :dc], self._inflated[:, :dc]) = (0, 0, 0) if dc > 0 else (self._counts[:, dc:]*0, self._seen_floor[:, dc:]*0, self._inflated[:, dc:]*0)

    # Update window origin to match the roll
    self.x0    += dc * cs
    self.z_top += dr * cs * (-1)  # dr rows down = z decreases
    self._cx, self._cz = xw, zw

# ---- Internal utilities ------------------------------------------------------
_EMPTY_I = np.empty((0,), dtype=np.int32)

# def _robust_floor_y(y: np.ndarray) -> float:
#     """Low-percentile + median-in-band estimator, robust to outliers."""
#     if y.size == 0:
#         return 0.0
#     y5 = np.percentile(y, 5.0)
#     band = np.abs(y - y5) < 0.05
#     if np.any(band):
#         return float(np.median(y[band]))
#     return float(y5)

def _mean_floor_y(y: np.ndarray, band_m: float, prev_ema: float | None = None, alpha: float = 0.15) -> float:
    """
    Simple floor estimator:
      1) take the 5th percentile (low end of Y, since Y is UP)
      2) keep only points within +band_m of that percentile
      3) return their MEAN; optionally EMA with previous estimate for stability
    """
    if y.size == 0:
        return 0.0 if prev_ema is None else prev_ema
    y = np.asarray(y, dtype=np.float32)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return 0.0 if prev_ema is None else prev_ema
    p5 = np.nanpercentile(y, 5.0)
    
    mask = (y >= p5) & (y <= (p5 + band_m))
    if not np.any(mask):
        y_hat = float(p5)
    else:
        y_hat = float(np.nanmean(y[mask]))
    if prev_ema is None:
        return y_hat
    y_something = (1.0 - alpha) * prev_ema + alpha * y_hat
    #print(y_something)
    return y_something

def _binary_dilate(mask: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Binary dilation with cv2 if available, else a simple Numpy fallback."""
    if _HAS_CV2:
        return cv2.dilate(mask.astype(np.uint8), kernel)
    # Numpy fallback (slow for large kernels/sizes but OK for small grids)
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    pad = np.pad(mask.astype(np.uint8), ((ph, ph), (pw, pw)), mode="constant")
    out = np.zeros_like(mask, dtype=np.uint8)
    # naive scan
    for dy in range(kh):
        for dx in range(kw):
            if kernel[dy, dx]:
                out |= pad[dy:dy+mask.shape[0], dx:dx+mask.shape[1]]
    return out

# 3) A dedicated 10 Hz grid worker thread
class LocalGrid2DThread:
    """
    Runs LocalGrid2D at a fixed rate (default 10 Hz) in ONE thread.

    You provide a fetch() callable that returns:
        pts_xyz: (N,3) float32 in WORLD (RIGHT_HAND_Y_UP)
        T_world_robot: 4x4 float32 or None
    The worker projects to robot frame if ego_centric=True.

    Access the latest result via .get_grid()  -> (grid, meta, t_stamp)
    """

    def __init__(
        self,
        grid_params: Grid2DParams,
        fetch_latest: Callable[[], Tuple[Optional[np.ndarray], Optional[np.ndarray]]],
        hz: float = 10.0,
        downsample_limit: int = 200_000,
        #rerun_log_fn: Optional[Callable[[np.ndarray], None]] = None,  # e.g., lambda img: rr.log("world/local_grid", rr.Image(img))
    ):
        self.grid = LocalGrid2D(grid_params)
        self.fetch_latest = fetch_latest
        self.downsample_limit = int(downsample_limit)
        #self.rerun_log_fn = rerun_log_fn
        self._rate = RateLimiter(hz, name="local_grid2d") if (hz and hz > 0) else None

        self._thr: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        self._last_grid: Optional[np.ndarray] = None
        self._last_meta: Optional[Dict] = None
        self._last_t: float = 0.0
        self._last_T_world_robot = None

    def start(self):
        if self._thr and self._thr.is_alive():
            return
        self._stop_evt.clear()
        self._thr = threading.Thread(target=self._loop, name="LocalGrid2D@10Hz", daemon=True)
        self._thr.start()

    def stop(self, join_timeout: Optional[float] = 1.5):
        self._stop_evt.set()
        if self._thr:
            self._thr.join(timeout=join_timeout)
        self._thr = None

    def get_grid(self) -> Tuple[Optional[np.ndarray], Optional[Dict], Optional[np.ndarray]]:
        with self._lock:
            return (
                None if self._last_grid is None else self._last_grid.copy(),
                None if self._last_meta is None else dict(self._last_meta),
                None if self._last_T_world_robot is None else self._last_T_world_robot.copy(),
            )


    # ---------- internal ----------
    def _loop(self):
        while not self._stop_evt.is_set():
            t0 = time.time()

            try:
                pts_xyz, T_world_robot = self.fetch_latest()

                if pts_xyz is not None and len(pts_xyz) > 0:
                    # Optional throttling of huge clouds (keeps thread snappy)
                    # if len(pts_xyz) > self.downsample_limit:
                    #     step = max(1, len(pts_xyz) // self.downsample_limit)
                    #     pts_xyz = pts_xyz[::step]

                    grid_codes, meta = self.grid.update(pts_xyz, T_world_robot)
                else:
                    grid_codes, meta = self.grid.as_grid()

                # Optional Rerun logging
                img = render_grid_rgb(grid_codes)
                rr.log("world/local_grid", rr.Image(img))
                # print(grid_codes.shape)

                with self._lock:
                    self._last_grid = grid_codes
                    self._last_meta = meta
                    self._last_t = t0
                    self._last_T_world_robot = T_world_robot

            except Exception:
                # Keep the worker alive on transient errors
                pass

            # rate control
            if self._rate is not None:
                self._rate.sleep()

# ---- Static grid + live overlay thread --------------------------------------
def compute_static_grid_from_points(pts_xyz: np.ndarray, grid_params: Grid2DParams):
    """
    Build a world-aligned grid + cost map from a static point cloud once.
    Returns (grid_codes, meta, cost_map, floor_y_est, inflate_kernel).
    """
    builder = LocalGrid2D(grid_params)
    grid_codes, meta = builder.update(pts_xyz, None)
    cost_map = gridcodes_to_float(grid_codes)
    kernel = builder._kernel.copy() if hasattr(builder, "_kernel") else None
    return grid_codes, meta, cost_map, builder.floor_y_est, kernel


class StaticGridWithLiveOverlayThread:
    """
    Reuses a precomputed static grid/cost map and overlays a lightweight grid
    from the latest pointcloud only, so we avoid reprocessing the full map.
    """

    def __init__(
        self,
        datastream,
        base_grid: np.ndarray,
        base_meta: Dict,
        base_cost_map: np.ndarray,
        floor_y: float,
        kernel: np.ndarray,
        grid_params: Grid2DParams,
        hz: float = 10.0,
        overlay_keep_fraction: float = 0.2,
    ):
        self.datastream = datastream
        self.base_grid = np.asarray(base_grid, dtype=np.int8)
        self.base_meta = dict(base_meta)
        self.base_cost_map = np.asarray(base_cost_map, dtype=np.float32)
        self.floor_y = float(floor_y)
        self.kernel = kernel if kernel is not None else np.ones((1, 1), dtype=np.uint8)
        self.params = grid_params

        self.H, self.W = self.base_grid.shape[:2]
        self.cs = float(self.base_meta.get("cell_size_m", self.params.res_m))
        self.x0 = float(self.base_meta.get("x0", 0.0))
        self.z_top = float(self.base_meta.get("z_top", 0.0))

        self._rate = RateLimiter(hz, name="static_overlay") if (hz and hz > 0) else None

        self._thr: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        self._last_grid: Optional[np.ndarray] = None
        self._last_meta: Optional[Dict] = None
        self._last_T_world_robot = None
        self._last_cost_map: Optional[np.ndarray] = None
        self.overlay_keep_fraction = float(np.clip(overlay_keep_fraction, 0.0, 1.0))

    def start(self):
        if self._thr and self._thr.is_alive():
            return
        self._stop_evt.clear()
        self._thr = threading.Thread(target=self._loop, name="StaticGridOverlay@10Hz", daemon=True)
        self._thr.start()

    def stop(self, join_timeout: Optional[float] = 1.5):
        self._stop_evt.set()
        if self._thr:
            self._thr.join(timeout=join_timeout)
        self._thr = None

    def get_grid(self) -> Tuple[Optional[np.ndarray], Optional[Dict], Optional[np.ndarray]]:
        with self._lock:
            grid = None if self._last_grid is None else self._last_grid.copy()
            meta = None if self._last_meta is None else dict(self._last_meta)
            cost_map = None if self._last_cost_map is None else self._last_cost_map
            T = None if self._last_T_world_robot is None else self._last_T_world_robot.copy()
        if meta is not None and cost_map is not None:
            meta["cost_map"] = cost_map
        return grid, meta, T

    # ---------- internal ----------
    def _loop(self):
        while not self._stop_evt.is_set():
            t0 = time.time()
            try:
                grid_codes, meta, cost_map, T_wr = self._compose_grid()
                if grid_codes is not None:
                    img = render_grid_rgb(grid_codes)
                    rr.log("world/local_grid", rr.Image(img))

                with self._lock:
                    self._last_grid = grid_codes
                    self._last_meta = meta
                    self._last_cost_map = cost_map
                    self._last_T_world_robot = T_wr
            except Exception:
                pass

            if self._rate is not None:
                self._rate.sleep()

    def _compose_grid(self):
        grid = self.base_grid.copy()
        cost_map = self.base_cost_map.copy()
        meta = dict(self.base_meta)

        T_wr = None
        try:
            _, _, T_wr = self.datastream.get_pose()
        except Exception:
            T_wr = None


        zed_pkt = None
        try:
            zed_pkt = self.datastream.get_pcd_pose()
        except Exception:
            zed_pkt = None

        if isinstance(zed_pkt, tuple) and len(zed_pkt) >= 2:
            pts_world = self._pcd_to_world_points(zed_pkt[0], zed_pkt[1])
            if pts_world is not None and pts_world.shape[0] > 0:
                pts_world = self._downsample_points(pts_world)
                occ_mask, infl_mask = self._overlay_masks(pts_world)
                if occ_mask is not None:
                    grid[occ_mask] = OCCUPIED
                    infl_only = infl_mask & (~occ_mask)
                    grid[infl_only] = np.maximum(grid[infl_only], INFLATED)

                    cost_map[occ_mask] = 1.0
                    cost_map[infl_only] = 1.0

        meta["cost_map"] = cost_map
        return grid, meta, cost_map, T_wr

    def _pcd_to_world_points(self, pcd, pose_qt: np.ndarray) -> Optional[np.ndarray]:
        arr = np.asarray(pcd)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            xyz = arr[..., :3].reshape(-1, 3).astype(np.float32, copy=False)
        elif arr.ndim == 2 and arr.shape[1] >= 3:
            xyz = arr[:, :3].astype(np.float32, copy=False)
        else:
            return None
        valid = np.isfinite(xyz).all(axis=1)
        if not np.any(valid):
            return None
        pts_cam = xyz[valid]

        qt = np.asarray(pose_qt, dtype=np.float32)
        if qt.shape[0] < 7 or np.linalg.norm(qt[:4]) < 1e-6:
            return None
        Rm = R.from_quat(qt[:4]).as_matrix().astype(np.float32)
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = Rm
        T[:3, 3] = qt[4:7]

        pts_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1), dtype=np.float32)], axis=1)
        pts_world = (T @ pts_h.T).T[:, :3]
        return pts_world

    def _downsample_points(self, pts_world: np.ndarray) -> np.ndarray:
        frac = self.overlay_keep_fraction
        if not (0.0 < frac < 1.0):
            return pts_world
        n = pts_world.shape[0]
        if n <= 1:
            return pts_world
        mask = np.random.rand(n) < frac
        if not np.any(mask):
            mask[np.random.randint(n)] = True
        return pts_world[mask]

    def _overlay_masks(self, pts_world: np.ndarray):
        dy = pts_world[:, 1] - self.floor_y
        is_obst = (dy >= self.params.min_obst_h_m) & (dy <= self.params.max_obst_h_m)
        if not np.any(is_obst):
            return None, None

        xz = pts_world[is_obst][:, [0, 2]]
        iz, ix = self._to_idx_world(xz)
        if ix.size == 0:
            return None, None

        flat = iz.astype(np.int64) * self.W + ix.astype(np.int64)
        uniq, counts = np.unique(flat, return_counts=True)
        mask = counts >= int(max(1, self.params.min_pts_per_obst_cell))
        if not np.any(mask):
            return None, None

        occ_mask = np.zeros((self.H, self.W), dtype=np.uint8)
        valid_cells = uniq[mask]
        gi = (valid_cells // self.W).astype(np.int32)
        gj = (valid_cells % self.W).astype(np.int32)
        occ_mask[gi, gj] = 1

        if np.any(occ_mask):
            infl_mask = _binary_dilate(occ_mask, self.kernel)
        else:
            infl_mask = np.zeros_like(occ_mask, dtype=np.uint8)

        return occ_mask.astype(bool), infl_mask.astype(bool)

    def _to_idx_world(self, xz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if xz.size == 0:
            return _EMPTY_I, _EMPTY_I
        x, z = xz[:, 0], xz[:, 1]
        x1, x2 = self.x0, self.x0 + self.W * self.cs
        z1, z2 = self.z_top - self.H * self.cs, self.z_top
        mask = (x >= x1) & (x < x2) & (z > z1) & (z <= z2)
        if not np.any(mask):
            return _EMPTY_I, _EMPTY_I
        xm = x[mask]; zm = z[mask]
        ix = np.floor((xm - self.x0) / self.cs).astype(np.int32)
        iz = np.floor((self.z_top - zm) / self.cs).astype(np.int32)
        np.clip(ix, 0, self.W - 1, out=ix)
        np.clip(iz, 0, self.H - 1, out=iz)
        return iz, ix


# 4) Convenience factory for your GlobalMapManager source
def make_grid2d_thread_from_globalmap(
    datastream,
    mapmgr,
    grid_params: Grid2DParams,
    hz: float = 10.0,
    T_robot_cam: Optional[np.ndarray] = None,
    #rerun_log_fn: Optional[Callable[[np.ndarray], None]] = None,
) -> LocalGrid2DThread:
    """
    Binds LocalGrid2DThread to your mapping_torch.GlobalMapManager:
      - points come from globalmapmgr.get_global_map()
      - pose comes from globalmapmgr.mapmgr.get_state() (last WORLD_T_CAM)
      - T_world_robot = WORLD_T_CAM @ inv(T_robot_cam) if provided
    """
    if T_robot_cam is None:
        T_robot_cam = np.eye(4, dtype=np.float32)

    def _se3_from_qt(qt: np.ndarray) -> np.ndarray:
        from scipy.spatial.transform import Rotation as R
        qx, qy, qz, qw, tx, ty, tz = qt
        Rm = R.from_quat([qx, qy, qz, qw]).as_matrix().astype(np.float32)
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = Rm; T[:3, 3] = [tx, ty, tz]
        return T

    def _fetch():
        # 1) Cloud
        cloud = mapmgr.get_map()
        if cloud is None:
            pts_np = None
        else:
            pts_np, _ = cloud.cpu_numpy()   # (N,3) in WORLD (RIGHT_HAND_Y_UP)

        # 2) Pose -> T_world_robot
        T_wr = None
        try:
            _,_, T_wr = datastream.get_pose()
        except Exception:
            T_wr = None
        # print(T_wr)
        return pts_np, T_wr

    return LocalGrid2DThread(grid_params, fetch_latest=_fetch, hz=hz)

class PathPlanner:
    """
    A* planner on 2D occupancy grid with spacing + LOS constraints + soft obstacle inflation:
      - min_spacing_m ≤ segment_length ≤ max_spacing_m
      - Every segment has obstacle-free line-of-sight via Bresenham.
      - Final hop to goal may be shorter than min_spacing_m if necessary.
      - Cells within `near_obstacle_radius_cells` of any obstacle get an additive cost penalty.

    Grid values:
      0.0 = free, 0.5 = unknown, 1.0 = obstacle.
      Set `treat_unknown_as_obstacle=True` to block 0.5 cells.
    """
    def __init__(
        self,
        grid: np.ndarray,
        grid_size: float = 0.05,
        min_spacing_m: float = 0.3,
        max_spacing_m: float = 0.4,
        treat_unknown_as_obstacle: bool = False,
        near_obstacle_radius_cells: int = 4, #8
        near_obstacle_penalty: float = 0.5, #3.0
    ):
        assert max_spacing_m >= min_spacing_m, "max_spacing_m must be ≥ min_spacing_m"
        self.grid = grid
        self.grid_size = float(grid_size)
        self.min_spacing_m = float(min_spacing_m)
        self.max_spacing_m = float(max_spacing_m)
        self.treat_unknown_as_obstacle = bool(treat_unknown_as_obstacle)

        # ---- Precompute soft-inflation penalty mask (additive cost) ----
        # Obstacles are 0 in the distance-transform input; everything else is 1.
        # If treat_unknown_as_obstacle=True, unknown cells (0.5) are also treated as obstacles.
        obs_mask = (grid >= 1.0) | (treat_unknown_as_obstacle & (grid == 0.5))
        # distanceTransform expects uint8: non-zero = foreground, zero = background (distance to zeros).
        # We want distance TO obstacles, so obstacles must be zeros.
        dt_input = np.where(obs_mask, 0, 1).astype(np.uint8)

        # Euclidean distance to nearest obstacle, measured in cells
        dist = cv2.distanceTransform(dt_input, cv2.DIST_L2, 5) #was 3

        # Penalty of +near_obstacle_penalty for any cell within radius
        self.penalty_mask = np.zeros_like(dist, dtype=np.float32)
        self.penalty_mask[dist <= float(near_obstacle_radius_cells)] = float(near_obstacle_penalty)

    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        # Mildly inflated heuristic to bias progress
        return 1.9 * float(np.hypot(a[0] - b[0], a[1] - b[1]))

    def _cell_is_free(self, r: int, c: int) -> bool:
        if not (0 <= r < self.grid.shape[0] and 0 <= c < self.grid.shape[1]):
            return False
        v = self.grid[r, c]
        return (v == 0.0) if self.treat_unknown_as_obstacle else (v < 1.0)

    def get_neighbors(self, node: Tuple[int, int]) -> List[Tuple[int, int, float]]:
        # Base move costs (grid steps)
        dirs = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, np.sqrt(2)), (-1, 1, np.sqrt(2)),
            (1, -1, np.sqrt(2)), (1, 1, np.sqrt(2))
        ]
        r, c = node
        out = []
        for dr, dc, base_cost in dirs:
            nr, nc = r + dr, c + dc
            if self._cell_is_free(nr, nc):
                # Additive penalty if near an obstacle
                penalty = float(self.penalty_mask[nr, nc])
                out.append((nr, nc, float(base_cost) + penalty))
        return out

    def _dist_m(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return self.grid_size * float(np.hypot(a[0] - b[0], a[1] - b[1]))

    @staticmethod
    def _bresenham_cells(a: Tuple[int, int], b: Tuple[int, int]) -> List[Tuple[int, int]]:
        r0, c0 = a; r1, c1 = b
        x0, y0, x1, y1 = int(c0), int(r0), int(c1), int(r1)
        dx = abs(x1 - x0); dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        cells = []
        while True:
            cells.append((y0, x0))
            if x0 == x1 and y0 == y1: break
            e2 = 2 * err
            if e2 >= dy:
                err += dy; x0 += sx
            if e2 <= dx:
                err += dx; y0 += sy
        return cells

    def _line_is_free(self, a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        for r, c in self._bresenham_cells(a, b):
            if not (0 <= r < self.grid.shape[0] and 0 <= c < self.grid.shape[1]):
                return False
            v = self.grid[r, c]
            if v >= 1.0:  # obstacle
                return False
            if self.treat_unknown_as_obstacle and v != 0.0:
                return False
        return True

    def _sparsify_path(self, path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Greedy string-pull under [min,max] spacing with LOS. Final hop may be < min."""
        if not path: return []
        if len(path) == 1: return path

        n = len(path)
        sparse = [path[0]]
        i = 0

        while i < n - 1:
            cur = sparse[-1]

            # Early exit: if goal is within max and LOS, jump there (even if < min).
            d_goal = self._dist_m(cur, path[-1])
            if d_goal <= self.max_spacing_m and self._line_is_free(cur, path[-1]):
                sparse.append(path[-1])
                break

            j_lo = None
            j_hi = None
            for j in range(i + 1, n):
                d = self._dist_m(cur, path[j])
                if j_lo is None and d >= self.min_spacing_m:
                    j_lo = j
                if d <= self.max_spacing_m:
                    j_hi = j
                else:
                    break

            best = None
            if j_lo is not None and j_hi is not None and j_lo <= j_hi:
                for j in range(min(j_hi, n - 1), j_lo - 1, -1):
                    if self._line_is_free(cur, path[j]):
                        best = j
                        break

            if best is None:
                j_limit = i + 1
                for j in range(i + 1, n):
                    if self._dist_m(cur, path[j]) <= self.max_spacing_m:
                        j_limit = j
                    else:
                        break
                for j in range(j_limit, i, -1):
                    if self._line_is_free(cur, path[j]):
                        best = j
                        break
                if best is None:
                    best = min(i + 1, n - 1)

            sparse.append(path[best])
            i = best

            if sparse[-1] == path[-1]:
                break

        if sparse[-1] != path[-1]:
            if self._line_is_free(sparse[-1], path[-1]) and self._dist_m(sparse[-1], path[-1]) <= self.max_spacing_m:
                sparse.append(path[-1])
            else:
                k = i
                while k < n - 1:
                    cur = sparse[-1]
                    j_limit = k + 1
                    for j in range(k + 1, n):
                        if self._dist_m(cur, path[j]) <= self.max_spacing_m:
                            j_limit = j
                        else:
                            break
                    picked = None
                    for j in range(j_limit, k, -1):
                        if self._line_is_free(cur, path[j]):
                            picked = j
                            break
                    if picked is None: picked = k + 1
                    sparse.append(path[picked])
                    k = picked
                    if sparse[-1] == path[-1]:
                        break

        return sparse

    def plan(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        open_set = []
        heapq.heappush(open_set, (self.heuristic(start, goal), 0.0, start))
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        cost_so_far: Dict[Tuple[int, int], float] = {start: 0.0}
        in_open = {start}

        while open_set:
            _, g, current = heapq.heappop(open_set)
            in_open.discard(current)
            if g > cost_so_far[current]:
                continue

            if current == goal:
                # Reconstruct raw path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return self._sparsify_path(path)

            for nr, nc, step_cost in self.get_neighbors(current):
                nxt = (nr, nc)
                new_cost = cost_so_far[current] + step_cost
                if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                    cost_so_far[nxt] = new_cost
                    came_from[nxt] = current
                    f = new_cost + self.heuristic(nxt, goal)
                    if nxt not in in_open:
                        heapq.heappush(open_set, (f, new_cost, nxt))
                        in_open.add(nxt)

        return []
    
# ---- Helpers: grid codes -> float map expected by PathPlanner ----
# FREE=0.0, UNKNOWN=0.5, OCCUPIED/INFLATED=1.0
def gridcodes_to_float(grid_codes: np.ndarray) -> np.ndarray:
    g = np.zeros_like(grid_codes, dtype=np.float32)
    g[grid_codes < 0] = 0.5   # UNKNOWN
    g[grid_codes >= 1] = 1.0  # OCCUPIED or INFLATED
    # FREE already 0.0
    return g

# ---- Index <-> world-xz conversion (ego-centric grid: robot ~ center cell) ----
def rc_to_world_xz(r: int, c: int, H: int, W: int, cell_m: float,
                   T_world_robot: np.ndarray) -> Tuple[float, float]:
    # Grid indices relative to robot-centered origin (row down, col right).
    # Convention: +x = right, +z = forward. row increases downward in the image,
    # so we flip sign for z.
    c_rel = (c - W // 2) * cell_m
    z_rel = -(r - H // 2) * cell_m
    x_rel = float(c_rel)
    # Map (x_rel, z_rel) in robot frame to world using T_world_robot
    p_r = np.array([x_rel, 0.0, z_rel, 1.0], dtype=np.float32)
    p_w = (T_world_robot @ p_r)
    return float(p_w[0]), float(p_w[2])

def world_xz_to_rc(wx: float, wz: float, H: int, W: int, cell_m: float,
                   T_world_robot: np.ndarray) -> Tuple[int, int]:
    # Inverse transform world→robot
    T_robot_world = np.linalg.inv(T_world_robot)
    p_w = np.array([wx, 0.0, wz, 1.0], dtype=np.float32)
    p_r = T_robot_world @ p_w
    x_rel, z_rel = float(p_r[0]), float(p_r[2])
    c = int(round(W // 2 + x_rel / cell_m))
    r = int(round(H // 2 - z_rel / cell_m))
    return r, c

def rc_to_world_xz_world(r: int, c: int, meta: Dict) -> Tuple[float, float]:
    cs   = float(meta["cell_size_m"])
    x0   = float(meta["x0"])
    ztop = float(meta["z_top"])
    xw = x0   + (c + 0.5) * cs
    zw = ztop - (r + 0.5) * cs
    return xw, zw

def world_xz_to_rc_world(wx: float, wz: float, meta: Dict) -> Tuple[int, int]:
    cs   = float(meta["cell_size_m"])
    x0   = float(meta["x0"])
    ztop = float(meta["z_top"])
    c = int(np.floor((wx - x0)   / cs))
    r = int(np.floor((ztop - wz) / cs))
    H, W = meta["shape"][0], meta["shape"][1]
    c = max(0, min(W-1, c))
    r = max(0, min(H-1, r))
    return r, c

def _closest_index_on_path_rc(path_rc, rc):
    """
    Find index of path cell closest (in grid space) to rc = (r, c).
    path_rc: list[(r, c)]
    rc: (r, c)
    """
    if not path_rc:
        return 0
    rr, cc = rc
    best_i = 0
    best_d2 = float("inf")
    for i, (r, c) in enumerate(path_rc):
        d2 = (r - rr) * (r - rr) + (c - cc) * (c - cc)
        if d2 < best_d2:
            best_d2 = d2
            best_i = i
    return best_i

def nearest_free_cell_around(planner, rc0, max_radius_cells: int = 30):
    """
    If rc0 is blocked, search outward in square rings for the nearest FREE cell.
    Uses planner._cell_is_free so it respects treat_unknown_as_obstacle.
    """
    r0, c0 = rc0
    if planner._cell_is_free(r0, c0):
        return (r0, c0)

    best = None
    best_d2 = float("inf")

    for rad in range(1, max_radius_cells + 1):
        rmin, rmax = r0 - rad, r0 + rad
        cmin, cmax = c0 - rad, c0 + rad

        # top and bottom edges
        for cc in range(cmin, cmax + 1):
            for rr in (rmin, rmax):
                if planner._cell_is_free(rr, cc):
                    d2 = (rr - r0) * (rr - r0) + (cc - c0) * (cc - c0)
                    if d2 < best_d2:
                        best, best_d2 = (rr, cc), d2

        # left and right edges (excluding corners already checked)
        for rr in range(rmin + 1, rmax):
            for cc in (cmin, cmax):
                if planner._cell_is_free(rr, cc):
                    d2 = (rr - r0) * (rr - r0) + (cc - c0) * (cc - c0)
                    if d2 < best_d2:
                        best, best_d2 = (rr, cc), d2

        if best is not None:
            return best

    return None


def _segment_blocked_rc(planner, segment_rc):
    """
    Return True if any cell along this segment is NOT free according to the PathPlanner.
    Uses planner._cell_is_free(), so it respects treat_unknown_as_obstacle and bounds.
    """
    if not segment_rc:
        return False
    for (r, c) in segment_rc:
        if not planner._cell_is_free(r, c):
            return True
    return False



class AStarPlannerThread:
    """
    Replans to the current goal at 10 Hz from the latest ego grid;
    republishes the latest path at 15 Hz for controllers/visualization.
    """
    def __init__(self, grid_thread,
                 treat_unknown_as_obstacle: bool = False,
                 near_obstacle_radius_cells: int = 4, #8
                 near_obstacle_penalty: float = 0.5, #3
                 log_entity_path_3d: str = "world/path",
                 log_entity_grid_overlay: Optional[str] = "world/local_grid_path_overlay",
                 grid_size_fallback_m: float = 0.05,
                 hold_last_good=False):
        
        
        self.grid_thread = grid_thread
        self.goal_world: Optional[Tuple[float, float]] = None  # (x,z) in world meters
        self._running = False
        self._thr = None
        self._lock = threading.Lock()
        self._latest_path_rc: List[Tuple[int, int]] = []
        self._latest_T_world_robot = None
        self._treat_unknown_as_obstacle = treat_unknown_as_obstacle
        self._near_obst_radius = near_obstacle_radius_cells
        self._near_obst_penalty = near_obstacle_penalty
        self._log_entity_path_3d = log_entity_path_3d
        self._log_entity_grid_overlay = log_entity_grid_overlay
        self._grid_size_fallback_m = grid_size_fallback_m
        self._hold_last_good = bool(hold_last_good)
        self._have_path = False

        self._last_goal_world: Optional[Tuple[float, float]] = None
        self._latest_lookahead_world: Optional[Tuple[float, float]] = None
        self._lookahead_dist_m: float = 0.2
        # Rates
        self._dt_plan = 1.0/5.0   # 5 Hz was 10 hz
        self._dt_pub  = 1.0/6.0  # 15 Hz

    # --- public API ---
    def set_goal_world(self, x_world: float, z_world: float):
        with self._lock:
            self.goal_world = (float(x_world), float(z_world))
            # print(self.goal_world)

    def set_latest_lookahead_world(self, x_world: float, z_world: float):
        """Called by PathFollower to update the current lookahead point (world X,Z)."""
        with self._lock:
            self._latest_lookahead_world = (float(x_world), float(z_world))

    def get_latest_lookahead_world(self) -> Optional[Tuple[float, float]]:
        """Return the most recent pure-pursuit lookahead point (world X,Z)."""
        with self._lock:
            return self._latest_lookahead_world


    def get_latest_path_world(self) -> List[Tuple[float, float]]:
        # Convert cached rc path to world xz using the latest pose/meta
        with self._lock:
            path_rc = list(self._latest_path_rc)
            T = None if self._latest_T_world_robot is None else self._latest_T_world_robot.copy()
        if not path_rc or T is None:
            return []
        grid, meta, _T_world_robot = self.grid_thread.get_grid()
        if grid is None:
            return []
        H, W = grid.shape[:2]
        cell = float(meta.get("cell_size_m", self._grid_size_fallback_m))
        out = []
        for r, c in path_rc:
            if meta.get("ego_centric", True):
                xw, zw = rc_to_world_xz(r, c, H, W, cell, T)
            else:
                xw, zw = rc_to_world_xz_world(r, c, meta)
            out.append((xw, zw))
        return out

    def start(self):
        if self._running: return
        self._running = True
        self._thr = threading.Thread(target=self._loop, daemon=True)
        self._thr.start()

    def stop(self):
        self._running = False
        if self._thr is not None:
            self._thr.join(timeout=1.0)
            self._thr = None

    # --- internals ---
    def _loop(self):
        t_last_plan = 0.0
        t_last_pub  = 0.0
        while self._running:
            t = time.time()
            did_work = False

            # PLAN @ 10 Hz
            if t - t_last_plan >= self._dt_plan:
                did_work = True
                t_last_plan = t
                try:
                    # print("[Planner] _step_plan tick")
                    self._step_plan()
                except Exception as e:
                    print(f"[AStarPlannerThread] plan error: {e}")

            # PUBLISH @ 15 Hz
            if t - t_last_pub >= self._dt_pub:
                did_work = True
                t_last_pub = t
                try:
                    self._step_publish()
                except Exception as e:
                    print(f"[AStarPlannerThread] publish error: {e}")

            if not did_work:
                time.sleep(0.001)

    def _step_plan(self):
        """
        Receding-horizon FULL replan:
          - Always plans from CURRENT robot pose to CURRENT goal on the latest grid.
          - Only replans when necessary (goal changed, path blocked, robot deviated, etc.).
          - Keeps exactly ONE active path in _latest_path_rc.
        """
        # 1) Get latest grid + pose
        grid_codes, meta, T_world_robot = self.grid_thread.get_grid()
        if grid_codes is None or T_world_robot is None:
            return

        cost_map = None
        if meta is not None:
            cost_map = meta.get("cost_map")

        # 2) Get current goal (world x,z)
        with self._lock:
            goal = self.goal_world
        if goal is None:
            return

        H, W = grid_codes.shape[:2]
        cell = float(meta.get("cell_size_m", self._grid_size_fallback_m))

        # 3) Convert grid codes → float map for PathPlanner
        if cost_map is not None and isinstance(cost_map, np.ndarray) and cost_map.shape == grid_codes.shape:
            grid_f = cost_map.astype(np.float32, copy=False)
        else:
            grid_f = gridcodes_to_float(grid_codes)

        # 4) Compute start/goal indices in grid space
        if meta.get("ego_centric", True):
            # Ego-centric: robot is approximately at center cell
            start_rc = (H // 2, W // 2)
            goal_rc  = world_xz_to_rc(goal[0], goal[1], H, W, cell, T_world_robot)
        else:
            # World-aligned grid
            x_r, z_r = float(T_world_robot[0, 3]), float(T_world_robot[2, 3])
            start_rc = world_xz_to_rc_world(x_r, z_r, meta)
            goal_rc  = world_xz_to_rc_world(goal[0], goal[1], meta)

        # 5) Check that goal lies inside the grid
        gr, gc = goal_rc
        if not (0 <= gr < H and 0 <= gc < W):
            if not self._hold_last_good:
                with self._lock:
                    self._latest_path_rc = []
                    self._latest_T_world_robot = None
                    self._have_path = False
            return

        # 6) Build a planner on this snapshot
        planner = PathPlanner(
            grid=grid_f,
            grid_size=cell,
            treat_unknown_as_obstacle=self._treat_unknown_as_obstacle,
            near_obstacle_radius_cells=self._near_obst_radius,
            near_obstacle_penalty=self._near_obst_penalty,
        )

        sr, sc = start_rc

        # If start cells are not free, don't try to plan
        if not planner._cell_is_free(sr, sc):
            if not self._hold_last_good:
                with self._lock:
                    self._latest_path_rc = []
                    self._latest_T_world_robot = None
                    self._have_path = False
            return
        
        # If the end cell is not free, try to find nearest free cell around it
        if not planner._cell_is_free(gr, gc):
            new_goal_rc = nearest_free_cell_around(
                planner=planner,
                rc0=(gr, gc),
                max_radius_cells=int(max(10, 2.0 / cell)),
            )
            if new_goal_rc is None:
                if not self._hold_last_good:
                    with self._lock:
                        self._latest_path_rc = []
                        self._latest_T_world_robot = None
                        self._have_path = False
                return
            
            gr, gc = new_goal_rc
            goal_rc = (gr, gc)

        # 7) Decide whether we actually need a new plan (receding horizon logic)
        with self._lock:
            prev_path_rc = list(self._latest_path_rc) if self._have_path else []
            last_goal    = self._last_goal_world

        need_replan = False

        # (A) No previous path → must plan
        if not prev_path_rc or last_goal is None:
            need_replan = True
        else:
            # (B) Goal changed by more than ~2 cm
            dgx = goal[0] - last_goal[0]
            dgz = goal[1] - last_goal[1]
            if (dgx * dgx + dgz * dgz) > (0.02 ** 2):
                need_replan = True
            else:
                # (C) Any cell on the current path is now blocked in this grid
                if _segment_blocked_rc(planner, prev_path_rc):
                    need_replan = True
                else:
                    # (D) Robot deviated too far from the current path
                    idx_close = _closest_index_on_path_rc(prev_path_rc, start_rc)
                    pr, pc = prev_path_rc[idx_close]
                    dr = pr - sr
                    dc = pc - sc
                    cell_dist = math.hypot(dr, dc)
                    dev_thresh_cells = max(2.0, 0.25 / cell)  # ≈ 25 cm in grid units
                    if cell_dist > dev_thresh_cells:
                        need_replan = True

        # If current plan is still good, keep following it
        if not need_replan:
            return

        # 8) FULL replan from CURRENT pose to CURRENT goal
        new_path_rc = planner.plan(start_rc, goal_rc)

        ####################################################
        if not new_path_rc:
            alt_goal_rc = nearest_free_cell_around(
                planner=planner,
                rc0=goal_rc,
                max_radius_cells=int(max(20, 4.0 / cell)),
            )
            if alt_goal_rc is not None and alt_goal_rc != goal_rc:
                new_path_rc = planner.plan(start_rc, alt_goal_rc)

        # 9) Commit or clear
        if new_path_rc:
            with self._lock:
                self._latest_path_rc = new_path_rc
                self._latest_T_world_robot = T_world_robot.copy()
                self._have_path = True
                self._last_goal_world = goal
        elif not self._hold_last_good:
            with self._lock:
                self._latest_path_rc = []
                self._latest_T_world_robot = None
                self._have_path = False




    def _step_publish(self):
        # Publish the latest path as 3D line (y=0) and optionally a grid+path overlay.
        # Note: We don’t recompute; just re-log at 15 Hz for smooth viewers/controllers.
        with self._lock:
            if not self._have_path:
                return
            path_rc = list(self._latest_path_rc)
            T = None if self._latest_T_world_robot is None else self._latest_T_world_robot.copy()
        if not path_rc or T is None:
            return

        grid_codes, meta, _ = self.grid_thread.get_grid()
        if grid_codes is None:
            return
        H, W = grid_codes.shape[:2]
        cell = float(meta.get("cell_size_m", self._grid_size_fallback_m))

        # Convert rc → world and log 3D line
        pts_world = []
        for r, c in path_rc:
            if meta.get("ego_centric", False):
                xw, zw = rc_to_world_xz(r, c, H, W, cell, T)
            else:
                xw, zw = rc_to_world_xz_world(r, c, meta)

            pts_world.append([xw, 0.0, zw])
        if pts_world:
            rr.log(self._log_entity_path_3d, rr.LineStrips3D(np.asarray([pts_world], dtype=np.float32)))

        # Optional: draw overlay on the current RGB grid image
        if self._log_entity_grid_overlay is not None:
            from robot.nav.pathPlanning import render_grid_rgb  # your function

            # Keep the vertical flip (top = far, bottom = near)
            img = render_grid_rgb(grid_codes, flip_vertical=True)

            if len(path_rc) >= 2:
                H, W = grid_codes.shape[:2]
                # Convert grid (r,c) → image (x=c, y=H-1-r) because of flipud
                poly = np.array([[c, H - 1 - r] for (r, c) in path_rc], dtype=np.int32)
                cv2.polylines(img, [poly], isClosed=False, thickness=2, color=(0, 255, 0))

            rr.log(self._log_entity_grid_overlay, rr.Image(img))
