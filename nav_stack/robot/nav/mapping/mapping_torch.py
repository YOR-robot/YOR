# mapping_torch.py  (Open3D-free, GPU-first point cloud mapping)
# - Uses ZED native point cloud (XYZRGBA) when available via datastream.get_pcd_pose()
# - RGB/Depth fallback path kept intact
# - Point cloud stored as torch tensors: {points: (N,3), colors: (N,3)}
# - Same MapManager structure (save/load/visualize + live mapping thread)
# - Simple voxel downsampling on GPU

from PIL import Image
import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import rerun as rr
from typing import List, Tuple, Optional

import threading
import time
import torch
from loop_rate_limiters import RateLimiter

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# ============================================================
# Point cloud container (torch tensors)
# ============================================================
class TorchPointCloud:
    """
    Stores a point cloud on GPU (if available).

    Attributes:
        points: (N,3) float32
        colors: (N,3) uint8 in [0,255]
    """
    def __init__(self, points: torch.Tensor, colors: torch.Tensor):
        assert points.ndim == 2 and points.shape[1] == 3, "points must be (N,3)"
        assert colors.ndim == 2 and colors.shape[1] == 3, "colors must be (N,3)"
        if points.device != DEVICE:
            points = points.to(DEVICE)
        if colors.device != DEVICE:
            colors = colors.to(DEVICE)
        # Enforce dtypes
        points = points.float()
        if colors.dtype != torch.uint8:
            colors = torch.clamp(colors, 0, 255).to(torch.uint8)

        self.points = points
        self.colors = colors

    def __len__(self):
        return self.points.shape[0]

    def clone(self) -> "TorchPointCloud":
        return TorchPointCloud(self.points.clone(), self.colors.clone())

    def to(self, device: torch.device) -> "TorchPointCloud":
        return TorchPointCloud(self.points.to(device), self.colors.to(device))

    def cpu_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (points_np, colors_np) for serialization/viz."""
        return self.points.detach().cpu().numpy(), self.colors.detach().cpu().numpy()

    def append(self, other: "TorchPointCloud"):
        """In-place concatenation."""
        if other is None or len(other) == 0:
            return
        # Move other to this device if needed
        if other.points.device != self.points.device:
            other = other.to(self.points.device)
        self.points = torch.cat([self.points, other.points], dim=0)
        self.colors = torch.cat([self.colors, other.colors], dim=0)

    def transformed(self, T_4x4: torch.Tensor) -> "TorchPointCloud":
        """Return a new cloud with transform applied (4x4, row-major)."""
        return TorchPointCloud(apply_transform(self.points, T_4x4), self.colors.clone())

    def transform_(self, T_4x4: torch.Tensor):
        """In-place transform."""
        self.points = apply_transform(self.points, T_4x4)

# ============================================================
# Math helpers (torch)
# ============================================================
def pose_to_matrix(quat_xyzw: np.ndarray, trans_xyz: np.ndarray, device: torch.device = DEVICE) -> torch.Tensor:
    """
    Convert pose into 4x4 torch transform.
    quat_xyzw: [x,y,z,w]
    trans_xyz: [tx,ty,tz]
    """
    rot = R.from_quat(quat_xyzw).as_matrix().astype(np.float32)
    T = torch.eye(4, dtype=torch.float32, device=device)
    T[:3, :3] = torch.from_numpy(rot).to(device)
    T[:3, 3] = torch.from_numpy(trans_xyz.astype(np.float32)).to(device)
    return T

def apply_transform(points: torch.Tensor, T_4x4: torch.Tensor) -> torch.Tensor:
    """Apply 4x4 transform to Nx3 points (torch) -> Nx3."""
    assert T_4x4.shape == (4, 4)
    ones = torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)
    homog = torch.cat([points, ones], dim=1)  # (N,4)
    out = (homog @ T_4x4.T)[:, :3]
    return out

def make_flip_transform(device: torch.device = DEVICE) -> torch.Tensor:
    """Open3D-style flip used previously (for some RGB-D sensors)."""
    T = torch.eye(4, dtype=torch.float32, device=device)
    T[1, 1] = -1.0
    T[2, 2] = -1.0
    return T

# ============================================================
# RGB-D -> TorchPointCloud (GPU)  [fallback path]
# ============================================================

@torch.no_grad()
def voxel_downsample_(pc: TorchPointCloud, voxel_size: float) -> TorchPointCloud:
    """
    Open3D-style voxel grid downsampling for TorchPointCloud.
    Filters out NaN/Inf rows first, then averages XYZ and RGB within each voxel.
    Runs on CPU or CUDA depending on pc.points.device.
    """
    if voxel_size <= 0:
        raise ValueError("voxel_size must be > 0")

    device = pc.points.device
    pts = pc.points                         # (N,3) float32
    cols_u8 = pc.colors                     # (N,3) uint8 (per your class)

    # 1) Filter invalid rows: points must be finite; if colors are float, check them too.
    pts_ok = torch.isfinite(pts).all(dim=1)
    if cols_u8.dtype.is_floating_point:
        cols_ok = torch.isfinite(cols_u8).all(dim=1)
    else:
        cols_ok = torch.ones_like(pts_ok, dtype=torch.bool, device=device)

    ok = pts_ok & cols_ok
    if not ok.any():
        # Return an empty cloud on the same device/dtypes
        return TorchPointCloud(pts[:0], cols_u8[:0])

    pts = pts[ok]
    cols_u8 = cols_u8[ok]
    cols = cols_u8.to(torch.float32)        # accumulate in float

    # 2) Open3D-like min bound: min - 0.5 * voxel_size
    min_bound = pts.amin(dim=0) - (voxel_size * 0.5)

    # 3) Integer voxel indices
    vox_idx = torch.floor((pts - min_bound) / voxel_size).to(torch.long)  # (M,3)

    # 4) Unique voxel cells + mapping per point (deterministic order)
    unique_vox, inverse = torch.unique(
        vox_idx, dim=0, return_inverse=True, sorted=True
    )  # unique_vox: (K,3), inverse: (M,)

    K = unique_vox.shape[0]
    ones = torch.ones((pts.shape[0], 1), device=device, dtype=torch.float32)

    # 5) Sum positions/colors per voxel
    pts_sum = torch.zeros((K, 3), device=device, dtype=torch.float32)
    pts_sum.index_add_(0, inverse, pts)

    col_sum = torch.zeros((K, 3), device=device, dtype=torch.float32)
    col_sum.index_add_(0, inverse, cols)

    # 6) Counts per voxel
    cnts = torch.zeros((K, 1), device=device, dtype=torch.float32)
    cnts.index_add_(0, inverse, ones)

    # 7) Means
    pts_mean = (pts_sum / cnts.clamp_min(1.0)).to(torch.float32)
    col_mean = col_sum / cnts.clamp_min(1.0)
    colors_ds = torch.clamp(torch.round(col_mean), 0, 255).to(torch.uint8)

    return TorchPointCloud(pts_mean, colors_ds)

@torch.no_grad()
def clean_outliers_torch(
    cloud: Optional["TorchPointCloud"],
    radius: float = 0.12,
    min_neighbors: int = 3,
    max_points: int = 4000,
) -> Optional["TorchPointCloud"]:
    """
    Remove isolated points using neighbor counting (torch.cdist).
    NOTE: O(N^2) → keep max_points small (few thousand).
    """
    if cloud is None:
        return None
    if len(cloud) == 0:
        return cloud

    points = cloud.points
    colors = cloud.colors

    # Uniform subsample to cap O(N^2) cost
    if max_points is not None and points.shape[0] > max_points:
        idx = torch.randperm(points.shape[0], device=points.device)[:max_points]
        points = points.index_select(0, idx)
        colors = colors.index_select(0, idx)

    if points.shape[0] == 0:
        return TorchPointCloud(points, colors)

    dists = torch.cdist(points, points)
    neighbor_counts = (dists < radius).sum(dim=1) - 1
    mask = neighbor_counts >= min_neighbors

    return TorchPointCloud(points[mask], colors[mask])


# ============================================================
# ZED PCD -> TorchPointCloud (GPU)  [preferred path]
# ============================================================
def _rgba_float_to_rgb_u8(rgba_float_np: np.ndarray) -> np.ndarray:
    """
    Vectorized reinterpretation of ZED packed RGBA stored as a float32.
    Returns (H,W,3) uint8 RGB.
    """
    f = np.ascontiguousarray(rgba_float_np.astype(np.float32, copy=False))
    u32 = f.view(np.uint32)
    # MEASURE.XYZRGBA packs bytes as 0xAABBGGRR in little-endian float layout.
    r = (u32 & 0x000000FF).astype(np.uint8)
    g = ((u32 >> 8) & 0x000000FF).astype(np.uint8)
    b = ((u32 >> 16) & 0x000000FF).astype(np.uint8)
    rgb = np.stack([r, g, b], axis=-1)
    return rgb

@torch.no_grad()
def zed_pcd_to_pointcloud_torch(
    zed_pcd,                         # sl.Mat or np.ndarray (H,W,4) float32, in CAMERA frame
    pose_qt: np.ndarray,             # [qx,qy,qz,qw, tx,ty,tz], WORLD_T_CAM
) -> TorchPointCloud:
    """
    Build a TorchPointCloud directly from ZED native point cloud.
    - Points are taken from XYZ (meters) and transformed to WORLD using pose_qt.
    - Colors are decoded from packed RGBA float (we keep RGB, drop alpha).
    - Invalid points (nan/inf) are removed.
    """
    try:
        import pyzed.sl as sl  # optional; only used when pcd is sl.Mat
        SL_AVAILABLE = True
    except Exception:
        SL_AVAILABLE = False

    if SL_AVAILABLE and hasattr(zed_pcd, "get_data"):
        arr = zed_pcd.get_data(sl.MEM.CPU)
    else:
        arr = np.asarray(zed_pcd)

    assert arr.ndim == 3 and arr.shape[2] >= 3, "Expected (H,W,4) or (H,W,3) array from ZED"

    xyz = arr[..., :3].astype(np.float32, copy=False)            # (H,W,3) in camera frame
    if arr.shape[2] >= 4:
        try:
            rgb = _rgba_float_to_rgb_u8(arr[..., 3])
            colors_u8 = rgb.reshape(-1, 3)
        except Exception:
            colors_u8 = np.full((xyz.size // 3, 3), 255, dtype=np.uint8)
    else:
        colors_u8 = np.full((xyz.size // 3, 3), 255, dtype=np.uint8)

    valid = np.isfinite(xyz).all(axis=2)
    if not np.any(valid):
        return TorchPointCloud(
            points=torch.zeros((0, 3), dtype=torch.float32, device=DEVICE),
            colors=torch.zeros((0, 3), dtype=torch.uint8, device=DEVICE),
        )

    pts_cam = torch.from_numpy(xyz[valid].reshape(-1, 3)).to(DEVICE, non_blocking=True).float()
    cols = torch.from_numpy(colors_u8[valid.reshape(-1)]).to(DEVICE, non_blocking=True)

    # Transform CAM -> WORLD (no flip needed for ZED point cloud)
    pose_qt = np.asarray(pose_qt, dtype=np.float32).reshape(-1)
    if pose_qt.size < 7:
        raise ValueError(f"pose_qt must have at least 7 elements, got {pose_qt.size}")
    quat, trans = pose_qt[:4], pose_qt[4:7]
    T_world_cam = pose_to_matrix(quat, trans, device=DEVICE)

    pts_world = apply_transform(pts_cam, T_world_cam)
    
    # return voxel_downsample_(TorchPointCloud(points=pts_world, colors=cols), 0.02)
    return TorchPointCloud(points=pts_world, colors=cols)


@torch.no_grad()
def rgbd_to_pointcloud_torch(
    image: np.ndarray,
    depth: np.ndarray,
    confidence: np.ndarray,
    pose: np.ndarray,
    focal,
    resolution,
    device: torch.device = DEVICE,
) -> TorchPointCloud:
    """
    Convert an RGB-D frame + pose into a TorchPointCloud in WORLD frame.

    image:      H x W x 3   (BGR from ZED)
    depth:      H x W       (meters)
    confidence: H x W       (currently unused)
    pose:       [qx, qy, qz, qw, tx, ty, tz]  (WORLD_T_CAM)
    focal:      [fx, fy] or [fx, fy, cx, cy]
    resolution: [W, H]  (optional, we infer from depth anyway)
    """
    # --- Depth & intrinsics ---
    depth_m = np.asarray(depth, dtype=np.float32)
    H, W = depth_m.shape

    if focal is None:
        fx = fy = 720.0
        cx = W / 2.0
        cy = H / 2.0
    else:
        focal_list = list(focal)
        if len(focal_list) >= 4:
            fx, fy, cx, cy = focal_list[:4]
        else:
            fx, fy = focal_list[:2]
            cx = W / 2.0
            cy = H / 2.0

    fx = float(fx)
    fy = float(fy)
    cx = float(cx)
    cy = float(cy)

    # Valid depth mask
    Z = depth_m
    valid = np.isfinite(Z) & (Z > 0.0)
    if not np.any(valid):
        return TorchPointCloud(
            points=torch.zeros((0, 3), dtype=torch.float32, device=device),
            colors=torch.zeros((0, 3), dtype=torch.uint8, device=device),
        )

    # --- Back-project to camera frame ---
    u = np.arange(W, dtype=np.float32)[None, :]   # (1, W)
    v = np.arange(H, dtype=np.float32)[:, None]   # (H, 1)

    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    pts_cam = np.stack([X[valid], Y[valid], Z[valid]], axis=1)  # (N, 3)
    pts_cam_t = torch.from_numpy(pts_cam).to(device=device, dtype=torch.float32)

    # --- Colors from RGB image ---
    img_np = np.asarray(image)
    if img_np.ndim == 3 and img_np.shape[2] >= 3:
        # ZED gives BGR; convert to RGB so it's consistent with zed_pcd_to_pointcloud_torch
        rgb = img_np[..., ::-1]  # BGR -> RGB
        cols = rgb[valid]
        cols_t = torch.from_numpy(cols).to(device=device, dtype=torch.uint8)
    else:
        cols_t = torch.zeros((pts_cam_t.shape[0], 3), dtype=torch.uint8, device=device)

    # --- Transform CAM -> WORLD using pose ---
    quat = np.asarray(pose[:4], dtype=np.float32)
    trans = np.asarray(pose[4:7], dtype=np.float32)
    T_world_cam = pose_to_matrix(quat, trans, device=device)

    pts_world = apply_transform(pts_cam_t, T_world_cam)

    return TorchPointCloud(points=pts_world, colors=cols_t)

# ============================================================
# Rerun helpers
# ============================================================
def rerun_init():
    rr.init("rerun_visualizer", spawn=True)
    rr.log(
        "world/axis",
        rr.Transform3D(translation=[0, 0, 0], rotation=rr.Quaternion(xyzw=[0, 0, 0, 1]), axis_length=0.5),
        static=True,
    )
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP)

def log_image(image: np.ndarray):
    rr.log("base/image", rr.Image(image))

def log_depth(depth: np.ndarray):
    rr.log("base/depth", rr.DepthImage(depth))

# ============================================================
# Map logging (TorchPointCloud)
# ============================================================
def log_map_from_zedpc(
    curr_map: Optional[TorchPointCloud],
    all_poses: List[np.ndarray],
    zed_pcd,            # sl.Mat or np.ndarray(H,W,4) float32
    pose: np.ndarray,
    *,
    frame_idx: int = 0,
    enable_clean: bool = True,
    clean_every_n: int = 3,       # set to 1 if you REALLY want "always"
    clean_voxel: float = 0.03,    # downsample incoming frame BEFORE cleaning
    clean_radius: float = 0.12,
    clean_min_neighbors: int = 3,
    clean_max_points: int = 4000,
):
    pose = np.asarray(pose, dtype=np.float32).reshape(-1)
    if pose.size < 7:
        raise ValueError(f"pose must have at least 7 elements, got {pose.size}")

    # Visual arrow (camera forward)
    vec = np.array([[0], [0], [-1]], dtype=np.float32)
    dirvec = 0.3 * (R.from_quat(pose[:4]).as_matrix().astype(np.float32) @ vec)
    rr.log(
        "world/pose/arrows",
        rr.Arrows3D(origins=pose[4:7], vectors=dirvec.T, colors=[255, 0, 0], radii=0.01),
    )

    # Build cloud directly from ZED point cloud
    pcd = zed_pcd_to_pointcloud_torch(zed_pcd, pose)

    # (NEW) Light downsample + optional cleanup on ONLY the incoming frame
    if clean_voxel is not None and clean_voxel > 0:
        pcd = voxel_downsample_(pcd, float(clean_voxel))

    if enable_clean and clean_every_n and clean_every_n > 0:
        if (frame_idx % int(clean_every_n)) == 0:
            pcd = clean_outliers_torch(
                pcd,
                radius=float(clean_radius),
                min_neighbors=int(clean_min_neighbors),
                max_points=int(clean_max_points),
            )

    # Merge
    if curr_map is None or len(curr_map) == 0:
        curr_map = pcd
    else:
        curr_map.append(pcd)

    # Keep global map compact
    curr_map = voxel_downsample_(curr_map, 0.02)

    # Track / log pose trail
    all_poses.append(pose)
    rr.log(
        "world/pose",
        rr.Points3D(
            positions=[p[4:7] for p in all_poses],
            radii=[0.01 for _ in all_poses],
            colors=[(0, 255, 0) for _ in all_poses],
        ),
    )
    return curr_map, all_poses


# ============================================================
# Map I/O (npz)
# ============================================================
def save_map_npz(map_cloud: TorchPointCloud, filename: str):
    pts_np, cols_np = map_cloud.cpu_numpy()
    np.savez_compressed(filename, points=pts_np, colors=cols_np)
    print(f"[MapManager] Map saved to {filename}")

def load_map_npz(filename: str) -> TorchPointCloud:
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Map file not found: {filename}")
    data = np.load(filename)
    pts = torch.from_numpy(data["points"]).to(DEVICE).float()
    cols = torch.from_numpy(data["colors"]).to(DEVICE)
    if cols.dtype != torch.uint8:
        cols = torch.clamp(cols, 0, 255).to(torch.uint8)
    return TorchPointCloud(pts, cols)


# ============================================================
# MapManager (same structure)
# ============================================================
class MapManager:
    """
    Save, load, visualize & (now) run live mapping in a separate thread.
    """

    def __init__(self):
        # thread state
        self._thread: Optional[threading.Thread] = None
        self._running: bool = False
        self.paused: bool = False
        self.last_error: Optional[str] = None
        self.datastream=None

        # shared mapping state
        self._lock = threading.Lock()
        self.curr_map: Optional[TorchPointCloud] = None
        self.all_poses: List[np.ndarray] = []

        # downsample cadence
        self._frame_count = 0

        # --- live cleanup knobs (safe defaults) ---
        self.enable_live_clean = True
        self.clean_every_n = 3          # set to 1 to run every frame (can be heavy)
        self.clean_voxel = 0.03         # downsample incoming frame first
        self.clean_radius = 0.12
        self.clean_min_neighbors = 3
        self.clean_max_points = 4000    # keep small (cdist is O(N^2))


    # --- I/O ---
    def save_map(self, map_cloud: TorchPointCloud, filename: str):
        """Save to .npz (points, colors)."""
        save_map_npz(map_cloud, filename)

    def load_map(self, filename: str) -> TorchPointCloud:
        """Load from .npz."""
        return load_map_npz(filename)


    # --- Live mapping control ---
    def start_mapping(self, datastream, *, load: bool = False, target_hz: float = 5.0, map_path: str = None):
        """
        Start the background mapping thread.

        Args:
            datastream: object with .get_rgb_depth_pose()
                - live=True  -> returns (image, depth, confidence, focal, resolution, pose)
                - live=False -> returns (image, depth, pose, timestamp)
            live: whether datastream returns the 'live' tuple above.
            target_hz: desired processing rate (0 or None for as-fast-as-possible)
        """
        if self._running:
            print("[MapManager] Mapping already running.")
            return

        self._running = True
        self.paused = False
        self.last_error = None
        self._frame_count = 0
        self.datastream=datastream
        self._thread = None

        if load:
            self.curr_map = self.load_map(map_path)
        else:
            self.curr_map = None
            self._thread = threading.Thread(
            target=self._mapping_loop, args=(datastream, load, target_hz), daemon=True
            )
            self._thread.start()
            print(f"[MapManager] Mapping thread started (load={load}, target_hz={target_hz}).")                
        

    def stop_mapping(self, join_timeout: Optional[float] = 2.0):
        """Signal the thread to stop and optionally join."""
        if not self._running:
            return
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=join_timeout)
        self._thread = None
        print("[MapManager] Mapping thread stopped.")

    def get_state(self) -> Tuple[Optional[TorchPointCloud], List[np.ndarray]]:
        """Thread-safe snapshot of (map, poses)."""
        with self._lock:
            curr = self.curr_map
            poses_copy = list(self.all_poses)
        return curr, poses_copy
    
    def get_map(self) -> Optional[TorchPointCloud]:
        """Thread-safe snapshot of map."""
        with self._lock:
            curr = self.curr_map
        return curr

    # ---------- internal loop ----------
    # def _mapping_loop(self, datastream, load: bool, target_hz: float):
    #     rate = RateLimiter(target_hz, name="map_manager") if (target_hz and target_hz > 0) else None
    #     while self._running:
    #         # Pause handling
    #         if self.paused:
    #             time.sleep(0.05)
    #             continue

    #         t0 = time.time()
    #         # Update map & log frames
    #         try:
    #             new_map = self.curr_map
    #             new_poses = self.all_poses
    #             zed_pkt = None
    #             try:
    #                 zed_pkt = datastream.get_pcd_pose()
    #             except Exception:
    #                 zed_pkt = None
    #             # print(zed_pkt)
    #             if isinstance(zed_pkt, tuple) and len(zed_pkt) >= 2:
    #                 zed_pcd, pose_zed = zed_pkt[0], zed_pkt[1]
    #                 new_map, new_poses = log_map_from_zedpc(
    #                     self.curr_map, self.all_poses, zed_pcd, pose_zed
    #                 )
    #                 # image, depth, confidence, focal, resolution, pose = datastream.get_rgb_depth_pose()
    #                 # log_image(image)
    #                 # log_depth(depth)

    #             with self._lock:
    #                 self.curr_map = new_map
    #                 self.all_poses = new_poses
    #         except Exception as e:
    #             self.last_error = f"log_map failed: {e}"

    #         if rate is not None:
    #             rate.sleep()

    def _mapping_loop(self, datastream, load: bool, target_hz: float):
        rate = RateLimiter(target_hz, name="map_manager") if (target_hz and target_hz > 0) else None
        while self._running:
            if self.paused:
                time.sleep(0.05)
                continue
            try:
                new_map = self.curr_map
                new_poses = self.all_poses

                zed_pkt = datastream.get_pcd_pose()
                if isinstance(zed_pkt, tuple) and len(zed_pkt) >= 2:
                    zed_pcd, pose_zed = zed_pkt[0], zed_pkt[1]

                    self._frame_count += 1
                    new_map, new_poses = log_map_from_zedpc(
                        self.curr_map, self.all_poses, zed_pcd, pose_zed,
                        frame_idx=self._frame_count,
                        enable_clean=self.enable_live_clean,
                        clean_every_n=self.clean_every_n,
                        clean_voxel=self.clean_voxel,
                        clean_radius=self.clean_radius,
                        clean_min_neighbors=self.clean_min_neighbors,
                        clean_max_points=self.clean_max_points,
                    )

                with self._lock:
                    self.curr_map = new_map
                    self.all_poses = new_poses

            except Exception as e:
                self.last_error = f"log_map failed: {e}"

            if rate is not None:
                rate.sleep()


# # ============================================================
# # Live Map Manager
# # ============================================================
# '''
# This class will output the latest static map+current pointcloud whenever called.

# If load = False, will map the area until Q is pressed and then this will be used as the static map.
# If load = True, mapping will not start and the loaded map will be used as the static map once Q is pressed ,for the entire duration of navigation.

# We will poll zedsub at 10hz to get latest_pcd_pose and apply the pose transformation before adding it to the static map and updating the global map variable.

# IS THIS TOO CONVOLUTED, SHOULD WE JUST INTEGRATE THIS INTO THE MAP MANAGER?

# It was too convoluted, no globul map manager 
# '''

# class GlobalMapManager:
#     """
#     Save, load, visualize & (now) run live mapping in a separate thread.
#     """

#     def __init__(self):
#         # thread state
#         self._thread: Optional[threading.Thread] = None
#         self._running: bool = False
#         self.paused: bool = False
#         self.last_error: Optional[str] = None

#         # shared mapping state
#         self._lock = threading.Lock()
#         self.static_map: TorchPointCloud
#         self.global_map: TorchPointCloud
#         # self.all_poses: List[np.ndarray] = []
#         self.load = False
#         self.mapmgr: MapManager
#         # downsample cadence
#         self._frame_count = 0
#         self.datastream = None
    
#     def start_globalmap(self, datastream, *, load: bool = False, curr_map_path: str = None, mapping_hz: float = 3.0, update_hz: float = 10.0):
#         """
#         Start the global mapping thread to update self.global_map.
#         """
#         if self._running:
#             print("[GlobalMapManager] Mapping already running.")
#             return
#         self.datastream = datastream
#         self.mapmgr = MapManager()
#         self._running = True
#         self.q_pressed = False
#         self.last_error = None
#         self.load = load
#         if load:
#             if curr_map_path is not None and os.path.exists(curr_map_path):
#                 self.static_map = self.mapmgr.load_map(curr_map_path)
#                 print(self.static_map)
#             else:
#                 print("load map failed, check whether path entered is correct")
#         else:
#             self.mapmgr.start_mapping(datastream, target_hz=mapping_hz)
#             print("started map manager hopefully")

#         self._thread = threading.Thread(
#             target=self._global_map_update_loop, args=(datastream, update_hz), daemon=True
#         )
#         self._thread.start()
#         print(f"[GlobalMapManager] global map thread started (update_hz={update_hz}).")

#     def stop_globalmapping(self, join_timeout: Optional[float] = 2.0):
#         """Signal the thread to stop and optionally join."""
#         if not self._running:
#             return
#         self._running = False
#         if self._thread is not None:
#             self._thread.join(timeout=join_timeout)
#         self._thread = None
#         print("[GlobalMapManager] Mapping thread stopped.")

#     def get_global_map(self) -> Optional[TorchPointCloud]:
#         """Thread-safe snapshot of (map, poses)."""
#         with self._lock:
#             _global = self.global_map  
#         return _global
    
#     def get_static_map(self) -> Optional[TorchPointCloud]:
#         """Thread-safe snapshot of (map, poses)."""
#         with self._lock:
#             _static = self.static_map  
#         return _static

#     # ---------- internal loop ----------
#     def _global_map_update_loop(self, datastream, target_hz: float):
#         period = (1.0 / target_hz) if (target_hz and target_hz > 0) else 0.0

#         while self._running:
#             # Pause handling
#             if not self.q_pressed:
#                 # self.mapmgr.paused = False
#                 with self._lock:
#                     if not self.load:
#                         static_map, _ = self.mapmgr.get_state()
#                         self.global_map = static_map
#                         # self.static_map = static_map
#                     else:
#                         self.global_map = self.static_map            
#                 time.sleep(0.05)
#                 continue
#             else:
#                 self.mapmgr.paused = True    # set to True if you want to restart mapping after first Q press
#                 # pass
            
#             t0 = time.time()
#             # --- inside _global_map_update_loop, when self.q_pressed is True ---

#             # Pause mapping so base doesn't change under us
#             self.mapmgr.paused = True

#             # 1) Choose the base (static) map
#             if self.load:
#                 base_map = self.static_map
#             else:
#                 base_map, _ = self.mapmgr.get_state()

#             if base_map is None or len(base_map) == 0:
#                 # nothing to overlay yet
#                 time.sleep(0.02)
#                 continue

#             # 2) Clone so we don't mutate the static map
#             overlay = base_map.clone()

#             # 3) Add the current live cloud only for this frame
#             zed_pcd, pose_zed = datastream.get_pcd_pose()
#             curr_pcd = zed_pcd_to_pointcloud_torch(zed_pcd, pose_zed)

#             overlay.append(curr_pcd)

#             # (optional) downsample the OVERLAY only — not the static map
#             global_map = voxel_downsample_(overlay, 0.05)

#             # 4) Publish the transient overlay as the global map
#             with self._lock:
#                 self.global_map = global_map

#             # Rate control
#             if period > 0:
#                 dt = time.time() - t0
#                 to_sleep = period - dt
#                 if to_sleep > 0:
#                     time.sleep(to_sleep)
