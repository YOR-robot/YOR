#!/usr/bin/env python3
"""
Refactored SLAM node that mirrors the structure of zed_pub_node.py:
  - Wraps all shared state in a Slam class (latest map, grid, goal, path, ZED stream).
  - Spins background threads for mapping (MapManager), visualization (Viser), and planning (A*).
  - Keeps the behavior of slam_node.py while making the runtime flow clearer.
"""

import argparse
import sys
import threading
import time
from typing import Optional, Tuple

import numpy as np
import rerun as rr

from commlink import Subscriber, RPCClient
from scipy.spatial.transform import Rotation as R
from robot.nav.mapping.mapping_torch import MapManager
from robot.nav.pathPlanning import (
    Grid2DParams,
    compute_static_grid_from_points,
    StaticGridWithLiveOverlayThread,
    AStarPlannerThread,
)
from robot.nav.viserBridge import start_viser_server, ViserMirrorThread
from robot.utils.logging import rerun_init
from robot.utils.utils import waitKey


# Pub/sub topics from zed_pub_node.py
POSE_TOPIC = "zed/pose"
IMAGE_TOPIC = "zed/image"
DEPTH_TOPIC = "zed/depth"
PCD_TOPIC = "zed/pcd"
ZED_PUB_PORT = 6000

# RPC from SLAM -> ConeE (follow_path via RPC)
CONE_E_RPC_HOST = "192.168.1.10" # 192.168.1.10 for new cone_e and 194 for old cone_e
CONE_E_RPC_PORT = 5557

def xyzw_xyz_to_matrix(qt7):
    """
    qt7: [qx, qy, qz, qw, tx, ty, tz]
    """
    qt7 = np.asarray(qt7, dtype=np.float32).reshape(-1)
    if qt7.shape[0] < 7:
        raise ValueError(f"Expected 7 values [qx,qy,qz,qw,tx,ty,tz], got {qt7.shape}")
    q = qt7[:4]
    t = qt7[4:7]
    R_mat = R.from_quat(q).as_matrix().astype(np.float32)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R_mat
    T[:3, 3] = t
    return T


class ZedSub:
    def __init__(self, host: str = "192.168.1.11", port: int = ZED_PUB_PORT, up_axis: str = "y"):
        self._up_axis = str(up_axis).lower()
        if self._up_axis not in ("y", "z"):
            self._up_axis = "y"
        self._zup_to_yup = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        topics = [IMAGE_TOPIC, DEPTH_TOPIC, POSE_TOPIC, PCD_TOPIC]
        self._sub = Subscriber(host=host, port=port, topics=topics)
        self._sub_lock = threading.Lock()


    def _zup_to_yup_transform(self, T: np.ndarray) -> np.ndarray:
        return self._zup_to_yup @ T @ self._zup_to_yup.T

    def _zup_to_yup_pose(self, pose_qt: np.ndarray) -> np.ndarray:
        pose_qt = np.asarray(pose_qt, dtype=np.float32).reshape(-1)
        if pose_qt.size < 7:
            return pose_qt
        quat = pose_qt[:4]
        trans = pose_qt[4:7]
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R.from_quat(quat).as_matrix().astype(np.float32)
        T[:3, 3] = trans
        T = self._zup_to_yup_transform(T)
        quat_y = R.from_matrix(T[:3, :3]).as_quat().astype(np.float32)
        trans_y = T[:3, 3].astype(np.float32)
        return np.concatenate([quat_y, trans_y])
    
    def _sub_get(self, topic):
        with self._sub_lock:
            return self._sub[topic]


    def _zup_to_yup_points(self, pcd):
        arr = np.asarray(pcd)
        if arr.ndim < 2 or arr.shape[-1] < 3:
            return arr
        xyz = arr[..., :3].astype(np.float32, copy=False)
        x = xyz[..., 0]
        y = xyz[..., 1]
        z = xyz[..., 2]
        xyz_yup = np.stack([x, z, -y], axis=-1)
        if arr.shape[-1] > 3:
            out = arr.copy()
            out[..., :3] = xyz_yup
            return out
        return xyz_yup

    def stop(self):
        self._sub.stop()
    
    def ready(self) -> bool:
        """Return True once at least the pose topic has been received."""
        ready_attr = getattr(self._sub, "ready", None)
        if callable(ready_attr):
            return bool(ready_attr())
        if ready_attr is not None:
            return bool(ready_attr)

        try:
            pose_msg = self._sub_get(POSE_TOPIC)
        except Exception:
            return False
        return pose_msg is not None

    def get_rgb_depth_pose(self):
        img_msg = self._sub_get(IMAGE_TOPIC)
        depth_msg = self._sub_get(DEPTH_TOPIC)
        pose_msg = self._sub_get(POSE_TOPIC)

        if img_msg is None or depth_msg is None or pose_msg is None:
            raise RuntimeError("ZedSub not ready yet")
        
        image_rgb = img_msg["image"]
        depth_m = depth_msg["depth"]
        pose_qt = pose_msg[7:14]
        if self._up_axis == "z":
            pose_qt = self._zup_to_yup_pose(pose_qt)

        return image_rgb, depth_m, pose_qt
    
    def get_pcd_pose(self):
        pcd_msg = self._sub_get(PCD_TOPIC)
        pose_msg = self._sub_get(POSE_TOPIC)

        if pcd_msg is None or pose_msg is None:
            raise RuntimeError("ZedSub not ready yet or Points not being streamed")
        
        pcd = pcd_msg["points"]
        pose_qt = pose_msg[7:14]  # Pose as [qx, qy, qz, qw, tx, ty, tz]
        if self._up_axis == "z":
            pose_qt = self._zup_to_yup_pose(pose_qt)
            pcd = self._zup_to_yup_points(pcd)
        return pcd, pose_qt
    
    def get_pose(self):
        pose_msg = self._sub_get(POSE_TOPIC)
        base_quat = pose_msg[0:7]
        base_transform = xyzw_xyz_to_matrix(base_quat)
        if self._up_axis == "z":
            base_transform = self._zup_to_yup_transform(base_transform)
        translation = base_transform[:3, 3].astype(np.float32)
        yaw = float(np.arctan2(base_transform[0, 2], base_transform[2, 2]))
        return translation, yaw, base_transform


class Slam:
    """
    Shared SLAM container:
      - Holds latest ZED connection, map, grid, goal, and path.
      - Spawns threads for mapping (MapManager), visualization (Viser), and planning (A*).
      - Keeps ConeE RPC streaming updated paths.
    """

    def __init__(
        self,
        *,
        target_hz: float,
        duration_s: float,
        load_map: bool,
        save_map: bool,
        map_path: Optional[str],
        cone_e_host: str = CONE_E_RPC_HOST,
        cone_e_port: int = CONE_E_RPC_PORT,
        zed_host: str = "127.0.0.1",
        zed_port: int = ZED_PUB_PORT,
        zed_up_axis: str = "y",
        path_step_m: Optional[float] = None,
    ):
        self.target_hz = target_hz
        self.duration_s = duration_s
        self.load_map = load_map
        self.save_map = save_map
        self.map_path = map_path

        self.datastream = ZedSub(host=zed_host, port=zed_port, up_axis=zed_up_axis)
        self.map_manager = MapManager()
        self.cone_e_host = cone_e_host
        self.cone_e_port = cone_e_port
        self._cone_rpc_lock = threading.Lock()
        self.cone_e_client = RPCClient(self.cone_e_host, self.cone_e_port)
        self.server = start_viser_server(host="0.0.0.0", port=8099)
        self.path_step_m = None if path_step_m is None else max(0.0, float(path_step_m))


        self.static_map_pts = None
        self.static_map_cols = None

        # Planner/grid configuration mirrors slam_node.py
        self.grid_params = Grid2DParams(
            res_m=0.05,
            x_half_m=4.0,
            z_front_m=6.0,
            z_back_m=2.0,
            floor_band_m=0.25,
            min_obst_h_m=0.3,
            max_obst_h_m=1.50,
            robot_radius_m=0.3,
            ego_centric=False,
            auto_size_from_map=True,
            auto_size_margin_m=0.5,
            min_pts_per_obst_cell=3,
            hit_increase=3,
            occ_threshold=2,
            min_world_width_m=4.0,
            min_world_height_m=4.0,
         
        )

        # Mutable latest-state fields
        self.latest_map = None
        self.latest_grid: Optional[Tuple[np.ndarray, dict, np.ndarray]] = None
        self.latest_goal: Optional[Tuple[float, float]] = None
        self.latest_path = None

        # Threads / runtime flags
        self.grid_thread: Optional[StaticGridWithLiveOverlayThread] = None
        self.planner: Optional[AStarPlannerThread] = None

        def preview_source():
            # IMPORTANT: use same lock to avoid EFSM / REQ socket clashes
            with self._cone_rpc_lock:
                return self.cone_e_client.get_nav_debug()
        self._preview_source = preview_source
        self.viser_mirror: Optional[ViserMirrorThread] = None
        self.path_thread: Optional[threading.Thread] = None
        self.state_thread: Optional[threading.Thread] = None

        self.running = False
        self.nav_initialized = False
        self.map_loaded = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def run(self):
        self.running = True
        self._init_rerun()

        if not self._wait_for_datastream():
            self.stop()
            return

        self._start_mapping()

        # Start monitoring latest state in the background
        self.state_thread = threading.Thread(target=self._state_monitor_loop, daemon=True)
        self.state_thread.start()

        # If a map was preloaded, bring up navigation immediately
        if self.map_loaded:
            self._start_planning_stack()

        start = time.time()
        try:
            while self.running:
                if self.duration_s and (time.time() - start) >= self.duration_s:
                    break

                self._log_status()

                key = waitKey(1) & 0xFF
                if key == ord("q"):
                    self._on_freeze_and_nav()
                elif key == ord("w"):
                    break

                time.sleep(0.25)
        except KeyboardInterrupt:
            print("\n[slam_node_new] Ctrl+C received; shutting down.")
        finally:
            self.stop()

    def stop(self):
        self.running = False


        self.map_manager.stop_mapping()
        self.datastream.stop()
        if self.grid_thread is not None:
            self.grid_thread.stop()
        if self.planner is not None:
            self.planner.stop()
        if self.viser_mirror is not None:
            self.viser_mirror.stop()


        # Optional save
        if self.save_map and self.map_path:
            try:
                map_cloud = self.map_manager.get_map()
            except Exception:
                map_cloud = None

            if map_cloud is not None and len(map_cloud) > 0:
                try:
                    self.map_manager.save_map(map_cloud, self.map_path)
                    print(f"[slam_node_new] Saved map to: {self.map_path}")
                except Exception as e:
                    print(f"[slam_node_new] Failed to save map '{self.map_path}': {e}", file=sys.stderr)

        print("[slam_node_new] Stopped.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _filter_floating_points(self, pts: np.ndarray, cols: np.ndarray | None,
                            voxel_m: float | None = None, min_pts: int = 3,
                            floor_y: float | None = None, floor_band_m: float = 0.25):
        if pts is None or len(pts) == 0:
            return pts, cols
        if voxel_m is None:
            voxel_m = float(getattr(self.grid_params, "res_m", 0.05))

        pts = np.asarray(pts)

        # Robust floor estimate from lowest 1% of points (y-up).
        if floor_y is None:
            floor_y = float(np.percentile(pts[:, 1], 1.0))

        # Voxelize and keep only voxels with enough points (fast denoise)
        vox = np.floor(pts / voxel_m).astype(np.int32, copy=False)
        _, inv, counts = np.unique(vox, axis=0, return_inverse=True, return_counts=True)
        keep = counts[inv] >= int(min_pts)

        # Always keep points close to floor
        keep_floor = np.abs(pts[:, 1] - floor_y) <= float(floor_band_m)
        keep = keep | keep_floor

        pts_f = pts[keep]
        if cols is None:
            return pts_f, None
        return pts_f, cols[keep]



        
    def _init_rerun(self):
        try:
            rerun_init()
        except Exception as e:
            print(f"[slam_node_new] rerun_init failed (continuing): {e}", file=sys.stderr)

    def _wait_for_datastream(self) -> bool:
        t0 = time.time()
        while not self.datastream.ready():
            if self.duration_s and (time.time() - t0) >= self.duration_s:
                print("[slam_node_new] Timed out waiting for first ZED frames.")
                return False
            if (time.time() - t0) > 5.0:
                print("[slam_node_new] No frames in 5s; continuing anyway.")
                break
            time.sleep(0.01)
        return True

    def _start_mapping(self):
        if self.load_map and self.map_path:
            try:
                self.map_manager.start_mapping(self.datastream, load=True, map_path=self.map_path)
                self.map_loaded = True
                print(f"[slam_node_new] Loaded map: {self.map_path}")
                return
            except Exception as e:
                print(f"[slam_node_new] Failed to load map '{self.map_path}': {e}", file=sys.stderr)

        self.map_manager.start_mapping(self.datastream, target_hz=self.target_hz)
        print("[slam_node_new] Mapping started. Press 'q' to stop mapping and freeze the static map.")

    def _on_freeze_and_nav(self):
        if not self.map_loaded:
            print("\n[slam_node_new] 'q' pressed: stopping mapping and freezing static map.")
            self.map_manager.stop_mapping()

            self.map_loaded = True

        if not self._start_planning_stack():
            return

        try:
            goal_x = float(input("\nX (world): "))
            goal_z = float(input("Z (world): "))
        except Exception as e:
            print("[slam_node_new] Invalid goal input:", e)
            return

        self.latest_goal = (goal_x, goal_z)
        try:
            self.planner.set_goal_world(goal_x, goal_z)
            print(
                f"[slam_node_new] Keyboard goal set to ({goal_x:.2f}, {goal_z:.2f}). "
                "Planner + controller will handle it."
            )
        except Exception as e:
            print(f"[slam_node_new] Failed to set planner goal: {e}")

    def _start_planning_stack(self) -> bool:
        if self.grid_thread is not None and self.planner is not None:
            return True

        static_map = self.map_manager.get_map()
        if static_map is None or len(static_map) == 0:
            print("[slam_node_new] Static map not ready yet; finish mapping with 'q' first.")
            return False
        

        pts_np, cols_np = static_map.cpu_numpy()
        pts_np, cols_np = self._filter_floating_points(pts_np, cols_np, voxel_m=self.grid_params.res_m, min_pts=8, floor_y=None, floor_band_m=0.12)

        self.static_map_pts = pts_np
        self.static_map_cols = cols_np
        base_grid, base_meta, base_cost, floor_y, kernel = compute_static_grid_from_points(
            pts_np, self.grid_params
        )

        self.grid_thread = StaticGridWithLiveOverlayThread(
            datastream=self.datastream,
            base_grid=base_grid,
            base_meta=base_meta,
            base_cost_map=base_cost,
            floor_y=floor_y,
            kernel=kernel,
            grid_params=self.grid_params,
            hz=10.0,
        )
        self.grid_thread.start()

        self.planner = AStarPlannerThread(
            self.grid_thread,
            treat_unknown_as_obstacle=False,
            near_obstacle_radius_cells=4,
            near_obstacle_penalty=0.5,
            log_entity_path_3d="world/path",
            log_entity_grid_overlay="world/local_grid_with_path",
            hold_last_good=False,
        )

        def ensure_nav_started():
            if self.nav_initialized:
                return

            print("\n[slam_node_new] Starting planner (auto).")


            global_map = self.map_manager.get_map()
            if global_map is not None:
                pts_np_local, cols_np_local = global_map.cpu_numpy()
                rr.log("world/global_map", rr.Points3D(positions=pts_np_local, colors=cols_np_local))

            try:
                self.planner.start()
                self.nav_initialized = True
            except Exception as e:
                print(f"[slam_node_new] Failed to start planner: {e}")

        _orig_set_goal_world = self.planner.set_goal_world

        def _set_goal_world_autostart(xw: float, zw: float):
            self.latest_goal = (float(xw), float(zw))
            ensure_nav_started()
            _orig_set_goal_world(xw, zw)

        self.planner.set_goal_world = _set_goal_world_autostart

        if self.server is not None:
            def map_provider():
                if self.static_map_pts is None:
                    return (None, None)
                return self.static_map_pts, self.static_map_cols

            origin_xy = (0.0, 0.0)
            grid_res_viser = self.grid_params.res_m
            floor_y_viser = 0.0

            for _ in range(50):  # ~2.5 s of retries
                grid_codes, meta, T_wr = self.grid_thread.get_grid()
                if grid_codes is not None and meta is not None:
                    grid_res_viser = float(meta.get("cell_size_m", self.grid_params.res_m))
                    if not meta.get("ego_centric", True) and "x0" in meta and "z_top" in meta:
                        H, W = grid_codes.shape[:2]
                        x0 = float(meta["x0"])
                        z_top = float(meta["z_top"])
                        z_min = z_top - H * grid_res_viser
                        origin_xy = (x0, z_min)
                    floor_y_viser = float(meta.get("floor_y_est", 0.0))
                    break
                time.sleep(0.05)

            self.viser_mirror = ViserMirrorThread(
                self.server,
                grid_thread=self.grid_thread,
                planner_thread=self.planner,
                pose_source=self.datastream,
                origin_xy=origin_xy,
                grid_res_m=grid_res_viser,
                floor_y=floor_y_viser,
                hz=10.0,
                grid_update_hz=5.0,
                map_update_hz=5.0,
                map_provider=map_provider,
                static_map_once=True,
                robot_radius_m=self.grid_params.robot_radius_m,
                preview_source=self._preview_source,
            )
            self.viser_mirror.start()

        if self.path_thread is None:
            self.path_thread = threading.Thread(target=self._path_sender_loop, daemon=True)
            self.path_thread.start()

        return True


    def _auto_path_step_m(self) -> float:
        """Choose a reasonable densify step from the current grid resolution.

        - Uses the live grid meta if available (cell_size_m).
        - Falls back to Grid2DParams.res_m.
        """
        cell = float(getattr(self.grid_params, "res_m", 0.05))
        try:
            if self.grid_thread is not None:
                _, meta, _ = self.grid_thread.get_grid()
                if isinstance(meta, dict):
                    cell = float(meta.get("cell_size_m", cell))
        except Exception as e:
            print(f"[slam_node_new] Failed to get cell_size_m: {e}")

        # Heuristic: ~2 cells per waypoint, clamped.
        return float(np.clip(2.0 * cell, 0.05, 0.15))


    def _densify_path(self, path_world):
        if not path_world or len(path_world) < 2:
            return path_world
        step = self.path_step_m
        if step is None:
            step = self._auto_path_step_m()
        if step <= 0.0:
            return path_world
        out = [path_world[0]]
        for (x0, z0), (x1, z1) in zip(path_world, path_world[1:]):
            dx = float(x1) - float(x0)
            dz = float(z1) - float(z0)
            dist = float(np.hypot(dx, dz))
            if dist <= step:
                out.append((float(x1), float(z1)))
                continue
            n = max(1, int(dist / step))
            for i in range(1, n):
                t = i / float(n)
                out.append((float(x0 + t * dx), float(z0 + t * dz)))
            out.append((float(x1), float(z1)))
        return out

    def _state_monitor_loop(self):
        while self.running:
            try:
                self.latest_map = self.map_manager.get_map()
            except Exception:
                self.latest_map = None

            if self.grid_thread is not None:
                try:
                    self.latest_grid = self.grid_thread.get_grid()
                except Exception:
                    self.latest_grid = None

            if self.planner is not None:
                try:
                    self.latest_path = self.planner.get_latest_path_world()
                except Exception:
                    self.latest_path = None

            time.sleep(0.2)

    def _reset_cone_e_client(self):
        try:
            # if RPCClient has a close(), call it; otherwise just drop it
            if hasattr(self.cone_e_client, "close"):
                self.cone_e_client.close()
        except Exception as e:
            print(f"[slam_node_new] Failed to close cone_e_client: {e}")
        self.cone_e_client = RPCClient(self.cone_e_host, self.cone_e_port)


    def _path_sender_loop(self):
        last_sent = None
        last_fail_t = 0.0

        while self.running:
            time.sleep(0.1)
            if self.planner is None or not self.nav_initialized:
                continue

            try:
                path_world = self.planner.get_latest_path_world()
            except Exception as e:
                path_world = None
                now = time.time()
                if now - last_fail_t > 1.0:
                    print(f"[slam_node_new] planner get_latest_path_world failed: {e}")
                    last_fail_t = now

            if not path_world:
                continue

            path_dense = self._densify_path(path_world)

            # Normalize to plain Python floats (helps serialization + consistent equality)
            path_dense = [(float(x), float(z)) for (x, z) in path_dense]

            if path_dense == last_sent:
                continue

            # ---- RPC call (REQ sockets must be serialized) ----
            try:
                with self._cone_rpc_lock:
                    self.cone_e_client.follow_path(path_dense)
                last_sent = list(path_dense)
                self.latest_path = path_dense

            except Exception as e:
                msg = str(e)
                now = time.time()

                # Don't spam console
                if now - last_fail_t > 1.0:
                    print(f"[slam_node_new] follow_path RPC failed: {e}")
                    last_fail_t = now

                # EFSM / stuck REQ socket -> recreate client
                if "Operation cannot be accomplished in current state" in msg:
                    print("[slam_node_new] RPC socket stuck (EFSM). Resetting RPCClient...")
                    with self._cone_rpc_lock:
                        self._reset_cone_e_client()

                # allow retry later
                last_sent = None
                time.sleep(0.25)


    def _log_status(self):
        curr_map, poses = self.map_manager.get_state()
        npts = (len(curr_map) if curr_map is not None else 0)
        print(f"\r[slam_node_new] points={npts:,}  poses={len(poses)}", end="")

        if self.map_manager.last_error:
            print("\n[slam_node_new] MapManager error:", self.map_manager.last_error)
            self.map_manager.last_error = None


def main():
    parser = argparse.ArgumentParser("SLAM node (pubsub ZED + MapManager + grid + A* + Viser)")
    parser.add_argument(
        "--hz",
        type=float,
        default=3.0,
        help="target mapping rate (Hz); 0 = as fast as possible",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="stop after N seconds (0 = run until Ctrl+C)",
    )
    parser.add_argument(
        "--load",
        action="store_true",
        help="whether to load previous map instead of starting new mapping",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="whether to save the map on exit",
    )
    parser.add_argument(
        "--map-path",
        type=str,
        default=None,
        help="optional .npz path to save/load map",
    )
    parser.add_argument(
        "--cone-e-host",
        type=str,
        default=CONE_E_RPC_HOST,
        help="ConeE RPC host (follow_path via RPC)",
    )
    parser.add_argument(
        "--cone-e-port",
        type=int,
        default=CONE_E_RPC_PORT,
        help="ConeE RPC port (follow_path via RPC)",
    )
    parser.add_argument(
        "--path-step-m",
        type=float,
        default=None,
        help="dense waypoint spacing for follow_path (meters; 0 disables)",
    )

    parser.add_argument(
        "--zed-up-axis",
        type=str,
        default="y",
        choices=["y", "z"],
        help="up axis for incoming ZED frames (y=default, z=swap Y/Z into Y-up)",
    )
    args = parser.parse_args()

    slam = Slam(
        target_hz=args.hz,
        duration_s=args.duration,
        load_map=args.load,
        save_map=args.save,
        map_path=args.map_path,
        cone_e_host=args.cone_e_host,
        cone_e_port=args.cone_e_port,
        path_step_m=args.path_step_m,
        zed_up_axis=args.zed_up_axis,
    )

    slam.run()


if __name__ == "__main__":
    main()
