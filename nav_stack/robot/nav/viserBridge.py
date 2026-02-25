#!/usr/bin/env python3
"""
Lightweight Viser integration for a dynamic pipeline (grid + planner + pose).

Functionality preserved:
- Grid overlay image (FREE / OCCUPIED / INFLATED / UNKNOWN)
- Click-to-goal (raycast to floor, world->grid, reject obstacles)
- Path polyline + waypoint dots
- Optional lookahead point dot
- Optional dynamic RGB map pointcloud (throttled)
- Robot pose dot + footprint ring + frame axes
- Commanded velocity arrow (from vel_source)
- Logs markdown panel + global VISER_LOG_FN hook
- DynaMem query marker (dot + optional label)

Plus:
- Robot/camera-facing cone marker (uncommented and corrected).
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple
from collections import deque
import threading
import time
import math
import numpy as np

from .pathPlanning import (
    rc_to_world_xz_world,
    world_xz_to_rc_world,
    UNKNOWN,
    FREE,
    OCCUPIED,
    INFLATED,
)

try:
    import viser
except Exception as e:
    viser = None
    _import_error = e

VISER_LOG_FN = None


# ----------------------------- small utilities -----------------------------
def ray_plane_y_intersect(
    ray_o: np.ndarray, ray_d: np.ndarray, y: float
) -> Optional[np.ndarray]:
    """Return intersection point of ray (o + t*d) with plane y=const, or None."""
    denom = float(ray_d[1])
    if abs(denom) < 1e-8:
        return None
    t = (float(y) - float(ray_o[1])) / denom
    if t < 0.0:
        return None
    return ray_o + t * ray_d


def world_to_grid(
    xw: float, zw: float, origin_xy: Tuple[float, float], res: float
) -> Tuple[int, int]:
    ox, oy = origin_xy
    r = int(np.floor((zw - oy) / res))
    c = int(np.floor((xw - ox) / res))
    return r, c


# ----------------------------- public API -----------------------------
def start_viser_server(host: str = "0.0.0.0", port: int = 8099):
    if viser is None:
        print(f"[Viser] Not available: {_import_error}")
        return None

    server = viser.ViserServer(host=host, port=port)
    try:
        server.scene.set_up_direction("+y")
    except Exception as e:
        print(f"[Viser] Failed to set up direction: {e}")

    print(f"[Viser] http://{host}:{port}")
    return server


class ViserMirrorThread:
    """
    Mirrors dynamic grid, path, robot pose to Viser at ~hz.
    Also registers click-to-goal and provides a log panel + query marker API.
    """

    def __init__(
        self,
        server,
        *,
        grid_thread,  # .get_grid() -> (grid_codes, meta, T_world_robot)
        planner_thread,  # .get_latest_path_world() and .set_goal_world(x,z)
        pose_source,  # .get_pose() -> (translation[3], yaw, T_world_robot)
        origin_xy: Tuple[float, float],
        grid_res_m: float,
        floor_y: float = 0.0,
        hz: float = 10.0,
        grid_update_hz: Optional[float] = None,
        map_update_hz: Optional[float] = None,
        map_provider: Optional[
            Callable[[], Tuple[Optional[np.ndarray], Optional[np.ndarray]]]
        ] = None,  # -> (points Nx3, colors Nx3 u8 or float)
        static_map_once: bool = False,
        on_confirm_move: Optional[Callable[[], None]] = None,
        vel_source: Optional[Callable[[], tuple[np.ndarray, float]]] = None,
        preview_source=None,
        robot_radius_m: float = 0.3,
    ):
        self.server = server
        self.grid_thread = grid_thread
        self.planner = planner_thread
        self.pose_source = pose_source

        self.origin_xy = origin_xy
        self.grid_res = float(grid_res_m)
        self.floor_y = float(floor_y)

        self.dt = 1.0 / max(1e-3, float(hz))
        self._grid_dt = (
            0.0
            if grid_update_hz is None
            else (1.0 / max(1e-3, float(grid_update_hz)) if grid_update_hz > 0 else 0.0)
        )
        self._map_dt = (
            0.0
            if map_update_hz is None
            else (1.0 / max(1e-3, float(map_update_hz)) if map_update_hz > 0 else 0.0)
        )

        self.map_provider = map_provider
        self._static_map_once = bool(static_map_once)
        self._static_map_logged = False

        self.on_confirm_move = on_confirm_move
        self.vel_source = vel_source
        self.preview_source = preview_source
        self.robot_radius_m = max(0.0, float(robot_radius_m))

        # confirm-flow state (kept, though not used unless you wire UI buttons)
        self._pending_goal = None  # (xw, zw, r, c)
        self._goal_planned = False
        self._last_pose_xz = None  # (xw, zw)
        self._last_pose_t = None
        self._vel_xz_lp = np.zeros(2, dtype=np.float32)

        # ---- log buffer + panel ----
        self._log_lines = deque(maxlen=200)
        try:
            self._log_panel = self.server.gui.add_markdown(
                "**Logs (last 200 lines)**\n\n_(waiting for messages)_"
            )
        except Exception:
            self._log_panel = None

        global VISER_LOG_FN
        VISER_LOG_FN = self._log_to_viser

        # ---- DynaMem query marker (thread-safe) ----
        self._query_lock = threading.Lock()
        self._query_marker = None  # (x, y|None, z, label|None) or None
        self._query_marker_dirty = False
        # ---- DynaMem snapped navigation-goal marker (thread-safe) ----
        self._nav_goal_marker = None  # (x, y|None, z, label|None) or None
        self._nav_goal_marker_dirty = False

        # ---- thread control ----
        self._thr = None
        self._stop = threading.Event()

        # ---- throttling / signatures ----
        self._last_grid_t = 0.0
        self._last_map_t = 0.0
        self._last_path_sig = None

        self._last_pose_xz = None      # np.array([x,z])
        self._last_pose_t  = None      # float seconds
        self._vel_xz_lp     = np.zeros(2, dtype=np.float32) 
        self._map_err_once = False

        if self.server is not None and viser is not None:
            self._register_click_to_goal()

    def _T_zup_to_yup(self, T_zup: np.ndarray) -> np.ndarray:
        """
        Convert a homogeneous transform expressed in a Z-up world to a Y-up world.

        Assumed mapping (common):
        (x, y, z)_yup = (x, z, -y)_zup
        This is a -90° rotation about +X.
        """
        S = np.array([
            [1, 0,  0, 0],
            [0, 0,  1, 0],
            [0,-1,  0, 0],
            [0, 0,  0, 1],
        ], dtype=np.float32)
        # similarity transform: represent same rigid transform in new basis
        return S @ T_zup @ S.T


    def _draw_cone_from_dir_xz(
    self,
    *,
    name: str,
    base_pos: np.ndarray,          # (3,)
    dir_xz: np.ndarray,            # (2,) world direction [dx, dz]
    length: float = 0.6,
    radius: float = 0.18,
    color=(66, 133, 245),
    opacity: float = 0.9,
    n_base: int = 14,
    ):
        d = np.asarray(dir_xz, dtype=np.float32).reshape(2)
        n = float(np.linalg.norm(d))
        if n < 1e-6:
            return
        d /= n

        fwd = np.array([d[0], 0.0, d[1]], dtype=np.float32)
        fwd[1]  = 0.0  # world XZ
        fwd = -fwd
        up  = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        side = np.cross(up, fwd)
        sn = float(np.linalg.norm(side))
        if sn < 1e-6:
            return
        side /= sn
        up2 = np.cross(fwd, side)

        tip = base_pos + fwd * float(length)

        angles = np.linspace(0.0, 2.0 * np.pi, int(n_base), endpoint=False, dtype=np.float32)
        circle = base_pos + (
            np.cos(angles)[:, None] * side[None, :] + np.sin(angles)[:, None] * up2[None, :]
        ) * float(radius)

        vertices = np.vstack([tip.reshape(1, 3), circle]).astype(np.float32)

        faces = []
        for i in range(1, 1 + circle.shape[0]):
            j = i + 1 if i + 1 < 1 + circle.shape[0] else 1
            faces.append([0, i, j])
        faces = np.asarray(faces, dtype=np.int32)

        self.server.scene.add_mesh_simple(
            name=name,
            vertices=vertices,
            faces=faces,
            color=color,
            opacity=float(opacity),
            flat_shading=True,
        )


    # ----------------------------- lifecycle -----------------------------
    def start(self):
        if self.server is None or viser is None:
            print("[Viser] Not started; server missing.")
            return
        if self._thr and self._thr.is_alive():
            return
        self._stop.clear()
        self._thr = threading.Thread(target=self._loop, name="ViserMirror", daemon=True)
        self._thr.start()

    def stop(self):
        self._stop.set()
        if self._thr:
            self._thr.join(timeout=1.0)
            self._thr = None

    # ----------------------------- logs -----------------------------
    def _log_to_viser(self, msg: str) -> None:
        """Append a line to the Viser Logs panel."""
        self._log_lines.append(str(msg))
        if self._log_panel is None:
            return
        text = "**Logs (last {} lines)**\n\n```text\n{}\n```".format(
            len(self._log_lines),
            "\n".join(self._log_lines),
        )
        try:
            self._log_panel.content = text
        except Exception as e:
            print(f"[Viser] Failed to update log panel content: {e}")

    # ----------------------------- query marker API -----------------------------
    def set_query_marker_world(
        self, x: float, z: float, y: Optional[float] = None, label: Optional[str] = None
    ):
        """Set/overwrite the red marker in world coords. Thread-safe."""
        with self._query_lock:
            self._query_marker = (float(x), float(y) if y is not None else None, float(z), label)
            self._query_marker_dirty = True

    def clear_query_marker(self):
        """Hide marker. Thread-safe."""
        with self._query_lock:
            self._query_marker = None
            self._query_marker_dirty = True

    def set_nav_goal_marker_world(
        self, x: float, z: float, y: Optional[float] = None, label: Optional[str] = None
    ):
        """Set/overwrite the snapped navigation goal marker in world coords. Thread-safe."""
        with self._query_lock:
            self._nav_goal_marker = (float(x), float(y) if y is not None else None, float(z), label)
            self._nav_goal_marker_dirty = True

    # ----------------------------- main loop -----------------------------
    def _loop(self):
        while not self._stop.is_set():
            t0 = time.time()

            # 1) Grid overlay (throttled)
            grid_codes, meta, _Twr = (None, {}, None)
            try:
                grid_codes, meta, _Twr = self.grid_thread.get_grid()
            except Exception as e:
                print(f"[Viser] Failed to get grid: {e}")

            if grid_codes is not None and meta is not None:
                if self._grid_dt <= 0.0 or (t0 - self._last_grid_t) >= self._grid_dt:
                    self._mirror_grid_once()
                    self._last_grid_t = t0

            # 2) Path + waypoints + lookahead
            self._mirror_path_once()

            # overlay from base controler
            self._mirror_preview_once()

            # 3) Query marker (dot + label)
            self._mirror_query_marker_once()
            # 3b) Snapped nav-goal marker (dot + label)
            self._mirror_nav_goal_marker_once()

            # 4) Robot pose + frame + footprint + velocity + camera-facing cone
            self._mirror_robot_once()

            # 5) Optional dynamic map points (throttled)
            self._mirror_map_points_once(t0)

            # sleep to target rate
            dt = time.time() - t0
            if dt < self.dt:
                time.sleep(self.dt - dt)

    # ----------------------------- grid -----------------------------
    def _mirror_grid_once(self):
        grid_codes, meta, _ = self.grid_thread.get_grid()
        if grid_codes is None or meta is None:
            return

        H, W = grid_codes.shape[:2]
        cs = float(meta.get("cell_size_m", self.grid_res))

        # RGBA overlay image
        img = np.zeros((H, W, 4), dtype=np.uint8)
        img[grid_codes == FREE] = np.array([255, 255, 255, 50], dtype=np.uint8)
        img[grid_codes == OCCUPIED] = np.array([240, 40, 80, 150], dtype=np.uint8)
        img[grid_codes == INFLATED] = np.array([240, 40, 80, 75], dtype=np.uint8)
        img[grid_codes == UNKNOWN] = np.array([20, 20, 20, 50], dtype=np.uint8)

        center_x = self.origin_xy[0] + (W * cs) / 2.0
        center_z = self.origin_xy[1] + (H * cs) / 2.0

        # Rotate image plane to lie on XZ (Viser image defaults on XY)
        self.server.scene.add_image(
            "grid/overlay_image",
            image=img,
            render_width=W * cs,
            render_height=H * cs,
            position=(center_x, self.floor_y + 0.1, center_z),
            wxyz=(0.7071068, -0.7071068, 0.0, 0.0),  # -90 deg about X
        )

    # ----------------------------- path -----------------------------
    def _mirror_path_once(self):
        try:
            path_world = self.planner.get_latest_path_world()
        except Exception:
            path_world = []

        if path_world and len(path_world) >= 2:
            # Keep your original "sig" behavior
            path_sig = (len(path_world), path_world[-1])
            if path_sig != self._last_path_sig:
                self._last_path_sig = path_sig

                pts = np.array(
                    [[x, self.floor_y + 0.10, z] for (x, z) in path_world],
                    dtype=np.float32,
                )
                segs = np.stack([pts[:-1], pts[1:]], axis=1)

                self.server.scene.add_line_segments(
                    name="path/a_star",
                    points=segs,
                    colors=(0, 255, 0),
                    line_width=3.5,
                )

                pts_wp = pts.copy()
                cols_wp = np.tile(
                    np.array([[0.0, 1.0, 0.4]], dtype=np.float32), (len(path_world), 1)
                )
                self.server.scene.add_point_cloud(
                    name="path/a_star_points",
                    points=pts_wp,
                    colors=cols_wp,
                    point_size=0.03,
                    point_shape="circle",
                )

            # Lookahead dot
            lookahead_world = None
            try:
                if hasattr(self.planner, "get_latest_lookahead_world"):
                    lookahead_world = self.planner.get_latest_lookahead_world()
            except Exception:
                lookahead_world = None

            if lookahead_world is not None:
                lx, lz = float(lookahead_world[0]), float(lookahead_world[1])
                lp = np.array([[lx, self.floor_y + 0.10, lz]], dtype=np.float32)
                lc = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
                self.server.scene.add_point_cloud(
                    name="path/lookahead",
                    points=lp,
                    colors=lc,
                    point_size=0.06,
                )

    def _mirror_preview_once(self):
        if self.preview_source is None:
            return

        dbg = self.preview_source()
        

        if not dbg:
            return

        path = dbg.get("path_world") or []
        look = dbg.get("lookahead_xz", None)
        pose = dbg.get("pose_xz", None)
        yaw = dbg.get("yaw", None)
        yaw_des = dbg.get("yaw_des", None)

        # ---- (1) pursuit polyline (orange) ----
        if path and len(path) >= 2:
            pts = np.array([[x, self.floor_y + 0.13, z] for (x, z) in path], dtype=np.float32)
            segs = np.stack([pts[:-1], pts[1:]], axis=1)
            cols = np.tile(np.array([[1.0, 0.63, 0.16]], dtype=np.float32), (segs.shape[0], 2, 1))

            self.server.scene.add_line_segments(
                name="debug/pursuit_polyline",
                points=segs,
                colors=cols,   # orange
                line_width=3.0,
            )

        # ---- (2) lookahead point (cyan) ----
        if look is not None:
            lx, lz = float(look[0]), float(look[1])
            self.server.scene.add_point_cloud(
                "debug/lookahead",
                points=np.array([[lx, self.floor_y + 0.16, lz]], dtype=np.float32),
                colors=np.array([[0.2, 1.0, 1.0]], dtype=np.float32),
                point_size=0.08,
            )

        # ---- (3) base yaw + desired yaw arrows ----
        if pose is not None and yaw is not None:
            x, z = float(pose[0]), float(pose[1])

            def _yaw_arrow(name: str, ang: float, color_rgb):
                dx = math.sin(ang)
                dz = math.cos(ang)
                p0 = np.array([x, self.floor_y + 0.18, z], dtype=np.float32)
                p1 = p0 + np.array([dx, 0.0, dz], dtype=np.float32) * 0.5

                self.server.scene.add_line_segments(
                    name=name,
                    points=np.array([[p0, p1]], dtype=np.float32),
                    colors=color_rgb,
                    line_width=6.0,
                )

            _yaw_arrow("debug/yaw_base", float(yaw), (255, 0, 255))  # magenta

            if yaw_des is not None:
                _yaw_arrow("debug/yaw_des", float(yaw_des), (255, 255, 0))  # yellow


    # ----------------------------- query marker -----------------------------
    def _mirror_query_marker_once(self):
        marker = None
        dirty = False
        with self._query_lock:
            dirty = self._query_marker_dirty
            if dirty:
                marker = self._query_marker
                self._query_marker_dirty = False

        if not dirty:
            return

        # If marker cleared, overwrite with empty geometry to hide
        if marker is None:
            self.server.scene.add_point_cloud(
                name="dynamem/query_hit",
                points=np.zeros((0, 3), dtype=np.float32),
                colors=np.zeros((0, 3), dtype=np.float32),
                point_size=0.05,
            )
            # Clear label by moving it off + empty text (cheap "hide")
            try:
                self.server.scene.add_label(
                    name="dynamem/query_label",
                    text="",
                    position=np.array([0.0, -9999.0, 0.0], dtype=np.float32),
                    font_size_mode="scene",
                    font_scene_height=0.06,
                    depth_test=False,
                    anchor="bottom-center",
                )
            except Exception as e:
                print(f"[Viser] Failed to clear query label: {e}")

    def _mirror_nav_goal_marker_once(self):
        marker = None
        dirty = False
        with self._query_lock:
            dirty = self._nav_goal_marker_dirty
            if dirty:
                marker = self._nav_goal_marker
                self._nav_goal_marker_dirty = False

        if not dirty:
            return

        # If marker cleared, overwrite with empty geometry to hide
        if marker is None:
            self.server.scene.add_point_cloud(
                name="dynamem/nav_goal",
                points=np.zeros((0, 3), dtype=np.float32),
                colors=np.zeros((0, 3), dtype=np.float32),
                point_size=0.05,
            )
            # Clear label by moving it off + empty text (cheap "hide")
            try:
                self.server.scene.add_label(
                    name="dynamem/nav_goal_label",
                    text="",
                    position=np.array([0.0, -9999.0, 0.0], dtype=np.float32),
                    font_size_mode="scene",
                    font_scene_height=0.06,
                    depth_test=False,
                    anchor="bottom-center",
                )
            except Exception as e:
                print(f"[Viser] Failed to clear nav goal label: {e}")
            return

        nx, ny, nz, nlabel = marker
        # default height slightly above floor so it doesn't z-fight
        if ny is None:
            ny = self.floor_y + 0.10
        np_pt = np.array([[nx, ny, nz]], dtype=np.float32)
        nc = np.array([[0.0, 0.4, 1.0]], dtype=np.float32)

        self.server.scene.add_point_cloud(
            name="dynamem/nav_goal",
            points=np_pt,
            colors=nc,
            point_size=0.05,
        )

        if nlabel:
            self.server.scene.add_label(
                name="dynamem/nav_goal_label",
                text=str(nlabel),
                position=np.array([nx, ny + 0.10, nz], dtype=np.float32),
                font_size_mode="scene",
                font_scene_height=0.06,
                depth_test=False,
                anchor="bottom-center",
            )
        else:
            try:
                self.server.scene.add_label(
                    name="dynamem/nav_goal_label",
                    text="",
                    position=np.array([0.0, -9999.0, 0.0], dtype=np.float32),
                    font_size_mode="scene",
                    font_scene_height=0.06,
                    depth_test=False,
                    anchor="bottom-center",
                )
            except Exception as e:
                print(f"[Viser] Failed to clear nav goal label: {e}")
            return

        mx, my, mz, mlabel = marker
        if my is None:
            my = self.floor_y + 0.12

        mp = np.array([[mx, my, mz]], dtype=np.float32)
        mc = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

        self.server.scene.add_point_cloud(
            name="dynamem/query_hit",
            points=mp,
            colors=mc,
            point_size=0.05,
        )

        if mlabel:
            self.server.scene.add_label(
                name="dynamem/query_label",
                text=str(mlabel),
                position=np.array([mx, my + 0.10, mz], dtype=np.float32),
                font_size_mode="scene",
                font_scene_height=0.06,
                depth_test=False,
                anchor="bottom-center",
            )
        else:
            # Clear label if no label provided
            try:
                self.server.scene.add_label(
                    name="dynamem/query_label",
                    text="",
                    position=np.array([0.0, -9999.0, 0.0], dtype=np.float32),
                    font_size_mode="scene",
                    font_scene_height=0.06,
                    depth_test=False,
                    anchor="bottom-center",
                )
            except Exception as e:
                print(f"[Viser] Failed to clear query label: {e}")

    # ----------------------------- robot -----------------------------
    def _mirror_robot_once(self):
        try:
            trans, yaw, T_wr = self.pose_source.get_pose()
            x = float(trans[0])
            z = float(trans[2])
            p = np.asarray([x, self.floor_y + 0.20, z], dtype=np.float32)

            fwd = T_wr[[0, 2], 0]
            norm = np.linalg.norm(fwd)
            if norm > 1e-6:
                fwd = fwd / norm
            c, s = fwd[0], fwd[1]
            R_2d = np.array([[c, -s], [s, c]], dtype=np.float32)
            arrow_len = 0.60
            w = 0.60

            pts_local = np.array([
                [0.0,        0.0],   # Tip (At Robot Center)
                [arrow_len,  w/2],   # Far Left
                [arrow_len, -w/2],   # Far Right
            ], dtype=np.float32)
            pts_2d = pts_local @ R_2d.T + p[[0, 2]]
            h_arrow = self.floor_y + 0.20
            v_world = np.insert(pts_2d, 1, h_arrow, axis=1)
            self.server.scene.add_mesh_simple(
                name="/robot/compass_arrow",
                vertices=v_world,
                faces=np.array([[0, 1, 2], [0, 2, 1]], dtype=np.int32),
                color=(66, 133, 245),
                opacity=0.6,
                flat_shading=True,
            )
            lines = np.array([
                [v_world[0], v_world[1]],
                [v_world[0], v_world[2]],
                [v_world[1], v_world[2]],
            ], dtype=np.float32)
            
            self.server.scene.add_line_segments(
                "/robot/compass_lines",
                points=lines,
                colors=(66, 133, 245),
                line_width=4.0,
            )

            self.server.scene.add_point_cloud(
                name="/robot/pose",
                points=p.reshape(1, 3),
                colors=np.array([[0.26, 0.52, 0.96]], np.float32),
                point_size=0.15,
                point_shape="circle",
            )

            # draw robot pose axes
            # rot = T_wr[:3, :3].astype(np.float32)
            # axis_len = 1.0
            # x_tip = p + rot @ np.array([axis_len, 0, 0], dtype=np.float32)
            # y_tip = p + rot @ np.array([0, axis_len, 0], dtype=np.float32)
            # z_tip = p + rot @ np.array([0, 0, axis_len], dtype=np.float32)

            # axis_lines = np.stack([np.stack([p, x_tip]),
            #                     np.stack([p, y_tip]),
            #                     np.stack([p, z_tip])], axis=0)

            # base_colors = np.eye(3, dtype=np.float32) * 255.0
            # point_colors = np.tile(base_colors[:, None, :], (1, 2, 1))

            # self.server.scene.add_line_segments(
            #     name="/robot/frame_axes",
            #     points=axis_lines,
            #     colors=point_colors,
            #     line_width=4.0,
            # )
        except Exception as e:
            print("[Viser] Failed to mirror robot pose:", e)
            return

    def _quat_xyzw_to_R(self, qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
        """Quaternion (x,y,z,w) -> 3x3 rotation. No scipy needed."""
        q = np.array([qx, qy, qz, qw], dtype=np.float32)
        n = float(np.linalg.norm(q))
        if n < 1e-8:
            return np.eye(3, dtype=np.float32)
        q /= n
        x, y, z, w = map(float, q)

        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z

        return np.array(
            [
                [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)],
                [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
                [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)],
            ],
            dtype=np.float32,
        )

    def _quatxyz_to_T(self, qx, qy, qz, qw, x, y, z) -> np.ndarray:
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = self._quat_xyzw_to_R(float(qx), float(qy), float(qz), float(qw))
        T[:3, 3] = np.array([float(x), float(y), float(z)], dtype=np.float32)
        return T

    def _yaw_from_T(self, T: np.ndarray) -> float:
        """Yaw around +Y for RIGHT_HANDED_Y_UP."""
        Rm = T[:3, :3]
        return float(math.atan2(float(Rm[0, 2]), float(Rm[0, 0])))

    def _as_flat_float_array(self, out) -> Optional[np.ndarray]:
        """
        Try to turn 'out' into a 1D float array.
        Handles: list/tuple/np.ndarray, including tuple(list_of_floats,) etc.
        """
        if out is None:
            return None

        # Unwrap single-element tuple/list (common in some pub/sub wrappers)
        if isinstance(out, (tuple, list)) and len(out) == 1:
            out = out[0]

        # If dict, not here
        if isinstance(out, dict):
            return None

        try:
            arr = np.asarray(out, dtype=np.float32).reshape(-1)
            if arr.size == 0:
                return None
            return arr
        except Exception:
            return None

    def _extract_pose_any(self, out):
        """
        Accept many formats and return:
        trans (3,), yaw (float), T_wr (4,4) or None, T_wc (4,4) or None
        Supports your zed/pose 19-float list:
        base(0:7) + cam(7:14) + base_pose(14:18) + ts(18)
        """
        # Case A: already (trans, yaw, T_wr)
        if isinstance(out, (tuple, list)) and len(out) >= 2:
            # Try: (trans, yaw, T)
            trans_candidate = out[0]
            yaw_candidate = out[1]
            T_candidate = out[2] if len(out) >= 3 else None

            try:
                trans = np.asarray(trans_candidate, dtype=np.float32).reshape(-1)
                if trans.size >= 3:
                    trans = trans[:3]
                    yaw = float(yaw_candidate) if yaw_candidate is not None else None
                    T_wr = None
                    if isinstance(T_candidate, np.ndarray):
                        Ta = np.asarray(T_candidate)
                        if Ta.ndim == 2 and Ta.shape[0] >= 3 and Ta.shape[1] >= 4:
                            # ensure 4x4
                            T_wr = np.eye(4, dtype=np.float32)
                            T_wr[:Ta.shape[0], :Ta.shape[1]] = Ta[:4, :4]
                            if yaw is None:
                                yaw = self._yaw_from_T(T_wr)
                    return trans, yaw, T_wr, None
            except Exception as e:
                print(f"[ViserBridge] Failed to extract pose: {e}")

        # Case B: dict format
        if isinstance(out, dict):
            # common keys
            T_wr = None
            if "base_pose_6DOF" in out:
                try:
                    Ta = np.asarray(out["base_pose_6DOF"], dtype=np.float32)
                    if Ta.ndim == 2 and Ta.shape[0] >= 3 and Ta.shape[1] >= 4:
                        T_wr = np.eye(4, dtype=np.float32)
                        T_wr[:Ta.shape[0], :Ta.shape[1]] = Ta[:4, :4]
                except Exception:
                    T_wr = None

            trans = None
            yaw = None
            if "base_pose" in out:
                try:
                    bp = np.asarray(out["base_pose"], dtype=np.float32).reshape(-1)
                    if bp.size >= 4:
                        trans = bp[:3]
                        yaw = float(bp[3])
                except Exception as e:
                    print(f"[ViserBridge] Failed to extract base_pose: {e}")

            if trans is None and T_wr is not None:
                trans = T_wr[:3, 3].copy()
            if yaw is None and T_wr is not None:
                yaw = self._yaw_from_T(T_wr)

            return trans, yaw, T_wr, None

        # Case C: flat array (your 19-float pose_quat_xyz, or 7-float qt, etc.)
        arr = self._as_flat_float_array(out)
        if arr is None:
            return None, None, None, None

        # Your current publisher: 19 floats
        if arr.size >= 19:
            b = arr[0:7]      # base quat+xyz
            c = arr[7:14]     # cam  quat+xyz
            pose_base = arr[14:18]  # [tx,ty,tz,yaw]
            T_wr = self._quatxyz_to_T(b[0], b[1], b[2], b[3], b[4], b[5], b[6])
            T_wc = self._quatxyz_to_T(c[0], c[1], c[2], c[3], c[4], c[5], c[6])
            trans = T_wr[:3, 3].copy()
            yaw = float(pose_base[3])
            return trans, yaw, T_wr, T_wc

        # Older/other: 7 floats [qx,qy,qz,qw,x,y,z]
        if arr.size == 7:
            T_wr = self._quatxyz_to_T(arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6])
            trans = T_wr[:3, 3].copy()
            yaw = self._yaw_from_T(T_wr)
            return trans, yaw, T_wr, None

        # 4 floats [x,y,z,yaw]
        if arr.size == 4:
            trans = arr[:3].copy()
            yaw = float(arr[3])
            # create a T from yaw for axes/cone
            cy, sy = math.cos(yaw), math.sin(yaw)
            T_wr = np.array([[cy, 0, sy, trans[0]],
                            [0,  1, 0,  trans[1]],
                            [-sy,0, cy, trans[2]],
                            [0,  0, 0,  1]], dtype=np.float32)
            return trans, yaw, T_wr, None

        # 3 floats [x,y,z]
        if arr.size == 3:
            trans = arr.copy()
            return trans, 0.0, None, None

        return None, None, None, None

    # ----------------------------- map points -----------------------------
    def _mirror_map_points_once(self, t0: float):
        if self.map_provider is None:
            return
        if self._static_map_once and self._static_map_logged:
            return

        if self._map_dt > 0.0 and (t0 - self._last_map_t) < self._map_dt:
            return

        try:
            P, C = self.map_provider()
            if P is None or len(P) == 0:
                return

            if P.shape[0] > 200_000:
                idx = np.random.choice(P.shape[0], 200_000, replace=False)
                Pshow = P[idx]
                if C is not None:
                    Csel = C[idx]
                    Cshow = (Csel / 255.0) if (Csel.dtype == np.uint8) else Csel
                else:
                    Cshow = None
            else:
                Pshow = P
                if C is not None:
                    Cshow = (C / 255.0) if (C.dtype == np.uint8) else C
                else:
                    Cshow = None

            self.server.scene.add_point_cloud(
                "map/points",
                points=Pshow,
                colors=Cshow,
                point_size=0.03,
            )
            if self._static_map_once:
                self._static_map_logged = True
        except Exception as e:
            print(f"[ViserBridge] Failed to mirror map points: {e}")
        finally:
            self._last_map_t = t0

    # ----------------------------- click-to-goal -----------------------------
    def _register_click_to_goal(self):
        lock = threading.Lock()

        @self.server.on_scene_pointer(event_type="click")
        def _on_click(ev):
            with lock:
                grid_codes, meta, _ = self.grid_thread.get_grid()
                if grid_codes is None or meta is None:
                    print("[Viser] No grid/meta yet in click handler.")
                    return

                H, W = grid_codes.shape[:2]

                ray_o = np.array(ev.ray_origin, dtype=np.float32)
                ray_d = np.array(ev.ray_direction, dtype=np.float32)
                hit = ray_plane_y_intersect(ray_o, ray_d, self.floor_y)
                if hit is None:
                    print("[Viser] Click ray did not hit floor plane.")
                    return

                xw, zw = float(hit[0]), float(hit[2])

                # Use same world->grid mapping as planner
                r, c = world_xz_to_rc_world(xw, zw, meta)
                print(f"[Viser] Click at world=({xw:.2f},{zw:.2f}) -> grid (r={r}, c={c})")

                if not (0 <= r < H and 0 <= c < W):
                    print("[Viser] Click outside grid.")
                    return

                cell = int(grid_codes[r, c])
                if cell >= 1:
                    print("[Viser] Clicked an obstacle cell;")
                try:
                    self.planner.set_goal_world(xw, zw)
                    print(f"[Viser] Goal set @ world=({xw:.2f},{zw:.2f}) grid=(r={r},c={c})")
                except Exception as e:
                    print("[Viser] Failed to set goal:", e)
                    return

                # Set goal in world coords (planner will plan, follower will move)
                try:
                    self.planner.set_goal_world(xw, zw)
                    print(f"[Viser] Goal set @ world=({xw:.2f},{zw:.2f}) grid=(r={r},c={c})")
                except Exception as e:
                    print("[Viser] Failed to set goal:", e)

    # ----------------------------- (optional) confirm flow hooks -----------------------------
    def _on_confirm_point(self):
        if self._pending_goal is None:
            return
        xw, zw, r, c = self._pending_goal
        try:
            self.planner.set_goal_world(xw, zw)
            self._goal_planned = True
            print(f"[Viser] Planning path to world=({xw:.2f},{zw:.2f}) grid=(r={r},c={c}).")
        except Exception as e:
            print("[Viser] Failed to set planner goal:", e)

    def _on_confirm_path(self):
        if not self._goal_planned:
            return
        print("[Viser] Confirmed goal path; starting motion.")
        if self.on_confirm_move is not None:
            try:
                self.on_confirm_move()
            except Exception as e:
                print("[Viser] on_confirm_move callback failed:", e)
