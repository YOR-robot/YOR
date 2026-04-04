import math
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

from commlink import Subscriber
from loop_rate_limiters import RateLimiter

from .base_motor import Base

THOR_IP = "192.168.1.11"
ZED_PUB_PORT = 6000
POSE_TOPIC = "zed/pose"


# -----------------------------
# Pose / frame helpers
# -----------------------------
def xyzw_xyz_to_matrix(qt7):
    qt7 = np.asarray(qt7, dtype=np.float32).reshape(-1)
    if qt7.shape[0] < 7:
        raise ValueError(
            f"Expected 7 values [qx,qy,qz,qw,tx,ty,tz], got {qt7.shape}"
        )
    q = qt7[:4]
    t = qt7[4:7]
    R_mat = R.from_quat(q).as_matrix().astype(np.float32)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R_mat
    T[:3, 3] = t
    return T


_ZUP_TO_YUP = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


def _zup_to_yup_transform(T: np.ndarray) -> np.ndarray:
    return _ZUP_TO_YUP @ T @ _ZUP_TO_YUP.T


def get_cam_pose(_sub):
    pose_msg = _sub[POSE_TOPIC]
    return pose_msg[7:14]


def get_pose(sub):
    pose_msg = sub[POSE_TOPIC]
    base_qt7 = pose_msg[0:7]
    base_transform = xyzw_xyz_to_matrix(base_qt7)
    translation = base_transform[:3, 3].astype(np.float32)
    theta = float(np.arctan2(-base_transform[2, 0], base_transform[0, 0])) % (2 * np.pi)
    return translation, theta, base_transform


def _wrap_pi(a: float) -> float:
    return ((a + math.pi) % (2 * math.pi)) - math.pi


# -----------------------------
# Polyline helpers
# -----------------------------
def _cumlen(pts: list[np.ndarray]) -> tuple[np.ndarray, float]:
    if len(pts) < 2:
        s = np.array([0.0], dtype=float)
        return s, 0.0
    segs = np.linalg.norm(np.diff(np.vstack(pts), axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(segs)])
    return s, float(s[-1])


def _closest_point_on_polyline(
    x: float,
    y: float,
    pts: list[np.ndarray],
    s_cum: np.ndarray,
) -> tuple[float, np.ndarray, int, float]:
    p = np.array([x, y], dtype=float)
    best_d2, best = float("inf"), (0.0, pts[0], 0, 0.0)
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        ab = b - a
        L2 = float(ab @ ab) if np.any(ab) else 1e-12
        t = float(np.clip(((p - a) @ ab) / L2, 0.0, 1.0))
        q = a + t * ab
        d2 = float(np.sum((p - q) ** 2))
        if d2 < best_d2:
            best_d2 = d2
            s_here = float(s_cum[i] + t * np.linalg.norm(ab))
            best = (s_here, q, i, t)
    return best


def _point_at_s(
    pts: list[np.ndarray],
    s_cum: np.ndarray,
    s: float,
) -> tuple[np.ndarray, np.ndarray]:
    s = float(np.clip(s, 0.0, s_cum[-1]))
    i = int(np.searchsorted(s_cum, s, side="right") - 1)
    i = max(0, min(i, len(pts) - 2))
    a, b = pts[i], pts[i + 1]
    seg_L = float(s_cum[i + 1] - s_cum[i]) or 1e-12
    t = float((s - s_cum[i]) / seg_L)
    p = a + t * (b - a)
    tan = (
        (b - a) / np.linalg.norm(b - a)
        if np.linalg.norm(b - a) > 1e-9
        else np.array([1.0, 0.0])
    )
    return p, tan


# -----------------------------
# Heading helpers
# -----------------------------
def signed_angle_2d(v1, v2):
    return np.arctan2(v1[0] * v2[1] - v1[1] * v2[0], np.dot(v1, v2))


def _norm2(v, eps=1e-9):
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v), 0.0
    return v / n, n


def fwd_xz_from_T(T_base: np.ndarray, forward_col: int = 0) -> np.ndarray:
    R_bw = T_base[:3, :3]
    fwd_xz = np.array([R_bw[0, forward_col], R_bw[2, forward_col]], dtype=float)
    fwd_xz, _ = _norm2(fwd_xz)
    return fwd_xz


def heading_error_from_dir(
    T_base: np.ndarray,
    desired_dir_xz: np.ndarray,
    forward_col: int = 0,
    flip_sign: bool = True,
) -> float:
    fwd_xz = fwd_xz_from_T(T_base, forward_col=forward_col)
    ddir, n = _norm2(desired_dir_xz)
    if n < 1e-9:
        return 0.0
    e = float(signed_angle_2d(fwd_xz, ddir))
    return -e if flip_sign else e


def dir_from_yaw(yaw: float) -> np.ndarray:
    return np.array([math.sin(yaw), math.cos(yaw)], dtype=float)


def save_home_pose(translation, T_base):
    home = {
        "x": float(translation[0]),
        "y": float(translation[2]),
        "fwd_xz": fwd_xz_from_T(T_base, forward_col=0),
    }
    return home


# -----------------------------
# Control primitives
# -----------------------------
class PID:
    def __init__(
        self,
        kp: float,
        ki: float = 0.0,
        kd: float = 0.0,
        i_limit: float = 1.0,
        out_limit: float | None = None,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.i_limit = i_limit
        self.out_limit = out_limit
        self.i = 0.0
        self.prev_e = None

    def reset(self):
        self.i = 0.0
        self.prev_e = None

    def step(self, e: float, dt: float) -> float:
        if dt <= 0.0:
            dt = 1e-3

        self.i += e * dt
        if self.i_limit is not None:
            self.i = max(-self.i_limit, min(self.i, self.i_limit))

        d = 0.0 if self.prev_e is None else (e - self.prev_e) / dt
        self.prev_e = e

        u = self.kp * e + self.ki * self.i + self.kd * d
        if self.out_limit is not None:
            u = max(-self.out_limit, min(self.out_limit, u))
        return float(u)


class BaseController:
    def __init__(
        self,
        yor,
        base_max_vel,
        base_max_accel,
        origin: tuple[float, float],
        grid_res: float,
        control_hz: int = 30,
        k_pos: float = 1.5,
        ki_pos: float = 0.01,
        kd_pos: float = 0.15,
        k_theta: float = 2.1,
        ki_theta: float = 0.01,
        kd_theta: float = 0.2,
        pos_tol: float = 0.05,   # 10 cm — comfortably above ZED pose noise (~3–5 cm at range)
        theta_tol: float = 0.03,
    ):
        self.origin = origin
        self.grid_res = grid_res
        self.rate = RateLimiter(control_hz, name="BaseController")

        self.yor = yor
        self.base = Base(max_vel=base_max_vel, max_accel=base_max_accel)
        self.zed_sub = None

        self.pos_tol = pos_tol
        self.theta_tol = theta_tol

        self.vel_alpha = 0.5   # was 0.2 — faster response to direction changes near goal
                               # (0.2 was so sluggish that overshoot → oscillation)

        self._vel_lock = threading.Lock()
        self.last_target_velocity = np.zeros(3, dtype=float)
        self.last_t = time.monotonic()
        self.heading_gate = math.radians(25.0)

        self.vmin = 0.05
        self.vmax = 0.35
        self.omegamin = 0.05
        self.omegamax = 1.0

        i_limit_lin = 0.08
        i_limit_yaw = 0.15
        self.pid_x = PID(k_pos, ki_pos, kd_pos, i_limit=i_limit_lin)
        self.pid_y = PID(k_pos, ki_pos, kd_pos, i_limit=i_limit_lin)
        self.pid_th = PID(k_theta, ki_theta, kd_theta, i_limit=i_limit_yaw)

        self.mode = "BASE_VEL"
        self.target_velocity = np.zeros(3, dtype=float)

        self._path_world = None
        self._goal = None

        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self.heading_gate_on = math.radians(35.0)
        self.heading_gate_off = math.radians(5.0)
        self._rot_only = False
        self._worker = threading.Thread(target=self._run, name="PathUpdater", daemon=True)
        self._worker.start()

        self._nav_lock = threading.Lock()
        self._nav = None

    def zed_sub_init(self, timeout_s: float = 1.0):
        if self.zed_sub is not None:
            return

        done = threading.Event()
        out = {"sub": None, "err": None}

        def _worker():
            try:
                out["sub"] = Subscriber(
                    host=THOR_IP,
                    port=ZED_PUB_PORT,
                    topics=[POSE_TOPIC],
                    buffer=False,
                )
            except Exception as e:
                out["err"] = e
            finally:
                done.set()

        threading.Thread(target=_worker, daemon=True).start()

        if not done.wait(timeout_s):
            return
        if out["sub"] is None:
            return

        self.zed_sub = out["sub"]
        print("Zed Subscriber Initialized")
        return

    def reset_pids(self):
        self.pid_x.reset()
        self.pid_y.reset()
        self.pid_th.reset()
        self.last_target_velocity = np.zeros(3, dtype=float)
        self.last_t = time.monotonic()

    def get_nav_debug(self):
        with self._nav_lock:
            return None if self._nav is None else dict(self._nav)

    def stop(self):
        self._stop_evt.set()
        if self._worker.is_alive():
            self._worker.join(timeout=1.5)

    def _run(self):
        Ld_base: float = 0.32
        Ld_gain: float = 0.4
        Ld_min: float = 0.30   # was 0.20 — higher min prevents chasing a point under the nose
        Ld_max: float = 0.40
        end_dist_tol: float = 0.25  # was 0.08 — switch to MOVE_TO further out for smooth braking
        last_path_uid = None

        v_meas_filt = 0.0
        last_xy = None
        last_pose_t = None
        tau_v = 0.30

        last_path_sig = None
        waypoints = None
        s_cum = None
        total_len = 0.0
        prev_mode = self.mode

        last_sub_try = 0.0

        while not self._stop_evt.is_set():
            if self.mode == "BASE_VEL":
                self.base.set_target_base_velocity(
                    np.asarray(self.target_velocity, dtype=float), smooth=True
                )
                self.rate.sleep()
                continue

            if self.zed_sub is None:
                now = time.monotonic()
                if (now - last_sub_try) > 1.0:
                    last_sub_try = now
                    self.zed_sub_init()

                self.yor.pose = None
                self.base.set_target_base_velocity(np.zeros(3, dtype=float), smooth=True)
                self.rate.sleep()
                continue

            try:
                pose = get_pose(self.zed_sub)
                self.yor.pose = pose
                translation, theta, T_base = pose
                theta = _wrap_pi(theta + math.pi)
                x = float(translation[0])
                y = float(translation[2])
            except Exception:
                self.yor.pose = None
                self.base.set_target_base_velocity(np.zeros(3, dtype=float), smooth=True)
                self.rate.sleep()
                continue

            if self.mode != prev_mode:
                self.reset_pids()
                prev_mode = self.mode

            tx, ty, tth = x, y, None
            dist_goal = None
            stop = False
            to_tgt = np.array([0.0, 0.0], dtype=float)

            match self.mode:
                case "MOVE_TO":
                    goal = self._goal
                    if not goal:
                        stop = True
                    else:
                        tx, ty, tth = goal
                        dist_goal = None
                        tx, ty = float(tx), float(ty)

                        heading_freeze_r = 0.10
                        dist = math.hypot(tx - x, ty - y)
                        dist_goal = dist

                        if tth is None:
                            to_tgt = np.array([tx - x, ty - y], dtype=float)
                            d_theta = heading_error_from_dir(
                                T_base, to_tgt, forward_col=0, flip_sign=True
                            )

                            if dist < self.pos_tol:
                                stop = True
                        else:
                            to_tgt = dir_from_yaw(tth)
                            d_theta = heading_error_from_dir(
                                T_base, to_tgt, forward_col=0, flip_sign=True
                            )

                            if dist < self.pos_tol and abs(d_theta) < self.theta_tol:
                                stop = True

                case "PATH_FOLLOWING":
                    path_world = self._path_world
                    if not path_world:
                        self.base.set_target_base_velocity(np.zeros(3, dtype=float), smooth=True)
                        self.rate.sleep()
                        continue

                    path_uid = (len(path_world), path_world[0], path_world[-1])
                    if path_uid != last_path_uid:
                        d_start = math.hypot(
                            float(path_world[0][0]) - x, float(path_world[0][1]) - y
                        )
                        d_end = math.hypot(
                            float(path_world[-1][0]) - x, float(path_world[-1][1]) - y
                        )
                        last_path_uid = (len(path_world), path_world[0], path_world[-1])

                    path_sig = (len(path_world), path_world[0], path_world[-1])
                    if path_sig != last_path_sig or waypoints is None:
                        waypoints = [
                            np.array([float(px), float(pz)], dtype=float)
                            for (px, pz) in path_world
                        ]
                        s_cum, total_len = _cumlen(waypoints)
                        last_path_sig = path_sig

                    s_closest, _, _, _ = _closest_point_on_polyline(x, y, waypoints, s_cum)

                    t_now = time.monotonic()
                    if last_xy is None:
                        last_xy, last_pose_t = (x, y), t_now

                    dt_pose = max(1e-3, t_now - last_pose_t)
                    dist_xy = math.hypot(x - last_xy[0], y - last_xy[1])
                    v_inst = dist_xy / dt_pose
                    alpha_v = dt_pose / (tau_v + dt_pose)
                    v_meas_filt = (1 - alpha_v) * v_meas_filt + alpha_v * v_inst
                    last_xy, last_pose_t = (x, y), t_now

                    Ld = float(np.clip(Ld_base + Ld_gain * v_meas_filt, Ld_min, Ld_max))
                    s_tgt = s_closest + Ld

                    if (total_len - s_closest) < end_dist_tol:
                        last_pt = path_world[-1]
                        self._goal = (float(last_pt[0]), float(last_pt[1]), None)
                        self.mode = "MOVE_TO"
                        self.rate.sleep()
                        continue

                    p_tgt, tan = _point_at_s(waypoints, s_cum, s_tgt)
                    tx, ty = float(p_tgt[0]), float(p_tgt[1])

                    tan = np.asarray(tan, dtype=float)
                    to_tgt = np.array([tx - x, ty - y], dtype=float)
                    if float(tan @ to_tgt) < 0.0:
                        tan = -tan

                    tth = math.atan2((tx - x), (ty - y))

                case _:
                    print("Nav mode set is not in [BASE_VEL, PATH_FOLLOWING, MOVE_TO]")
                    self.rate.sleep()
                    continue

            if stop:
                self.mode = "BASE_VEL"
                self.target_velocity = np.zeros(3, dtype=float)
                self.base.set_target_base_velocity(self.target_velocity, smooth=True)
                self.rate.sleep()
                continue

            path_copy = None
            if self._path_world:
                path_copy = [(float(px), float(py)) for (px, py) in self._path_world]

            debug_info = {
                "mode": str(self.mode),
                "path_world": path_copy,
                "lookahead_xz": (float(tx), float(ty)),
                "pose_xz": (float(x), float(y)),
                "yaw": float(theta),
                "yaw_des": (None if tth is None else float(tth)),
                "rot_only": bool(self._rot_only),
            }
            with self._nav_lock:
                self._nav = debug_info

            d_theta = heading_error_from_dir(
                T_base,
                to_tgt,
                forward_col=0,
                flip_sign=True,
            )

            if self._rot_only:
                if abs(d_theta) < self.heading_gate_off:
                    self._rot_only = False
            else:
                if abs(d_theta) > self.heading_gate_on:
                    self._rot_only = True

            rotation_only = self._rot_only

            dx = float(tx - x)
            dz = float(ty - y)

            R_bw = T_base[:3, :3]
            d_world = np.array([dx, 0.0, dz], dtype=float)
            d_body = R_bw.T @ d_world

            d_fwd = float(d_body[2])
            d_left = float(d_body[0])

            now = time.monotonic()
            dt = max(1e-3, min(0.25, now - self.last_t))
            self.last_t = now

            if rotation_only:
                self.pid_x.reset()
                self.pid_y.reset()
                vx, vy = 0.0, 0.0
            else:
                vx = self.pid_x.step(d_fwd, dt)
                vy = self.pid_y.step(d_left, dt)

            omega = self.pid_th.step(d_theta, dt)
            def _soft_clip(v, v_min, v_max):
                a_v = abs(v)
                if a_v < 1e-4:
                    return 0.0
                if a_v < v_min:
                    return v # Don't boost to v_min if it's already small
                return np.sign(v) * float(np.clip(a_v, v_min, v_max))

            vx = _soft_clip(vx, self.vmin, self.vmax)
            vy = _soft_clip(vy, self.vmin, self.vmax)
            omega = _soft_clip(omega, self.omegamin, self.omegamax)

            if rotation_only:
                self.target_velocity = np.array([0.0, 0.0, omega], dtype=float)
                self.last_target_velocity = self.target_velocity
            else:
                new_cmd = np.array([vy, vx, omega], dtype=float)
                self.target_velocity = (
                    self.vel_alpha * new_cmd
                    + (1.0 - self.vel_alpha) * self.last_target_velocity
                )
                self.last_target_velocity = self.target_velocity

            self.base.set_target_base_velocity(self.target_velocity, smooth=True)
            self.rate.sleep()


if __name__ == "__main__":
    # Intentionally minimal: the controller depends on a yor object from your runtime.
    pass
