import math
import re
import threading
import time
from enum import IntEnum
from queue import Queue
from typing import Any, Optional, Tuple, Union

import numpy as np
import queue  # for queue.Empty / queue.Full

from loop_rate_limiters import RateLimiter

# --- New motor API ---
from sparkcan_py import SparkFlex

# Optional enums if your binding exposes them (safe to ignore if not present)
try:
    from sparkcan_py import CtrlType, IdleMode, MotorType, SensorType  # noqa: F401
except Exception:
    IdleMode = CtrlType = MotorType = SensorType = None  # type: ignore

# Pico lift (serial)
try:
    import serial  # type: ignore
except Exception:
    serial = None  # type: ignore


# ----------------------------
# Constants / config
# ----------------------------
drivetrain_can = "can0"

POLICY_CONTROL_FREQ = 10
POLICY_CONTROL_PERIOD_NS = int(1e9 / POLICY_CONTROL_FREQ)

CONTROL_FREQ = 250  # Hz control loop
CONTROL_PERIOD = 1.0 / CONTROL_FREQ

NUM_SWERVES = 4
LENGTH = 0.1225  # m
WIDTH = 0.170  # m
TIRE_RADIUS = 0.0381  # m

MODULE_ORDER = ("FL", "FR", "RR", "RL")

CAN_IDS_DRIVE = (1, 4, 3, 2)  # [FL, FR, RR, RL]
CAN_IDS_ROT = (5, 8, 7, 6)  # [FL, FR, RR, RL]

ROTATION_OFFSETS = np.array([0.75, 0.00, 0.25, 0.50], dtype=float)

ROT_DIAG_SWAP_PERM = np.array([1, 0, 3, 2], dtype=int)
TRANS_OPPOSITE_MASK = np.array([False, False, False, False], dtype=bool)

TWO_PI = 2.0 * math.pi

USE_FEEDBACK_FOR_STEER = False
DRIVE_VEL_SCALE = 2.0


# ----------------------------
# Math helpers
# ----------------------------
def wrap_pi(a: np.ndarray) -> np.ndarray:
    return ((a + math.pi) % (2 * math.pi)) - math.pi


def diff_angle(a: np.ndarray, b: Union[np.ndarray, float]) -> np.ndarray:
    return ((a - b) + math.pi) % (2 * math.pi) - math.pi


def frac_to_rad(f: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return ((np.array(f) + 0.5) % 1.0 - 0.5) * TWO_PI


def rad_to_frac(rad: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return (np.array(rad) / TWO_PI) % 1.0


# ----------------------------
# Pico lift (serial)
# ----------------------------
class PicoLift:
    _HEIGHT_PATTERN = re.compile(r"Height:\s*(-?[\d.]+)\s*mm")

    def __init__(
        self,
        device_path: str = "/dev/ttyACM0",
        baud: int = 115200,
        timeout: float = 0.2,
    ):
        self.device_path = device_path
        self.baud = baud
        self.timeout = timeout

        self._ser = None
        self._lock = threading.Lock()
        self._last_cmd = None
        self._last_send_ts = 0.0
        self._min_repeat_interval = 0.05

        self._drain_thread = None
        self._drain_stop = threading.Event()

        self._height_m: Optional[float] = None
        self._height_lock = threading.Lock()

        self._ensure_open()
        self._ensure_drain()

    def _ensure_open(self) -> None:
        if serial is None:
            print("[PicoLift] pyserial not installed; lift disabled")
            return
        if self._ser is not None and getattr(self._ser, "is_open", False):
            return
        try:
            print(f"[PicoLift] Opening serial {self.device_path} @ {self.baud}...")
            self._ser = serial.Serial(
                self.device_path,
                self.baud,
                timeout=self.timeout,
                write_timeout=0.02,
                rtscts=False,
                dsrdtr=False,
                xonxoff=False,
            )
            try:
                self._ser.reset_input_buffer()
                self._ser.reset_output_buffer()
            except Exception:
                pass
            print("[PicoLift] Serial open OK")
        except Exception as e:
            print(f"[PicoLift] Failed to open {self.device_path}: {e}")
            self._ser = None

    def _ensure_drain(self) -> None:
        if serial is None:
            return
        if self._drain_thread is not None and self._drain_thread.is_alive():
            return
        self._drain_stop.clear()
        self._drain_thread = threading.Thread(target=self._drain_loop, daemon=True)
        self._drain_thread.start()

    def _drain_loop(self) -> None:
        while not self._drain_stop.is_set():
            try:
                if self._ser is None or not getattr(self._ser, "is_open", False):
                    self._ensure_open()
                    time.sleep(0.1)
                    continue

                data = self._ser.readline()
                if data:
                    try:
                        line = data.decode("utf-8").strip()
                        print(f"[LIFT] {line}")
                        match = self._HEIGHT_PATTERN.search(line)
                        if match:
                            height_mm = float(match.group(1))
                            with self._height_lock:
                                self._height_m = height_mm / 1000.0
                    except Exception:
                        print(f"[LIFT] Raw: {data}")

            except Exception:
                try:
                    if self._ser:
                        self._ser.close()
                except Exception:
                    pass
                self._ser = None
                time.sleep(0.1)

    def _send(self, cmd: str) -> None:
        if serial is None:
            return
        now = time.monotonic()
        with self._lock:
            self._ensure_open()
            if self._ser is None:
                print(f"[PicoLift] Not connected; drop cmd '{cmd}'")
                return

            if (
                cmd != "stop"
                and cmd == self._last_cmd
                and (now - self._last_send_ts) < self._min_repeat_interval
            ):
                return

            try:
                payload = (cmd + "\n").encode()
                if cmd == "stop":
                    try:
                        self._ser.write(b"\r\n")
                    except Exception:
                        pass
                    payload = (cmd + "\r\n").encode()

                self._ser.write(payload)

                if cmd == "stop":
                    try:
                        self._ser.flush()
                    except Exception:
                        pass

                self._last_cmd = cmd
                self._last_send_ts = now
                print(f"[PicoLift] sent '{cmd}'")

            except Exception as e:
                print(f"[PicoLift] write error: {e}")
                try:
                    self._ser.close()
                except Exception:
                    pass
                self._ser = None

    def up(self) -> None:
        self._send("up")

    def down(self) -> None:
        self._send("down")

    def home(self) -> None:
        self._send("home")

    def stop(self) -> None:
        self._last_cmd = None
        self._send("stop")

    def get_height(self) -> Optional[float]:
        with self._height_lock:
            return self._height_m


# ----------------------------
# Commands
# ----------------------------
class CommandType(IntEnum):
    BASE_VELOCITY = 1
    BASE_POSITION = 2
    LIFT_POSITION = 3


# ----------------------------
# SparkFlex wrappers
# ----------------------------
class RotationMotor:
    """Rotation motor driven by SparkFlex position setpoint (fraction 0..1)."""

    def __init__(self, can_if: str, can_id: int, offset_frac: float = 0.0):
        self.dev = SparkFlex(can_if, can_id)
        self.offset = float(offset_frac)
        self.last_cmd_frac = 0.0

        try:
            if IdleMode and hasattr(self.dev, "SetIdleMode"):
                self.dev.SetIdleMode(IdleMode.kCoast)
        except Exception:
            pass

    def heartbeat(self) -> None:
        self.dev.Heartbeat()

    def set_position_fraction(self, frac_0_1: float) -> None:
        f = (frac_0_1 + self.offset) % 1.0
        self.last_cmd_frac = f
        self.dev.SetPosition(float(f))

    def get_position_deg(self) -> float:
        try:
            return float(self.dev.GetAbsoluteEncoderPosition())
        except Exception:
            return float("nan")

    def get_position_rad(self) -> float:
        if not USE_FEEDBACK_FOR_STEER:
            frac_no_off = (self.last_cmd_frac - self.offset) % 1.0
            return float(frac_to_rad(frac_no_off))

        try:
            deg = float(self.dev.GetAbsoluteEncoderPosition())
            frac = (deg / 360.0) % 1.0
            frac_no_off = (frac - self.offset) % 1.0
            return float(frac_to_rad(frac_no_off))
        except Exception:
            return float(frac_to_rad(self.last_cmd_frac))

    def get_position_counts(self) -> float:
        try:
            return float(self.dev.GetPosition())
        except Exception:
            return float("nan")


class DriveMotor:
    """Drive motor driven by SparkFlex velocity setpoint (PID on controller)."""

    def __init__(self, can_if: str, can_id: int):
        self.dev = SparkFlex(can_if, can_id)

        if IdleMode and hasattr(self.dev, "SetIdleMode"):
            try:
                self.dev.SetIdleMode(IdleMode.kCoast)
            except Exception:
                pass
                
        if CtrlType and hasattr(self.dev, "SetCtrlType"):
            try:
                self.dev.SetCtrlType(CtrlType.kVelocity)
            except Exception:
                pass
                
        # Request Status Frame 2 (velocity + position) at 20 ms
        if hasattr(self.dev, "SetPeriodicStatus2Period"):
            try:
                self.dev.SetPeriodicStatus2Period(20)
            except Exception:
                pass

    def heartbeat(self) -> None:
        self.dev.Heartbeat()

    def set_velocity_mps(self, v_mps: float) -> None:
        self.dev.SetVelocity(float(v_mps * DRIVE_VEL_SCALE))

    def get_velocity_raw(self) -> float:
        try:
            return float(self.dev.GetVelocity())
        except Exception:
            return float("nan")

    def get_position_counts(self) -> float:
        try:
            return float(self.dev.GetPosition())
        except Exception:
            return float("nan")


# ----------------------------
# Base (swerve control)
# ----------------------------
class Base:
    def __init__(
        self,
        max_vel=np.array((1.0, 1.0, 1.57)),
        max_accel=np.array((1.0, 1.0, 1.57)),
    ):
        self.max_vel = max_vel
        self.max_accel = max_accel

        self.C = np.array(
            [
                [1, 0, WIDTH],
                [1, 0, -WIDTH],
                [1, 0, -WIDTH],
                [1, 0, WIDTH],
                [0, 1, LENGTH],
                [0, 1, LENGTH],
                [0, 1, -LENGTH],
                [0, 1, -LENGTH],
            ]
        )

        self.rotation_motors = [
            RotationMotor(drivetrain_can, CAN_IDS_ROT[i], ROTATION_OFFSETS[i])
            for i in range(NUM_SWERVES)
        ]
        self.drive_motors = [
            DriveMotor(drivetrain_can, CAN_IDS_DRIVE[i]) for i in range(NUM_SWERVES)
        ]

        self._pico_lift = PicoLift()

        self.steer_pos = np.zeros(NUM_SWERVES)
        self.drive_vel = np.zeros(NUM_SWERVES)
        self.x = np.zeros(3)
        self.dx = np.zeros(3)

        self._command_queue: Queue[dict[str, Any]] = Queue(3)
        self.base_target = np.zeros(3)

        self.control_loop_thread: Optional[threading.Thread] = threading.Thread(
            target=self.control_loop, daemon=True
        )
        self.control_loop_running = False

        self._last_loop_time = time.monotonic()

        # --- S-curve profiling state (kept; now optional per-command) ---
        self._smooth_active = False  # whether to apply smoothing for the *current* command
        self._v_prof = np.zeros(3, dtype=float)
        self._seg_v0 = np.zeros(3, dtype=float)
        self._seg_v1 = np.zeros(3, dtype=float)
        self._seg_t = 0.0
        self._seg_T = 0.0

        self._a_max = np.array([1.9, 1.9, 6.5], dtype=float)
        self._T_min = 0.01
        self._retarget_eps = 1e-3

    # --- Lift controls ---
    def lift_up(self) -> None:
        if self._pico_lift:
            self._pico_lift.up()

    def lift_down(self) -> None:
        if self._pico_lift:
            self._pico_lift.down()

    def lift_home(self) -> None:
        if self._pico_lift:
            self._pico_lift.home()

    def lift_stop(self) -> None:
        if self._pico_lift:
            self._pico_lift.stop()

    def get_lift_height(self) -> Optional[float]:
        if self._pico_lift:
            return self._pico_lift.get_height()
        return None
    
    def lift_delta_height(
        self,
        delta_m: float,
        tolerance_m: float = 0.002,
        timeout_s: float = 30.0,
        min_height_m: float = 0.0,
        max_height_m: float = 0.5,
    ) -> bool:
        """
        Move lift up/down by delta in meters (positive=up, negative=down).
        Returns True if target reached within tolerance, False otherwise.
        """
        if self._pico_lift is None:
            return False

        current_height = self._pico_lift.get_height()
        if current_height is None:
            print("[lift_delta_height] Lift height unknown; cannot move by delta")
            return False

        return self.lift_to_height(
            target_m=current_height + float(delta_m),
            tolerance_m=tolerance_m,
            timeout_s=timeout_s,
            min_height_m=min_height_m,
            max_height_m=max_height_m,
        )


    def lift_to_height(
        self,
        target_m: float,
        tolerance_m: float = 0.002,
        timeout_s: float = 30.0,
        min_height_m: float = 0.0,
        max_height_m: float = 0.5,
    ) -> bool:
        """
        Move lift to an absolute height position (blocking).
        Returns True if target reached within tolerance, False on timeout/stall/unknown height.

        Safety features:
        - clamps target within [min_height_m, max_height_m]
        - timeout
        - overshoot stop
        - stall detection
        """
        if self._pico_lift is None:
            return False

        # Clamp target
        target_m = float(max(min_height_m, min(max_height_m, float(target_m))))

        current_height = self._pico_lift.get_height()
        if current_height is None:
            print("[lift_to_height] Lift height unknown; cannot move to target")
            return False

        error = target_m - float(current_height)
        if abs(error) <= tolerance_m:
            return True

        moving_up = error > 0.0
        (self.lift_up if moving_up else self.lift_down)()

        rate = RateLimiter(60)
        start_time = time.monotonic()

        last_height = float(current_height)
        stall_start: Optional[float] = None

        try:
            while True:
                now = time.monotonic()
                if (now - start_time) > timeout_s:
                    print(f"[lift_to_height] Timeout after {timeout_s:.1f}s")
                    self.lift_stop()
                    return False

                height = self._pico_lift.get_height()
                if height is None:
                    # If we lost telemetry, safest is stop and fail
                    print("[lift_to_height] Lost height telemetry; stopping")
                    self.lift_stop()
                    return False

                height = float(height)
                error = target_m - height

                # Within tolerance
                if abs(error) <= tolerance_m:
                    self.lift_stop()
                    return True

                # Overshoot detection
                if (moving_up and height > target_m) or ((not moving_up) and height < target_m):
                    self.lift_stop()
                    return True

                # Stall detection (no meaningful movement for >1s)
                if abs(height - last_height) < 0.0005:
                    if stall_start is None:
                        stall_start = now
                    elif (now - stall_start) > 1.0:
                        print(f"[lift_to_height] Stall detected at {height:.4f} m")
                        self.lift_stop()
                        return False
                else:
                    stall_start = None
                    last_height = height

                rate.sleep()

        except KeyboardInterrupt:
            self.lift_stop()
            return False
        except Exception as e:
            print(f"[lift_to_height] Error: {e}")
            self.lift_stop()
            return False


    # def lift_delta_height(self, delta_m: float) -> None:
    #     if self._pico_lift is None:
    #         return
    #     current_height = self._pico_lift.get_height()
    #     if current_height is None:
    #         print("Lift height unknown; cannot move by delta")
    #         return

    #     target_height = current_height + delta_m
    #     if target_height <= 0.0:
    #         target_height = 0.0
    #         if delta_m < 0.0:
    #             return

    #     if target_height > current_height:
    #         self.lift_up()
    #     else:
    #         self.lift_down()

    #     rate = RateLimiter(60)
    #     while True:
    #         height = self._pico_lift.get_height()
    #         if height is None:
    #             break
    #         if (delta_m > 0 and height >= target_height) or (
    #             delta_m < 0 and height <= target_height
    #         ):
    #             break
    #         rate.sleep()

    #     self.lift_stop()

    # --- Public API ---
    def start_control(self):
        if self.control_loop_thread is None:
            print("To initiate a new control loop, create a new Base() instance first")
            return
        self.control_loop_running = True
        self.control_loop_thread.start()

    def stop_control(self):
        if self.control_loop_thread is None:
            print("Control loop not running")
            return
        self.control_loop_running = False
        self.control_loop_thread.join()
        self.control_loop_thread = None

    def set_target_base_velocity(self, target: np.ndarray, smooth: bool = False):
        """target: np.array([vx, vy, omega]) in vehicle frame (m/s, m/s, rad/s)"""
        self._enqueue_command(
            {
                "type": CommandType.BASE_VELOCITY,
                "target": np.array(target, dtype=float),
                "smooth": bool(smooth),
            }
        )

    # ---------------- control loop ----------------
    def control_loop(self):
        rate_limiter = RateLimiter(CONTROL_FREQ, name="base-controller")
        disable_motors = True
        last_command_time_ns = time.perf_counter_ns()

        while self.control_loop_running:
            cmd = None
            try:
                while True:
                    cmd = self._command_queue.get_nowait()
            except queue.Empty:
                pass

            if cmd is not None:
                self.base_target = np.array(cmd["target"], dtype=float)
                self._smooth_active = bool(cmd.get("smooth", False))
                last_command_time_ns = time.perf_counter_ns()
                if cmd["type"] == CommandType.BASE_VELOCITY:
                    disable_motors = False

            if (time.perf_counter_ns() - last_command_time_ns) > 2.5 * POLICY_CONTROL_PERIOD_NS:
                disable_motors = True

            for m in self.drive_motors:
                m.heartbeat()
            for m in self.rotation_motors:
                m.heartbeat()

            self._update_state()

            if disable_motors:
                for d in self.drive_motors:
                    d.set_velocity_mps(0.0)

            else:
                dt = 1.0 / CONTROL_FREQ
                v_cmd = self.base_target

                if self._smooth_active:
                    if np.linalg.norm(v_cmd - self._seg_v1) > self._retarget_eps:
                        self._start_scurve_segment(v_cmd)
                    v_used = self._update_scurve(dt)
                else:
                    # Keep profiling state consistent so enabling smoothing later doesn't jump from stale state
                    self._v_prof = v_cmd.copy()
                    self._seg_v0 = v_cmd.copy()
                    self._seg_v1 = v_cmd.copy()
                    self._seg_t = 0.0
                    self._seg_T = 0.0
                    v_used = v_cmd

                wheel_speeds, wheel_angles = self._vehicle_velocity_to_angle_and_speed(
                    v_used, cos_error_scaling=True
                )

                target_fracs = rad_to_frac(wheel_angles)
                for i, rm in enumerate(self.rotation_motors):
                    rm.set_position_fraction(float(target_fracs[i]))

                for i, dm in enumerate(self.drive_motors):
                    dm.set_velocity_mps(float(wheel_speeds[i]))

            rate_limiter.sleep()

    # -------------- helpers --------------
    def _update_state(self) -> None:
        now = time.monotonic()
        _dt = now - self._last_loop_time
        self._last_loop_time = now

        for i, rm in enumerate(self.rotation_motors):
            self.steer_pos[i] = rm.get_position_rad()

        for i, dm in enumerate(self.drive_motors):
            self.drive_vel[i] = dm.get_velocity_raw()

    def _angle_and_speed_to_vehicle_velocity(
        self, wheel_speeds: np.ndarray, wheel_angles: np.ndarray
    ) -> np.ndarray:
        vx, vy = wheel_speeds * np.cos(wheel_angles), wheel_speeds * np.sin(wheel_angles)
        return np.linalg.lstsq(self.C, np.concatenate((vx, vy)), rcond=None)[0]

    def _start_scurve_segment(self, v_target: np.ndarray):
        v_target = np.asarray(v_target, dtype=float)

        if getattr(self, "_seg_T", 0.0) > 0 and np.allclose(v_target, self._seg_v1, atol=1e-3):
            return

        dv = v_target - self._v_prof
        abs_dv = np.abs(dv)

        if np.all(abs_dv < 1e-3):
            return

        T_needed = np.max((abs_dv * np.pi) / (2.0 * np.maximum(self._a_max, 1e-6)))
        T = max(self._T_min, float(T_needed))

        self._seg_v0 = self._v_prof.copy()
        self._seg_v1 = v_target.copy()
        self._seg_t = 0.0
        self._seg_T = T

    def _update_scurve(self, dt: float) -> np.ndarray:
        if self._seg_T <= 1e-9:
            return self._v_prof

        self._seg_t = min(self._seg_t + dt, self._seg_T)
        tau = self._seg_t / self._seg_T
        s = 0.5 * (1.0 - np.cos(np.pi * tau))
        self._v_prof = self._seg_v0 + (self._seg_v1 - self._seg_v0) * s
        return self._v_prof

    def _vehicle_velocity_to_angle_and_speed(
        self, u_3dof: np.ndarray, cos_error_scaling: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        vx, vy, omega = float(u_3dof[0]), float(u_3dof[1]), float(u_3dof[2])

        vx_t = np.array([vx, vx, vx, vx], dtype=float)
        vy_t = np.array([vy, vy, vy, vy], dtype=float)
        sign = np.where(TRANS_OPPOSITE_MASK, -1.0, 1.0)
        vx_t *= sign
        vy_t *= sign

        vx_r = np.array(
            [+WIDTH * omega, -WIDTH * omega, -WIDTH * omega, +WIDTH * omega], dtype=float
        )
        vy_r = np.array(
            [+LENGTH * omega, +LENGTH * omega, -LENGTH * omega, -LENGTH * omega],
            dtype=float,
        )
        vx_r = vx_r[ROT_DIAG_SWAP_PERM]
        vy_r = vy_r[ROT_DIAG_SWAP_PERM]

        vx_w = vx_t + vx_r
        vy_w = vy_t + vy_r

        wheel_speeds = np.hypot(vx_w, vy_w)
        wheel_angles = np.arctan2(vy_w, vx_w)

        error = diff_angle(wheel_angles, self.steer_pos)
        wheel_angles = np.where(
            np.abs(error) > np.pi / 2, diff_angle(wheel_angles, np.pi), wheel_angles
        )
        wheel_speeds = np.where(np.abs(error) > np.pi / 2, -wheel_speeds, wheel_speeds)

        if cos_error_scaling:
            wheel_speeds *= np.cos(diff_angle(wheel_angles, self.steer_pos))

        return wheel_speeds, wheel_angles

    def _map_steer_angles(self, wheel_angles: np.ndarray) -> np.ndarray:
        ang = wheel_angles.copy()
        ang[TRANS_OPPOSITE_MASK] = ang[TRANS_OPPOSITE_MASK] + math.pi
        ang = ang[ROT_DIAG_SWAP_PERM]
        return wrap_pi(ang)

    def _enqueue_command(self, cmd: dict) -> None:
        if self._command_queue is None:
            return

        try:
            self._command_queue.put_nowait(cmd)
            return
        except queue.Full:
            pass

        try:
            while True:
                _ = self._command_queue.get_nowait()
        except queue.Empty:
            pass

        try:
            self._command_queue.put_nowait(cmd)
        except queue.Full:
            pass


# ---------------- Example usage ----------------
if __name__ == "__main__":
    base = Base()
    base.start_control()
    rate = RateLimiter(50)
    t0 = time.time()
    try:
        while time.time() - t0 < 5.0:
            # Without smoothing (default)
            # base.set_target_base_velocity(np.array([0.0, 0.0, 0.5]))

            # With smoothing (per-call)
            base.set_target_base_velocity(np.array([0.0, 0.0, 0.5]), smooth=True)
            rate.sleep()
    except KeyboardInterrupt:
        pass
    finally:
        base.stop_control()
