"""
whole_body_control.py — YOR Whole-Body Control

Given a world-frame end-effector target (x, y, z, yaw, pitch, roll), this script:

  1. Lift height    — raises/lowers the lift so the arm Z-workspace covers the target
  2. Base nav       — drives (x, y, heading) so the arm XY-workspace covers the target
  3. Arm tracking   — continuously re-commands the arm EE via ZED frames so the arm
                      compensates for base movement (chickenhead behavior)

Usage:
    python whole_body_control.py --side left --x 1.0 --y 0.0 --z 0.9
    python whole_body_control.py --side left --x 1.0 --y 0.0 --z 0.9 --yaw 30

Physical parameters (edit these to match the real robot):
--------------------------------------------------------------------
"""
import sys
import time
import math
import argparse
from pathlib import Path

import numpy as np
import mink

sys.path.append(str(Path(__file__).parent.parent))
from robot.yor import YOR

from commlink import Subscriber
from loop_rate_limiters import RateLimiter

# ── Editable physical parameters ──────────────────────────────────────────────
# Source: nero-welded-base-and-lift.mjcf + left/right_arm_base_link pos

FLOOR_TO_BASE_Z      = 0.1048   # m: chassis Z above floor (base body pos z in MJCF)
ARM_Z_ON_LIFT        = 0.0575   # m: arm base link pos z above lift top (Lift_Top child pos z)
LIFT_MIN_HEIGHT      = 0.0      # m: lift fully down (home / hard stop)
LIFT_MAX_HEIGHT      = 0.416    # m: lift fully up (2× slider joint range 0–0.208)

# Arm mount lateral offsets in the robot body frame (lift body Y-axis)
LEFT_ARM_Y_OFFSET    = -0.125   # m: from MJCF left_arm_base_link pos y
RIGHT_ARM_Y_OFFSET   =  0.125   # m: from MJCF right_arm_base_link pos y
ARM_X_OFFSET         = -0.005   # m: both arms, from MJCF left/right_arm_base_link pos x

# Navigation parameters
BASE_GOAL_TOL        = 0.05     # m: base xy convergence tolerance before IK
BASE_HEADING_TOL     = 0.05     # rad: heading convergence tolerance
BASE_TIMEOUT         = 30.0     # s: give up waiting for base after this

# Lift parameters
LIFT_HOME_DURATION   = 8.0      # s: drive lift down blindly to home (no encoder)
LIFT_GOAL_TOL        = 0.01     # m: acceptable lift height error (via get_lift_height)
LIFT_TIMEOUT         = 15.0     # s: give up waiting for lift after this

# Arm tracking parameters
ARM_TRACKING_HZ      = 30.0     # Hz: continuous arm re-command rate

# ZED subscriber defaults
ZED_HOST             = "10.21.16.110"
ZED_PORT             = 6000
POSE_TOPIC           = "zed/pose"

# ──────────────────────────────────────────────────────────────────────────────


# ── ZED utilities ──────────────────────────────────────────────────────────────

def make_zed_subscriber(host: str, port: int) -> Subscriber:
    """Create a ZED pose subscriber, handling Subscriber API variations."""
    from inspect import signature
    try:
        params = set(signature(Subscriber.__init__).parameters.keys())
        if "buffer" in params:
            return Subscriber(host=host, port=port, topics=[POSE_TOPIC], buffer=False)
        if "keep_old" in params:
            return Subscriber(host=host, port=port, topics=[POSE_TOPIC], keep_old=False)
    except Exception:
        pass
    return Subscriber(host=host, port=port, topics=[POSE_TOPIC])


def zed_pose_as_se3(sub: Subscriber) -> mink.SE3:
    """Parse ZED pose payload [qx, qy, qz, qw, tx, ty, tz] → mink.SE3 (wxyz + xyz)."""
    arr = np.asarray(sub[POSE_TOPIC], dtype=np.float64).reshape(-1)
    if arr.size < 7:
        raise RuntimeError(f"ZED pose payload too short: {arr.size} < 7")
    q_wxyz = arr[[3, 0, 1, 2]]   # xyzw → wxyz
    t_xyz  = arr[4:7]
    return mink.SE3(np.concatenate([q_wxyz, t_xyz]))


# ── Geometry helpers ───────────────────────────────────────────────────────────

def euler_to_wxyz(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """ZYX Euler (rad) → quaternion [w, x, y, z]."""
    cy, sy = math.cos(yaw / 2),   math.sin(yaw / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cr, sr = math.cos(roll / 2),  math.sin(roll / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def world_target_to_arm_frame(
    base_se3: mink.SE3,
    lift_height: float,
    side: str,
    x_w: float,
    y_w: float,
    z_w: float,
    yaw: float,
    pitch: float,
    roll: float,
) -> mink.SE3:
    """
    Convert a world-frame EE target into the arm's local IK frame.

    Chain:  world → robot_body (ZED) → lift_top → arm_base → EE target

    Args:
        base_se3:    Current robot world pose from ZED (mink.SE3).
        lift_height: Current lift height in metres.
        side:        'left' or 'right'.
        x_w/y_w/z_w: Target world position.
        yaw/pitch/roll: Target world orientation (ZYX Euler, radians).
    """
    # World-frame target SE3
    wxyz = euler_to_wxyz(yaw, pitch, roll)
    target_world = mink.SE3.from_rotation_and_translation(
        mink.SO3(wxyz), np.array([x_w, y_w, z_w])
    )

    # Robot base position in world: ZED gives world_T_base directly
    # We then offset to the arm base in that frame
    arm_y = LEFT_ARM_Y_OFFSET if side == "left" else RIGHT_ARM_Y_OFFSET
    arm_offset_body = np.array([ARM_X_OFFSET, arm_y, ARM_Z_ON_LIFT + lift_height])

    T_world_armbase = mink.SE3.from_rotation_and_translation(
        base_se3.rotation(),
        base_se3.translation() + base_se3.rotation().apply(arm_offset_body),
    )

    # Target in arm frame
    return T_world_armbase.inverse() @ target_world


# ── Lift / base helpers ────────────────────────────────────────────────────────

def home_lift(yor: YOR) -> None:
    """Drive lift to physical bottom stop to establish height = 0."""
    print(f"[WBC] Lift homing: driving down for {LIFT_HOME_DURATION:.1f}s...")
    yor.lift_down()
    time.sleep(LIFT_HOME_DURATION)
    yor.lift_stop()
    print("[WBC] Lift homed to bottom (height = 0.0 m)")


def wait_for_zed(yor: YOR, timeout: float = 10.0) -> bool:
    """Block until yor.pose is populated."""
    print("[WBC] Waiting for ZED pose via yor...")
    t0 = time.time()
    while time.time() - t0 < timeout:
        if yor.pose is not None:
            print("[WBC] ZED pose received.")
            return True
        time.sleep(0.1)
    print("[WBC] WARNING: ZED pose not received within timeout.")
    return False


def get_robot_xy_theta(yor: YOR):
    """Extract (x, y, theta) from yor.pose, or None."""
    if yor.pose is None:
        return None
    (xyz, theta, _T) = yor.pose
    return float(xyz[0]), float(xyz[2]), float(theta)


def command_base_to(yor: YOR, goal_x: float, goal_y: float, goal_theta: float) -> bool:
    """Command base to goal and wait for convergence. Returns True if converged."""
    print(f"[WBC] Commanding base to ({goal_x:.3f}, {goal_y:.3f}, "
          f"θ={math.degrees(goal_theta):.1f}°)")
    yor.move_to((goal_x, goal_y, goal_theta))

    t0 = time.time()
    while time.time() - t0 < BASE_TIMEOUT:
        pose = get_robot_xy_theta(yor)
        if pose is None:
            time.sleep(0.2)
            continue
        rx, ry, rtheta = pose
        dist   = math.hypot(goal_x - rx, goal_y - ry)
        dtheta = abs(goal_theta - rtheta)
        if dist < BASE_GOAL_TOL and dtheta < BASE_HEADING_TOL:
            print(f"[WBC]   Base converged (dist={dist:.3f}m)")
            return True
        time.sleep(0.1)

    print(f"[WBC] WARNING: Base did not converge within {BASE_TIMEOUT:.0f}s")
    return False


def command_lift_to(yor: YOR, target_h: float) -> bool:
    """Adjust lift to target_h. Returns True if converged or unavailable."""
    target_h = float(np.clip(target_h, LIFT_MIN_HEIGHT, LIFT_MAX_HEIGHT))
    print(f"[WBC] Commanding lift to {target_h:.3f}m...")

    if hasattr(yor, 'lift_to_height') and callable(yor.lift_to_height):
        ok = yor.lift_to_height(
            target_h,
            tolerance_m=LIFT_GOAL_TOL,
            timeout_s=LIFT_TIMEOUT,
            min_height_m=LIFT_MIN_HEIGHT,
            max_height_m=LIFT_MAX_HEIGHT,
        )
        if ok:
            print(f"[WBC]   Lift reached {target_h:.3f}m")
        else:
            print(f"[WBC] WARNING: lift_to_height did not converge")
        return ok

    print("[WBC]   lift_to_height() not available — skipping.")
    return True


# ── Main reach + hold function ─────────────────────────────────────────────────

def reach_and_hold(
    yor: YOR,
    zed_sub: Subscriber,
    side: str,
    x_w: float,
    y_w: float,
    z_w: float,
    yaw: float = 0.0,
    pitch: float = 0.0,
    roll: float = 0.0,
    skip_base: bool = False,
    skip_lift: bool = False,
    hz: float = ARM_TRACKING_HZ,
    gripper_target: float = 1.0,
) -> None:
    """
    Move arm to world-frame target, then hold it continuously via ZED tracking.

    Stage 1 (one-shot): lift + base navigation.
    Stage 2 (continuous loop): re-command arm IK on every ZED frame so the arm
    compensates for any base movement (chickenhead behavior).

    Args:
        yor:             Initialized YOR instance.
        zed_sub:         Live ZED Subscriber.
        side:            'left' or 'right'.
        x_w/y_w/z_w:    Target world position (m).
        yaw/pitch/roll:  Target world orientation (rad, ZYX).
        skip_base:       Skip base navigation.
        skip_lift:       Skip lift adjustment.
        hz:              Arm re-command rate (Hz).
        gripper_target:  Gripper target (0=closed, 1=open).
    """
    arm = yor.left_arm if side == "left" else yor.right_arm
    assert arm is not None, f"{side} arm is None"

    dt = 1.0 / max(1.0, hz)

    print(f"\n[WBC] Target: side={side}  "
          f"xyz=({x_w:.3f}, {y_w:.3f}, {z_w:.3f})  "
          f"yaw={math.degrees(yaw):.1f}° pitch={math.degrees(pitch):.1f}° "
          f"roll={math.degrees(roll):.1f}°")

    # ── Stage 1: Lift ──────────────────────────────────────────────────────────
    desired_lift_h = float(np.clip(
        z_w - FLOOR_TO_BASE_Z - ARM_Z_ON_LIFT,
        LIFT_MIN_HEIGHT, LIFT_MAX_HEIGHT
    ))

    if not skip_lift:
        command_lift_to(yor, desired_lift_h)

    actual_lift_h = desired_lift_h
    if hasattr(yor, 'get_lift_height') and callable(yor.get_lift_height):
        try:
            actual_lift_h = float(yor.get_lift_height())
        except Exception:
            pass

    # ── Stage 1: Base ─────────────────────────────────────────────────────────
    if not skip_base:
        pose = get_robot_xy_theta(yor)
        if pose is not None:
            rx, ry, _ = pose
            goal_theta = math.atan2(y_w - ry, x_w - rx)
            command_base_to(yor, rx, ry, goal_theta)
        else:
            print("[WBC] Skipping base nav (no ZED pose).")

    # ── Stage 2: Continuous arm tracking loop ──────────────────────────────────
    print(f"\n[WBC] Entering continuous arm tracking at {hz:.0f}Hz. Ctrl+C to stop.")
    rate = RateLimiter(hz, name="wbc_arm")

    try:
        while True:
            # Get latest ZED base pose
            base_se3 = zed_pose_as_se3(zed_sub)

            # Re-derive arm target in arm's local frame using current base pose
            target_arm_frame = world_target_to_arm_frame(
                base_se3, actual_lift_h, side,
                x_w, y_w, z_w, yaw, pitch, roll
            )

            arm.set_ee_target(
                target_arm_frame,
                gripper_target=gripper_target,
                preview_time=dt,
            )
            rate.sleep()

    except KeyboardInterrupt:
        print("\n[WBC] Tracking stopped.")


# ── CLI entry point ────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="YOR Whole-Body Control w/ continuous arm tracking")
    p.add_argument("--side",      choices=["left", "right"], default="left")
    p.add_argument("--x",         type=float, required=True,  help="World target X (m)")
    p.add_argument("--y",         type=float, required=True,  help="World target Y (m)")
    p.add_argument("--z",         type=float, required=True,  help="World target Z (m)")
    p.add_argument("--yaw",       type=float, default=0.0, help="Target yaw   (deg)")
    p.add_argument("--pitch",     type=float, default=0.0, help="Target pitch (deg)")
    p.add_argument("--roll",      type=float, default=0.0, help="Target roll  (deg)")
    p.add_argument("--hz",        type=float, default=ARM_TRACKING_HZ,
                   help="Arm re-command rate (Hz)")
    p.add_argument("--grip",      type=float, default=1.0, help="Gripper target 0–1")
    p.add_argument("--no-lift-home", action="store_true", help="Skip lift homing")
    p.add_argument("--skip-base",    action="store_true", help="Skip base navigation")
    p.add_argument("--skip-lift",    action="store_true", help="Skip lift adjustment")
    p.add_argument("--zed-host", default=ZED_HOST)
    p.add_argument("--zed-port", type=int, default=ZED_PORT)
    return p.parse_args()


def main():
    args = parse_args()
    yaw   = math.radians(args.yaw)
    pitch = math.radians(args.pitch)
    roll  = math.radians(args.roll)

    yor = YOR(no_arms=False)

    print("[WBC] Initializing YOR and homing arms...")
    yor.init()

    if not args.no_lift_home:
        home_lift(yor)

    wait_for_zed(yor, timeout=10.0)

    print("[WBC] Connecting to ZED subscriber...")
    zed_sub = make_zed_subscriber(args.zed_host, args.zed_port)

    try:
        reach_and_hold(
            yor, zed_sub,
            side          = args.side,
            x_w           = args.x,
            y_w           = args.y,
            z_w           = args.z,
            yaw           = yaw,
            pitch         = pitch,
            roll          = roll,
            skip_base     = args.skip_base,
            skip_lift     = args.skip_lift,
            hz            = args.hz,
            gripper_target= args.grip,
        )
    finally:
        print("[WBC] Stopping arms...")
        if hasattr(yor, 'left_arm')  and yor.left_arm:
            yor.left_arm.stop()
        if hasattr(yor, 'right_arm') and yor.right_arm:
            yor.right_arm.stop()


if __name__ == "__main__":
    main()
