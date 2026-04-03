#!/usr/bin/env python3
"""
GC Torque Delivery Test
========================
Checks whether gravity compensation torques are actually reaching the motors.

Protocol:
  1. Home the arm
  2. Go fully compliant (Kp=0, Kd=0) — arm falls freely under gravity
  3. Measure fall rate of joint2 for 1 second (baseline)
  4. Enable gravity compensation
  5. Measure deceleration of joint2 over next 1 second
  6. Compare: if GC is working, fall rate should reduce significantly

Joint2 is the clearest indicator: sim says it needs +7.6 Nm to hold against gravity.
If fall rate drops after enabling GC → torques are reaching the motor.
If fall rate is unchanged → GC feedforward is not arriving at the motor.
"""

import sys, time, threading
import numpy as np

CAN_PORT = "can_left"
MJCF_PATH = "/home/cone-e/code/robot_nero/robot_nero/cone-e-description/nero-mit-mode.mjcf"
IS_LEFT_ARM = True

sys.path.insert(0, "/home/cone-e/code/robot_nero")
from robot_nero.arm.arm_nerolib import ArmNode


def measure_fall_rate(arm, duration=1.0, sample_hz=20):
    """Sample joint2 for `duration` seconds, return (start, end, avg_velocity)."""
    readings = []
    dt = 1.0 / sample_hz
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < duration:
        q = arm.get_joint_positions()
        readings.append((time.perf_counter() - t0, q[1]))  # joint2
        time.sleep(dt)
    
    if len(readings) < 2:
        return readings[0][1], readings[-1][1], 0.0
    
    # Estimate velocity by linear regression
    ts = np.array([r[0] for r in readings])
    qs = np.array([r[1] for r in readings])
    slope = np.polyfit(ts, qs, 1)[0]  # rad/s
    return readings[0][1], readings[-1][1], slope


def run():
    print("[Test] Initializing arm...")
    arm = ArmNode(can_port=CAN_PORT, mjcf_path=MJCF_PATH, is_left_arm=IS_LEFT_ARM)
    if not arm.nero:
        print("[Test] Failed to connect.")
        return

    print("[Test] Homing...")
    arm.home()
    time.sleep(2.0)

    # --- Phase 1: Go compliant, no GC ---
    print("\n[Test] PHASE 1: Compliant mode (Kp=0, Kd=0.2), GC OFF")
    print("  Arm will fall freely. Measuring joint2 fall rate...")
    arm.sync_target()
    arm.set_gravity_comp(False)
    arm.set_gain(0.0, 0.2)   # Kp=0, tiny damping
    time.sleep(0.5)           # Let it start falling

    q_start, q_end, rate_no_gc = measure_fall_rate(arm, duration=1.5)
    print(f"  joint2: {q_start:.3f} → {q_end:.3f} rad,  rate={rate_no_gc:+.3f} rad/s")
    print(f"  (sim predicted fall direction: negative, ~{-1.74:.2f} rad equilibrium)")

    # --- Safety: stop falling if arm is near joint limit ---
    q_now = arm.get_joint_positions()
    if abs(q_now[1]) > 1.5:
        print("\n[Test] Joint2 near limit. Re-homing before GC phase.")
        arm.set_gain(5.0, 0.8)
        arm.home()
        time.sleep(2.5)
        arm.sync_target()
        arm.set_gravity_comp(False)
        arm.set_gain(0.0, 0.2)
        time.sleep(0.5)

    # --- Sync target to current pose ---
    arm.sync_target()
    time.sleep(0.05)

    # --- Phase 2: Enable GC ---
    print("\n[Test] PHASE 2: Enabling gravity compensation NOW...")
    arm.set_gravity_comp(True)
    time.sleep(0.1)   # Let GC command propagate

    q_gc_start, q_gc_end, rate_with_gc = measure_fall_rate(arm, duration=1.5)
    print(f"  joint2: {q_gc_start:.3f} → {q_gc_end:.3f} rad,  rate={rate_with_gc:+.3f} rad/s")

    # --- Analysis ---
    print(f"\n=== RESULTS ===")
    print(f"  Fall rate WITHOUT GC: {rate_no_gc:+.4f} rad/s")
    print(f"  Fall rate WITH GC:    {rate_with_gc:+.4f} rad/s")

    reduction = abs(rate_no_gc) - abs(rate_with_gc)
    if abs(rate_no_gc) > 0.001:
        reduction_pct = 100 * reduction / abs(rate_no_gc)
    else:
        reduction_pct = 0

    print(f"  Reduction: {reduction:+.4f} rad/s  ({reduction_pct:.0f}%)")
    print()

    if reduction_pct > 50:
        print("  ✓ GC is WORKING — fall rate reduced by >50%")
        print("    The URDF torques are reaching the motors.")
    elif reduction_pct > 10:
        print("  ~ GC is PARTIALLY WORKING — some torque is arriving")
        print("    Possible: torque scaling wrong, or URDF mass underestimates gravity")
    else:
        print("  ✗ GC is NOT WORKING — fall rate unchanged")
        print("    The feedforward torque command is not arriving at the motor.")
        print("    Check: set_joint_pos_vel_torque() and the CAN command format.")

    # --- Cleanup ---
    print("\n[Test] Stopping arm...")
    import os
    arm.set_gain(10.0, 0.8)
    t = threading.Thread(target=arm.stop); t.daemon = True; t.start()
    t.join(timeout=2.5)
    os._exit(0)


if __name__ == "__main__":
    run()
