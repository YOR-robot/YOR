#!/usr/bin/env python3
"""
Gravity Compensation Hold Mode Test (Interactive)
===================================================
Tests real gravity compensation on the physical Nero arm with live tuning.
Allows the arm to be easily moved by hand, but it will hold its position
when released. 

Type 'p 3.0' to set Kp to 3.0.
Type 's 0.7' to set GC scale to 0.7.
Type 'q' to quit.
"""

import sys, time, threading, select, os
import numpy as np

# Add the parent directory of robot_nero to Python path
sys.path.insert(0, "/home/cone-e/code/robot_nero")

from robot_nero.arm.arm_nerolib import ArmNode

CAN_PORT = "can_left"
MJCF_PATH = "/home/cone-e/code/robot_nero/robot_nero/cone-e-description/nero-mit-mode.mjcf"
IS_LEFT_ARM = True

# Initial gains
HOLD_KP = 2.0
HOLD_KD = 0.5
GC_SCALE = 0.4

SYNC_RATE_HZ = 50
SYNC_PERIOD = 1.0 / SYNC_RATE_HZ


def run():
    print("[GravComp] Initializing arm...")
    arm = ArmNode(
        can_port=CAN_PORT,
        mjcf_path=MJCF_PATH,
        is_left_arm=IS_LEFT_ARM,
        gravity_comp_scale=GC_SCALE
    )

    if not arm.nero:
        print("[GravComp] Failed to connect to arm. Exiting.")
        return

    print("[GravComp] Arm connected. Homing...")
    arm.home()
    time.sleep(2.0)

    print("\n[GravComp] Entering Gravity Compensation Hold Mode.")
    print("           Move the arm freely. Release to hold position.")
    
    # Enable GC, sync target, lower gains
    arm.set_gravity_comp(True)
    time.sleep(0.02)
    arm.sync_target()
    time.sleep(0.05)
    arm.set_gain(HOLD_KP, HOLD_KD)

    print("\n[GravComp] Interactive Tuning Mode\n"
          "Commands:\n"
          "  p <val>  - Set Kp (e.g., 'p 3.0')\n"
          "  s <val>  - Set GC Scale (e.g., 's 0.7')\n"
          "  q        - Quit\n")

    stop_event = threading.Event()

    def input_thread_func():
        global HOLD_KP, HOLD_KD, GC_SCALE
        while not stop_event.is_set():
            # Non-blocking read (linux)
            if select.select([sys.stdin], [], [], 0.1)[0]:
                cmd = sys.stdin.readline().strip().lower()
                if not cmd: continue
                try:
                    if cmd == 'q':
                        stop_event.set()
                        break
                    elif cmd.startswith('p '):
                        val = float(cmd.split()[1])
                        HOLD_KP = val
                        arm.set_gain(HOLD_KP, HOLD_KD)
                        print(f"  --> Updated Kp to {val}")
                    elif cmd.startswith('s '):
                        val = float(cmd.split()[1])
                        GC_SCALE = val
                        arm.set_gravity_comp_scale(GC_SCALE)
                        print(f"  --> Updated GC Scale to {val}")
                except Exception as e:
                    print(f"Invalid command: {cmd}")

    input_thread = threading.Thread(target=input_thread_func, daemon=True)
    input_thread.start()

    loop_count = 0
    loop_times = []

    try:
        while not stop_event.is_set():
            loop_start = time.perf_counter()

            # Anchor the "spring" mechanism at the current physical position
            arm.sync_target()

            # Terminal output occasionally
            loop_count += 1
            if loop_count % SYNC_RATE_HZ == 0:
                q_actual = arm.get_joint_positions()
                t_actual = arm.get_joint_torques()
                
                # Loop stats
                if loop_times:
                    mean_t = np.mean(loop_times) * 1000
                    max_t = np.max(loop_times) * 1000
                    min_t = np.min(loop_times) * 1000
                    overruns = sum(1 for t in loop_times if t > SYNC_PERIOD)
                    print(f"[{SYNC_RATE_HZ}Hz] loop: mean={mean_t:.2f}ms  max={max_t:.2f}ms  min={min_t:.2f}ms  overruns={overruns}/{len(loop_times)}")
                    print(f"         q=[{', '.join(f'{x:.3f}' for x in q_actual)}]")
                    print(f"         τ=[{', '.join(f'{x:.2f}' for x in t_actual)}]")
                    loop_times.clear()

            # Pacing
            elapsed = time.perf_counter() - loop_start
            loop_times.append(elapsed)
            sleep_time = SYNC_PERIOD - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[GravComp] Caught Ctrl-C.")
    finally:
        stop_event.set()
        print("[GravComp] Exiting. Restoring firm gains.")
        arm.set_gravity_comp(False)
        arm.set_gain(10.0, 0.8)
        
        t = threading.Thread(target=arm.stop)
        t.daemon = True
        t.start()
        t.join(timeout=2.0)
        os._exit(0)

if __name__ == "__main__":
    run()
