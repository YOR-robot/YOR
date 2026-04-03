import time
import numpy as np
import sys
from pathlib import Path

# Force Python output to be completely unbuffered so you see all prints instantly
class Unbuffered(object):
    def __init__(self, stream):
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
    def writelines(self, datas):
        self.stream.writelines(datas)
        self.stream.flush()
    def __getattr__(self, attr):
        return getattr(self.stream, attr)
sys.stdout = Unbuffered(sys.stdout)

# Add the parent directory to Python path
sys.path.insert(0, "/home/cone-e/code/robot_nero")
from robot_nero.arm.arm_nerolib import ArmNode

CAN_PORT = "can_left"
MJCF_PATH = "/home/cone-e/code/robot_nero/robot_nero/cone-e-description/nero-mit-mode.mjcf"
IS_LEFT_ARM = True

def run():
    print("[SpeedTest] Initializing arm...", flush=True)
    # Initialize at a higher solver_dt rate to test if C++ loop can keep up
    arm = ArmNode(
        can_port=CAN_PORT,
        mjcf_path=MJCF_PATH,
        is_left_arm=IS_LEFT_ARM,
        solver_dt=0.005  # Ask C++ loop to run at 100Hz instead of default 50Hz
    )

    if not arm.nero:
        print("[SpeedTest] Failed to connect to arm. Exiting.", flush=True)
        return

    print("[SpeedTest] Arm connected. Homing...", flush=True)
    arm.home()
    time.sleep(2.0)

    print("\n===============================================")
    print("[SpeedTest] 1. Python -> C++ API Latency")
    print("===============================================")
    
    iters = 1000
    
    # 1. Test Read Latency
    start = time.perf_counter()
    for _ in range(iters):
        _ = arm.get_joint_positions()
    end = time.perf_counter()
    read_hz = iters / (end - start)
    read_ms = (end - start) / iters * 1000
    print(f" -> get_joint_positions(): {read_hz:8.2f} Hz  ({read_ms:.3f} ms/call)")

    # 2. Test Write Latency
    q_target = arm.get_joint_positions()
    start = time.perf_counter()
    for _ in range(iters):
        arm.set_joint_target(q_target, preview_time=0.0)
    end = time.perf_counter()
    write_hz = iters / (end - start)
    write_ms = (end - start) / iters * 1000
    print(f" -> set_joint_target():    {write_hz:8.2f} Hz  ({write_ms:.3f} ms/call)")

    print("\n===============================================")
    print("[SpeedTest] 2. Physical Tracking Bandwidth")
    print("===============================================")
    print("  Commanding a sine wave on Joint 1 (+/- 0.3 rad).")
    print("  We will sweep through frequencies to see tracking error.")
    print("  Press Ctrl+C to stop early.\n")
    
    freqs = [0.5, 1.0]  # Hz
    amplitude = 0.3 # rad
    
    # Run the python command loop at 200Hz to ensure we oversample the 100Hz C++ controller
    dt = 0.0025 
    
    base_q = arm.get_joint_positions().copy()
    
    try:
        for f in freqs:
            print(f"--- Running {f} Hz Sine Wave ---")
            duration = 3.0 # seconds per frequency
            start_time = time.perf_counter()
            
            errors = []
            
            while True:
                t = time.perf_counter() - start_time
                if t > duration:
                    break
                    
                target = base_q.copy()
                target[0] += amplitude * np.sin(2 * np.pi * f * t)
                
                # Send immediately
                arm.set_joint_target(target, preview_time=0.0)
                
                actual = arm.get_joint_positions()
                error = abs(target[0] - actual[0])
                errors.append(error)
                
                time.sleep(dt) 
                
            mean_err = np.mean(errors)
            max_err = np.max(errors)
            print(f"    Result: Mean Error = {mean_err:.4f} rad, Max Error = {max_err:.4f} rad")
            
            # recenter smoothly
            arm.set_joint_target(base_q, preview_time=1.0)
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        print("\n[SpeedTest] Interrupted by user.")
        
    finally:
        print("\n[SpeedTest] Done. Stopping arm...")
        arm.stop()

if __name__ == "__main__":
    run()
