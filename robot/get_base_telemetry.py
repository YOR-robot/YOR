#!/usr/bin/env python3

import time
import argparse
import numpy as np
from yor import YOR

def main():
    parser = argparse.ArgumentParser(description="Fetch and display telemetry from the YOR robot base.")
    parser.add_argument("--no-arms", action="store_true", default=True, 
                        help="Initialize only the base (default: True)")
    parser.add_argument("--hz", type=float, default=20.0, 
                        help="Sampling frequency in Hz (default: 20.0)")
    args = parser.parse_args()

    # Initialize the robot
    robot = YOR(no_arms=args.no_arms)
    print(f"Initializing YOR (no_arms={args.no_arms})...")
    robot.init()
    
    # Wait for the background control loop to synchronize
    time.sleep(1.0)
    
    print("\nTelemetry Streaming (Press Ctrl+C to stop)...")
    print("-" * 115)
    header = (
        f"{'Time':<10} | "
        f"{'S1 (deg)':<8} {'D1 (cnt)':<10} | "
        f"{'S2 (deg)':<8} {'D2 (cnt)':<10} | "
        f"{'S3 (deg)':<8} {'D3 (cnt)':<10} | "
        f"{'S4 (deg)':<8} {'D4 (cnt)':<10} | "
        f"{'Lift(m)':<8}"
    )
    print(header)
    print("-" * 115)

    try:
        while True:
            # get_base_encoders() returns a dictionary with steering and drive data
            telemetry = robot.get_base_encoders()
            
            s_deg = telemetry['steer_deg']
            d_cnt = telemetry['drive_counts']
            lift  = telemetry['lift_height_m']
            ts_str = time.strftime('%H:%M:%S', time.localtime(telemetry['timestamp']))
            
            row = (
                f"{ts_str:<10} | "
                f"{s_deg[0]:8.1f} {d_cnt[0]:10.0f} | "
                f"{s_deg[1]:8.1f} {d_cnt[1]:10.0f} | "
                f"{s_deg[2]:8.1f} {d_cnt[2]:10.0f} | "
                f"{s_deg[3]:8.1f} {d_cnt[3]:10.0f} | "
                f"{lift if lift is not None else 0.0:8.3f}"
            )
            print(row)
            
            time.sleep(1.0 / args.hz)
            
    except KeyboardInterrupt:
        print("\nStopping telemetry stream. Goodbye!")

if __name__ == "__main__":
    main()
