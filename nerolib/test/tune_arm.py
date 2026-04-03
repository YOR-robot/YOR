import time
import numpy as np
import sys
from pathlib import Path

# Add robot_nero to path if needed
sys.path.append("/home/cone-e/code/robot_nero")

from robot_nero.arm.arm_nerolib import ArmNode
from nerolib import ControlMode, MoveMode

def main():
    _HERE = Path(__file__).parent
    mjcf_path = "/home/cone-e/code/robot_nero/robot_nero/cone-e-description/nero-welded-base-and-lift.mjcf"
    
    # Initialize Left Arm
    print("[Tuner] Initializing Left Arm on can_left...")
    arm = ArmNode(
        can_port="can_left",
        mjcf_path=mjcf_path,
        is_left_arm=True
    )
    
    print("[Tuner] Starting controller...")
    arm.init() # This calls nero.start() which includes reset_arm()
    
    try:
        while True:
            print("\n--- NeroLib Dynamic Tuner ---")
            print("1. Set Compliant Mode (Stiffness=0, Damping=0.5)")
            print("2. Set Spring Mode (Stiffness=5.0, Damping=0.3)")
            print("3. Toggle Gravity Compensation")
            print("4. Set Stiff Mode (Default Gains)")
            print("5. Custom Gains")
            print("6. Show Status")
            print("q. Quit")
            
            choice = input("Select an option: ").strip().lower()
            
            if choice == '1':
                print("[Tuner] Entering Compliant Mode")
                arm.set_compliant_mode()
                
            elif choice == '2':
                print("[Tuner] Entering Spring Mode")
                arm.set_spring_mode()
                
            elif choice == '3':
                enabled = input("Enable Grav Comp? (y/n): ").strip().lower() == 'y'
                arm.set_gravity_comp(enabled)
                print(f"[Tuner] Gravity compensation {'enabled' if enabled else 'disabled'}")
                
            elif choice == '4':
                print("[Tuner] Returning to Stiff Mode")
                arm.set_firm_mode()
                
            elif choice == '5':
                kp = float(input("Enter Kp: "))
                kd = float(input("Enter Kd: "))
                arm.set_gain(kp, kd)
                
            elif choice == '6':
                q = arm.get_joint_positions()
                print(f"Current Joint Positions: {np.round(q, 4)}")
                
            elif choice == 'q':
                break
            
            time.sleep(0.1)

    except KeyboardInterrupt:
        pass
    finally:
        import threading
        import os
        print("[Tuner] Stopping arm (2s timeout)...")
        
        # Run stop in a background thread so we don't hang the terminal
        stop_thread = threading.Thread(target=arm.stop)
        stop_thread.daemon = True
        stop_thread.start()
        stop_thread.join(timeout=2.0)
        
        if stop_thread.is_alive():
            print("[Tuner] Shutdown timed out. Forcing exit.")
        else:
            print("[Tuner] Shutdown complete.")
        
        os._exit(0)

if __name__ == "__main__":
    main()
