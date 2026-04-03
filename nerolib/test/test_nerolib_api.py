import time
import numpy as np
import sys
from pathlib import Path

# Add robot_nero to path
sys.path.append("/home/cone-e/code/robot_nero")

from robot_nero.arm.arm_nerolib import ArmNode
from nerolib import ControlMode, MoveMode

def main():
    _HERE = Path(__file__).parent
    mjcf_path = "/home/cone-e/code/robot_nero/robot_nero/cone-e-description/nero-welded-base-and-lift.mjcf"
    
    # Initialize Left Arm
    print("[Test] Initializing Left Arm...")
    arm = ArmNode(
        can_port="can_left",
        mjcf_path=mjcf_path,
        is_left_arm=True
    )
    
    print("[Test] Starting controller...")
    arm.init()
    
    try:
        # 1. Test Position Mode (Stiff)
        print("[Test] Step 1: Holding stiff for 3 seconds...")
        arm.set_gain(15.0, 0.8)
        time.sleep(3)
        
        # 2. Test Compliant Mode (Zero Kp)
        print("[Test] Step 2: Entering Compliant Mode (Kp=0) for 5 seconds...")
        print("[Test] You should be able to move the arm easily now.")
        arm.set_mode(ControlMode.CAN_COMMAND, MoveMode.MIT)
        arm.set_gain(0.0, 0.5)
        arm.set_gravity_comp(True)
        time.sleep(5)
        
        # 3. Test Spring Mode (Low Kp)
        print("[Test] Step 3: Entering Spring Mode (Kp=5) for 5 seconds...")
        print("[Test] The arm should pull back to center.")
        arm.set_gain(5.0, 0.3)
        time.sleep(5)
        
        # 4. Return to Home
        print("[Test] Step 4: Returning to Stiff Mode and Homing...")
        arm.set_gain(15.0, 0.8)
        arm.home()
        time.sleep(2)

    except Exception as e:
        print(f"[Test] Error: {e}")
    finally:
        print("[Test] Shutting down...")
        arm.stop()

if __name__ == "__main__":
    main()
