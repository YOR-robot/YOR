import time
import sys
from pathlib import Path

# Add the parent directory to sys.path if needed
sys.path.append(str(Path(__file__).parent.parent))

from robot.yor import YOR

def main():
    # Initialize YOR with arms enabled
    yor = YOR(no_arms=False)
    
    print("Initializing YOR...")
    yor.init()
    
    print("\n--- YOR Initialized ---")
    print("Current Left Arm Joint Positions:", yor.get_left_joint_positions())
    
    print("\nEnabling Compliant Mode (Gravity Comp = False, Low Stiffness/Damping) on LEFT ARM...")
    # set_compliant_mode turns off grav comp and lowers stiffness by default
    yor.left_arm.set_compliant_mode() 
    
    # We turn on gravity compensation so it doesn't just collapse
    print("Enabling Gravity Compensation on LEFT ARM...")
    yor.left_arm.set_gravity_comp(True)
    
    # Scale factor: nerolib defaults to 0.0 internally, so this MUST be set.
    # Start at 1.0 (raw Pinocchio output). Motor clips at ±8Nm so Joint 1
    # may still sag slightly. Increase carefully if the arm still droops.
    GC_SCALE = 1.0
    yor.left_arm.set_gravity_comp_scale(GC_SCALE)
    print(f"Gravity comp scale set to {GC_SCALE}")
    
    print("\nArm is now in compliant mode WITH gravity compensation.")
    print("Move the arm manually. It should feel 'weightless' or much easier to move.")
    
    try:
        print("Press Ctrl+C to exit and stop the robot...")
        while True:
            time.sleep(1.0)
            q = yor.get_left_joint_positions()
            tau = yor.left_arm.get_joint_torques()
            print(f"Q:   {[round(x,3) for x in q]}")
            print(f"Tau: {[round(x,3) for x in tau]} (observed motor torque)")
            
    except KeyboardInterrupt:
        print("\nExiting...")
        
    finally:
        print("Stopping left arm...")
        yor.left_arm.stop()
        if hasattr(yor, 'right_arm'):
             yor.right_arm.stop()

if __name__ == "__main__":
    main()
