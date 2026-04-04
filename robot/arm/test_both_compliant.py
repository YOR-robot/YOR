import time
import sys
from pathlib import Path

# Add the parent directory to sys.path if needed
sys.path.append(str(Path(__file__).parent.parent.parent))

from robot.yor import YOR

def main():
    # Initialize YOR with arms enabled
    yor = YOR(no_arms=False)
    
    print("Initializing YOR...")
    yor.init()
    
    print("\n--- YOR Initialized ---")
    if hasattr(yor, 'left_arm') and yor.left_arm is not None:
        print("Current Left Arm Joint Positions:", yor.left_arm.get_joint_positions())
    if hasattr(yor, 'right_arm') and yor.right_arm is not None:
        print("Current Right Arm Joint Positions:", yor.right_arm.get_joint_positions())
    
    if hasattr(yor, 'left_arm') and yor.left_arm is not None:
        print("\nEnabling Compliant Mode (Gravity Comp = False, Low Stiffness/Damping) on LEFT ARM...")
        yor.left_arm.set_compliant_mode() 

    if hasattr(yor, 'right_arm') and yor.right_arm is not None:
        print("\nEnabling Compliant Mode (Gravity Comp = False, Low Stiffness/Damping) on RIGHT ARM...")
        yor.right_arm.set_compliant_mode() 
        
    print("\nArms are now in compliant mode WITH gravity compensation.")
    print("Move the arms manually. They should feel 'weightless' or much easier to move.")
    
    try:
        print("Press Ctrl+C to exit and stop the robot...")
        while True:
            time.sleep(1.0)
            if hasattr(yor, 'left_arm') and yor.left_arm is not None:
                q_l = yor.left_arm.get_joint_positions()
            
            if hasattr(yor, 'right_arm') and yor.right_arm is not None:
                q_r = yor.right_arm.get_joint_positions()

            
    except KeyboardInterrupt:
        print("\nExiting...")
        
    finally:
        if hasattr(yor, 'left_arm') and yor.left_arm is not None:
            print("Stopping left arm...")
            yor.left_arm.stop()
        if hasattr(yor, 'right_arm') and yor.right_arm is not None:
            print("Stopping right arm...")
            yor.right_arm.stop()

if __name__ == "__main__":
    main()
