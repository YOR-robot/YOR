import numpy as np
from scipy.spatial.transform import Rotation as R
import xml.etree.ElementTree as ET

def mjcf_to_urdf_params(mjcf_path):
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    
    bodies = {}
    for body in root.findall(".//body"):
        name = body.get("name")
        if not name: continue
        
        # Inertial
        inertial = body.find("inertial")
        if inertial is not None:
            mass = float(inertial.get("mass"))
            pos = [float(x) for x in inertial.get("pos").split()]
            # Quat in MJCF is w x y z
            quat = [float(x) for x in inertial.get("quat", "1 0 0 0").split()]
            diaginertia = [float(x) for x in inertial.get("diaginertia").split()]
        else:
            mass = 0.001
            pos = [0, 0, 0]
            quat = [1, 0, 0, 0]
            diaginertia = [1e-6, 1e-6, 1e-6]
            
        # Joint
        joint = body.find("joint")
        joint_name = joint.get("name") if joint is not None else None
        
        # Body transform relative to parent
        b_pos = [float(x) for x in body.get("pos", "0 0 0").split()]
        
        # Orientation can be quat, euler, or rpy in MJCF
        if body.get("quat"):
            b_quat = [float(x) for x in body.get("quat").split()]
            r = R.from_quat([b_quat[1], b_quat[2], b_quat[3], b_quat[0]])
            b_rpy = r.as_euler('xyz')
        elif body.get("euler"):
            b_euler = [float(x) for x in body.get("euler").split()]
            # MJCF default for euler is usually xyz
            b_rpy = b_euler
        else:
            b_rpy = [0, 0, 0]
            
        bodies[name] = {
            "mass": mass,
            "com": pos,
            "diaginertia": diaginertia,
            "pos": b_pos,
            "rpy": b_rpy,
            "joint_name": joint_name
        }
    return bodies

def generate_urdf(arm_prefix, bodies):
    arm_bodies = {k: v for k, v in bodies.items() if arm_prefix in k}
    
    # Sort or find root? The base_link is the parent-most of the arm
    base_name = f"{arm_prefix}_base_link"
    
    # Print header
    print(f'<?xml version="1.0" encoding="utf-8"?>')
    print(f'<robot name="nero_{arm_prefix}">')
    print(f'  <link name="dummy_link" />')
    
    # Root joint
    p = arm_bodies[base_name]
    print(f'  <joint name="base_to_dummy" type="fixed">')
    print(f'    <parent link="dummy_link" />')
    print(f'    <child link="base_link" />')
    print(f'    <origin xyz="{" ".join(map(str, p["pos"]))}" rpy="{" ".join(map(str, p["rpy"]))}" />')
    print(f'  </joint>')
    
    # Base link
    print(f'  <link name="base_link">')
    print(f'    <inertial>')
    print(f'      <origin xyz="{" ".join(map(str, p["com"]))}" rpy="0 0 0" />')
    print(f'      <mass value="{p["mass"]}" />')
    print(f'      <inertia ixx="{p["diaginertia"][0]}" ixy="0" ixz="0" iyy="{p["diaginertia"][1]}" iyz="0" izz="{p["diaginertia"][2]}" />')
    print(f'    </inertial>')
    print(f'  </link>')
    
    # Joints and Links 1-7 (arm chain)
    # We follow the chain: link1 -> link2 -> link3 -> link4 -> link5 -> link6 -> link7
    # Note: MJCF has link7 and then end_effector. In URDF we can weld them.
    
    parent = "base_link"
    for i in range(1, 8):
        name = f"{arm_prefix}_link{i}"
        if name not in arm_bodies: continue
        
        p = arm_bodies[name]
        
        # Joint
        print(f'  <joint name="joint{i}" type="revolute">')
        print(f'    <origin xyz="{" ".join(map(str, p["pos"]))}" rpy="{" ".join(map(str, p["rpy"]))}" />')
        print(f'    <parent link="{parent}" />')
        print(f'    <child link="link{i}" />')
        print(f'    <axis xyz="0 0 1" />')
        # Standard limits
        print(f'    <limit lower="-3.14" upper="3.14" effort="100" velocity="5" />')
        print(f'  </joint>')
        
        # Link
        mass = p["mass"]
        com = np.array(p["com"])
        inertia = np.array(p["diaginertia"])
        
        # [REMOVED] end_effector merger since user doesn't have it attached.

        print(f'  <link name="link{i}">')
        print(f'    <inertial>')
        print(f'      <origin xyz="{" ".join(map(str, com))}" rpy="0 0 0" />')
        print(f'      <mass value="{mass}" />')
        print(f'      <inertia ixx="{inertia[0]}" ixy="0" ixz="0" iyy="{inertia[1]}" iyz="0" izz="{inertia[2]}" />')
        print(f'    </inertial>')
        print(f'  </link>')
        
        parent = f"link{i}"
        
    print(f'</robot>')

mjcf_path = "/home/cone-e/code/robot_nero/robot_nero/cone-e-description/nero-welded-base-and-lift.mjcf"
params = mjcf_to_urdf_params(mjcf_path)

import sys
if len(sys.argv) > 1:
    generate_urdf(sys.argv[1], params)
else:
    print("Usage: extract_mjcf.py [left_arm|right_arm]")
