"""
Compare FK of nero_cone-e_left_fixed.urdf vs MJCF
by loading both into MuJoCo and comparing end-effector positions.
Also checks gimbal lock effect on joint orientations.
"""
import mujoco
import numpy as np
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation

MJCF = "/home/cone-e/code/robot_nero/robot_nero/cone-e-description/nero-welded-base-and-lift.mjcf"
URDF = "/home/cone-e/code/nerolib/urdf/nero_cone-e_left_fixed.urdf"
ARM  = "left_arm"

# ------ Load MJCF ------
mjcf_model = mujoco.MjModel.from_xml_path(MJCF)
mjcf_data  = mujoco.MjData(mjcf_model)
j_addrs    = [mjcf_model.joint(f"{ARM}_joint{i}").id for i in range(1,8)]
qpos_addrs = [mjcf_model.jnt_qposadr[j] for j in j_addrs]
ee_site    = mjcf_model.site(f"{ARM}_ee").id

# ------ MJCF FK ------
test_poses = [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.57, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 0.5, -0.5, 0.5, 0.0, 0.0, 0.0],
]

mjcf_ee = {}
print("=== MJCF end-effector FK ===")
for q in test_poses:
    mujoco.mj_resetData(mjcf_model, mjcf_data)
    for addr, val in zip(qpos_addrs, q):
        mjcf_data.qpos[addr] = val
    mujoco.mj_forward(mjcf_model, mjcf_data)
    ee_pos = mjcf_data.site_xpos[ee_site].copy()
    mjcf_ee[tuple(q)] = ee_pos
    print(f"  q={[round(v,2) for v in q]}  ee={[round(x,4) for x in ee_pos]}")

# ------ Parse URDF joints  ------
def parse_urdf_joints(path):
    tree = ET.parse(path)
    joints = {}
    for j in tree.findall(".//joint"):
        name = j.get("name")
        orig = j.find("origin")
        if orig is not None:
            xyz = [float(x) for x in orig.get("xyz","0 0 0").split()]
            rpy = [float(x) for x in orig.get("rpy","0 0 0").split()]
            joints[name] = {"xyz": xyz, "rpy": rpy, "type": j.get("type","?")}
    return joints

urdf_joints = parse_urdf_joints(URDF)
print("\n=== URDF joint origins (nero_cone-e_left_fixed) ===")
for name in [f"joint{i}" for i in range(1,8)]:
    j = urdf_joints.get(name, {})
    print(f"  {name}: xyz={j.get('xyz','?')}  rpy={[round(v,5) for v in j.get('rpy',[0,0,0])]}")

# ------ Cross-check: MJCF body quaternions as Euler ------
print("\n=== MJCF left arm body quats → RPY (for comparison) ===")
for i in range(1, 8):
    bname = f"{ARM}_link{i}"
    try:
        bid = mjcf_model.body(bname).id
        pos = mjcf_model.body_pos[bid]
        quat_wxyz = mjcf_model.body_quat[bid]  # wxyz
        # scipy uses xyzw
        quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
        rpy = Rotation.from_quat(quat_xyzw).as_euler('xyz')
        print(f"  link{i}: pos={[round(v,5) for v in pos]}  rpy(euler_xyz)={[round(v,5) for v in rpy]}")
    except Exception as e:
        print(f"  link{i}: {e}")

# ------ Compare rpy values numerically ------
print("\n=== COMPARISON: URDF rpy vs MJCF body rpy ===")
mjcf_rpy = {}
for i in range(1, 8):
    bname = f"{ARM}_link{i}"
    try:
        bid = mjcf_model.body(bname).id
        quat_wxyz = mjcf_model.body_quat[bid]
        quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
        mjcf_rpy[i] = Rotation.from_quat(quat_xyzw).as_euler('xyz')
    except:
        mjcf_rpy[i] = None

for i in range(1, 8):
    name = f"joint{i}"
    j = urdf_joints.get(name)
    if j is None:
        print(f"  joint{i}: NOT FOUND in URDF")
        continue
    u_rpy = np.array(j['rpy'])
    m_rpy = mjcf_rpy.get(i)
    if m_rpy is None:
        print(f"  joint{i}: MJCF link not found")
        continue
    diff = np.abs(u_rpy - m_rpy)
    max_diff_deg = np.max(diff) * 180 / np.pi
    flag = " ⚠ MISMATCH" if max_diff_deg > 2.0 else " ✓ OK"
    print(f"  joint{i}: urdf_rpy={[round(v,4) for v in u_rpy]}  mjcf_rpy={[round(v,4) for v in m_rpy]}  diff={max_diff_deg:.2f}°{flag}")

print()
