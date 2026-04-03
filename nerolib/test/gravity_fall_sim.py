#!/usr/bin/env python3
"""Gravity fall simulation - where does each joint end up under pure gravity?"""
import mujoco
import numpy as np

MJCF_PATH = "/home/cone-e/code/robot_nero/robot_nero/cone-e-description/nero-welded-base-and-lift.mjcf"
ARM = "left_arm"
JOINT_NAMES = [f"{ARM}_joint{i}" for i in range(1, 8)]
Q_HOME = [1.57, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

model = mujoco.MjModel.from_xml_path(MJCF_PATH)
data  = mujoco.MjData(model)

qpos_addrs, dof_addrs = [], []
for name in JOINT_NAMES:
    jid = model.joint(name).id
    qpos_addrs.append(model.jnt_qposadr[jid])
    dof_addrs.append(model.jnt_dofadr[jid])

act_ids = [model.actuator(f"{ARM}_joint{i}_pos").id for i in range(1, 8)]

# ====================================================
# Test 1: gravcomp=1, KP=0 — should hold at home
# ====================================================
mujoco.mj_resetData(model, data)
for addr, val in zip(qpos_addrs, Q_HOME):
    data.qpos[addr] = val
for aid in act_ids:
    model.actuator_gainprm[aid, 0] = 0.0
    model.actuator_biasprm[aid, 1] = 0.0
mujoco.mj_forward(model, data)

print("=== GC torques at home q=[1.57, 0...] (what C++ RNEA should produce) ===")
for name, daddr in zip(JOINT_NAMES, dof_addrs):
    print(f"  {name}: {data.qfrc_bias[daddr]:+.4f} Nm")

for _ in range(int(3.0 / model.opt.timestep)):
    mujoco.mj_step(model, data)
q1 = [data.qpos[a] for a in qpos_addrs]
print(f"\n[Test 1] gravcomp=1, KP=0, 3s hold:  q={[round(x,3) for x in q1]}")
print(f"  drift = {[round(abs(q1[i]-Q_HOME[i]),4) for i in range(7)]}")

# ====================================================
# Test 2: gravcomp=0, KP=0 — free fall prediction
# ====================================================
mujoco.mj_resetData(model, data)
for addr, val in zip(qpos_addrs, Q_HOME):
    data.qpos[addr] = val
# Kill gravcomp on arm bodies
for b in range(model.nbody):
    if ARM in model.body(b).name:
        model.body_gravcomp[b] = 0.0
for aid in act_ids:
    model.actuator_gainprm[aid, 0] = 0.0
    model.actuator_biasprm[aid, 1] = 0.0
mujoco.mj_forward(model, data)

for _ in range(int(3.0 / model.opt.timestep)):
    mujoco.mj_step(model, data)
q2 = [data.qpos[a] for a in qpos_addrs]
print(f"\n[Test 2] gravcomp=0, KP=0, 3s free fall:  q={[round(x,3) for x in q2]}")
print()
print("=== SIGN CHECK: Compare to real arm in compliant mode (Kp=0, KD=0) ===")
for i, name in enumerate(JOINT_NAMES):
    delta = round(q2[i] - Q_HOME[i], 3)
    sign = "+" if delta > 0 else "-"
    print(f"  {name}: home={Q_HOME[i]:.2f} → fell to {q2[i]:.3f}  (delta={delta:+.3f})")
print()
print(f">>> Joint2 sim prediction: {q2[1]:.3f} rad ({q2[1]*180/3.14159:.1f}°)")
print(f"    If hardware joint2 shows +{abs(q2[1]):.2f} (opposite sign) → joint2 encoder is FLIPPED")
print(f"    If hardware joint2 shows ~{q2[1]:.2f} (same sign) → sign is OK, torque magnitude wrong")
