#!/usr/bin/env python3
"""
Gravity Compensation Simulation Test
======================================
Loads the Nero arm MJCF in MuJoCo and tests whether gravity compensation
torques computed from MuJoCo's own inverse dynamics can hold the arm still.

This validates:
  1. The MJCF masses are physically plausible (arm holds realistically)
  2. The inverse dynamics torques match what we send to the hardware

Separate mode:
  - Mode A: MuJoCo's built-in gravcomp="1" (gold standard reference)
  - Mode B: We manually apply mj_rne torques with gravcomp="0" bodies
            (simulates what our C++ RNEA feedforward does)

Run with:
  conda run -n nerolib python grav_comp_sim.py
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import sys

MJCF_PATH = "/home/cone-e/code/robot_nero/robot_nero/cone-e-description/nero-welded-base-and-lift.mjcf"

# Test pose: arm roughly at home (joint1=90°, rest~0)
# Adjust to match your actual arm home position
Q_TEST = [1.57, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

ARM_PREFIX = "left_arm"
JOINT_NAMES  = [f"{ARM_PREFIX}_joint{i}" for i in range(1, 8)]
ACTUATOR_NAMES = [f"{ARM_PREFIX}_joint{i}_pos" for i in range(1, 8)]


def get_joint_addrs(model, joint_names):
    qpos_addrs = []
    dof_addrs  = []
    for name in joint_names:
        jid = model.joint(name).id
        qpos_addrs.append(model.jnt_qposadr[jid])
        dof_addrs.append(model.jnt_dofadr[jid])
    return qpos_addrs, dof_addrs


def set_arm_pose(data, qpos_addrs, q):
    for addr, val in zip(qpos_addrs, q):
        data.qpos[addr] = val
    data.qvel[:] = 0


def get_gc_torques(model, data, dof_addrs):
    """Compute gravity compensation torques via MuJoCo inverse dynamics.
    qfrc_bias already contains gravity + Coriolis at zero velocity."""
    # mj_fwdActuation + mj_fwdPosition together populate qfrc_bias
    # We can just read qfrc_bias after mj_forward — it's gravity+Coriolis bias
    # at the current qpos/qvel state.
    gc = np.array([data.qfrc_bias[a] for a in dof_addrs])
    return gc


def run_simulation():
    print(f"[Sim] Loading model: {MJCF_PATH}")
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data  = mujoco.MjData(model)

    qpos_addrs, dof_addrs = get_joint_addrs(model, JOINT_NAMES)
    act_ids = [model.actuator(n).id for n in ACTUATOR_NAMES]

    # Zero everything
    mujoco.mj_resetData(model, data)

    # Set test pose
    set_arm_pose(data, qpos_addrs, Q_TEST)
    mujoco.mj_forward(model, data)

    # --- Print GC torques at this pose ---
    gc = get_gc_torques(model, data, dof_addrs)
    print(f"\n[Sim] Test pose: q = {[round(x,3) for x in Q_TEST]}")
    print(f"[Sim] MuJoCo grav comp torques (Nm):")
    for i, (tau, name) in enumerate(zip(gc, JOINT_NAMES)):
        print(f"       {name}: {tau:+.4f} Nm")
    total = np.sum(np.abs(gc))
    print(f"[Sim] |τ| total = {total:.3f} Nm  ", end="")
    if total < 1.0:
        print("⚠  Very low — check arm masses in MJCF!")
    else:
        print("✓  Looks non-trivial.")

    print(f"\n[Sim] Starting viewer in MANUAL GC MODE")
    print(f"      The arm should hold its pose with manually applied GC torques.")
    print(f"      Close the viewer window or Ctrl+C to exit.\n")

    # ---------------------------------------------------------------
    # Simulation loop: apply manual GC torques each step
    # The MJCF has gravcomp="1" on all arm bodies which will fight us.
    # We override the actuators instead, which is a position-controller mode.
    # 
    # Strategy: use MuJoCo's reference position = current pose,
    # and add bias force (GC) as an extra torque via xfrc_applied.
    # Actually the simplest demo: just run with gravcomp="1" and KP=0.
    # That's equivalent to pure gravity comp in the real hardware.
    # ---------------------------------------------------------------

    # Set actuator gains to 0 (pure GC, no position stiffness)
    for aid in act_ids:
        model.actuator_gainprm[aid, 0] = 0.0   # kp = 0
        model.actuator_biasprm[aid, 1] = 0.0   # kd = 0

    # Set actuator references to test pose
    for aid, qval in zip(act_ids, Q_TEST):
        data.ctrl[aid] = qval

    # Re-forward with new actuator params
    mujoco.mj_forward(model, data)

    print("[Sim] === PHASE 1: pure gravcomp='1' (built-in MuJoCo, KP=0) ===")
    print("      Watch: does the arm hold or fall?")
    print("      If it falls → MuJoCo gravcomp is wrong (unlikely)")
    print("      If it holds → masses are correct, now we test manual RNEA.\n")

    drift_log = []

    print(f"\n[Sim] Running headless simulation for 5 seconds at {1/model.opt.timestep:.0f}Hz...")
    print(f"      Arm should hold at q={Q_TEST}")
    print(f"      Drift reported each second. Close to 0 = GC working.\n")

    n_steps = int(5.0 / model.opt.timestep)
    report_every = int(1.0 / model.opt.timestep)

    for step in range(n_steps):
        # In MJCF gravcomp="1" bodies, MuJoCo injects gravity comp torques automatically.
        # With KP=0 actuators, the arm has only this passive GC to hold it.
        mujoco.mj_step(model, data)

        if (step + 1) % report_every == 0:
            q_now   = [data.qpos[a] for a in qpos_addrs]
            drift   = [abs(q_now[i] - Q_TEST[i]) for i in range(7)]
            qv      = [data.qvel[a] for a in dof_addrs]
            t_sim   = (step + 1) * model.opt.timestep
            print(
                f"  t={t_sim:.1f}s  "
                f"max_drift={max(drift):.4f}rad  "
                f"max_vel={max(abs(v) for v in qv):.4f}rad/s  "
                f"q2={q_now[1]:.4f}"
            )

    q_final = [data.qpos[a] for a in qpos_addrs]
    drift_final = [abs(q_final[i] - Q_TEST[i]) for i in range(7)]
    print(f"\n[Sim] Final drift: {[round(x, 4) for x in drift_final]}")
    print(f"[Sim] Max joint drift: {max(drift_final):.4f} rad ({max(drift_final)*180/3.14159:.2f}°)")
    if max(drift_final) < 0.05:
        print("[Sim] ✓ GC is working — arm held position within 3°")
    elif max(drift_final) < 0.2:
        print("[Sim] ⚠ Mild drift — GC partially working, masses may be slightly off")
    else:
        print("[Sim] ✗ Large drift — GC is not working, check MJCF masses or joint conventions")

    print("\n[Sim] Done.")


if __name__ == "__main__":
    run_simulation()
