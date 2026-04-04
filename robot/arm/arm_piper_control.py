import time
import numpy as np
from typing import Optional
from pathlib import Path

from piper_control import piper_interface, piper_init
import mink

from robot.arm.ik_solver import SingleArmIK

GRIPPER_OPEN = -22.0


class ArmNode:
    def __init__(
        self,
        can_port: str,
        mjcf_path: str,
        urdf_path: Optional[str] = None,
        solver_dt: float = 0.01,
        is_left_arm: bool = True,
    ):
        _HERE = Path(__file__).parent
        self.can_port = can_port
        # if mjcf_path is None:
        #     if is_left_arm:
        #         self.mjcf_path = (_HERE / "mujoco/scene_piper_left.xml").as_posix()
        #     else:
        #         self.mjcf_path = (_HERE / "mujoco/scene_piper_right.xml").as_posix()
        # else:
        #     self.mjcf_path = mjcf_path
        if urdf_path is None:
            if is_left_arm:
                self.urdf_path = (_HERE / "urdf/piper_description_left.xml").as_posix()
            else:
                self.urdf_path = (_HERE / "urdf/piper_description_right.xml").as_posix()
        else:
            self.urdf_path = urdf_path
        self.solver_dt = solver_dt

        # initialize arm
        # self.robot_config = RobotConfigFactory.get_instance().get_config("piper")
        # self.controller_config = ControllerConfigFactory.get_instance().get_config("joint_controller")
        # self.robot_config.urdf_path = self.urdf_path
        # self.controller_config.controller_dt = 0.005
        # self.controller_config.default_kp = np.array([7.0, 7.0, 7.0, 5.0, 5.0, 5.0]) # np.array([2.5, 2.5, 2.5, 1.0, 1.0, 1.0])
        # self.controller_config.default_kd = np.array([0.4, 0.4, 0.4, 0.3, 0.3, 0.3]) # np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        # self.controller_config.gravity_compensation = True
        # self.controller_config.interpolation_method = "cubic"
        # self.piper = PiperJointController(self.robot_config, self.controller_config, self.can_port)
        self.piper = piper_interface.PiperInterface(self.can_port)
        piper_init.reset_arm(
            self.piper,
            arm_controller=piper_interface.ArmController.POSITION_VELOCITY,
            move_mode=piper_interface.MoveMode.JOINT,
        )
        self.target: Optional[mink.SE3] = None
        self.gripper_target: Optional[float] = None
        self.target_timestamp: Optional[int] = None
        if is_left_arm:
            joint_names = [
                "left_arm_joint1",
                "left_arm_joint2",
                "left_arm_joint3",
                "left_arm_joint4",
                "left_arm_joint5",
                "left_arm_joint6",
            ]
            ee_frame = "left_arm_ee"
            self.home_q = np.array([0.0, 1.58065, -0.578175, 0.0, -0.912, 0.78])
        else:
            joint_names = [
                "right_arm_joint1",
                "right_arm_joint2",
                "right_arm_joint3",
                "right_arm_joint4",
                "right_arm_joint5",
                "right_arm_joint6",
            ]
            ee_frame = "right_arm_ee"
            self.home_q = np.array([0.0, 1.58065, -0.578175, 0.0, -0.912, -0.78])
            # self.home_q = np.array([0.0, 1.54, -0.875, 0, -0.5, 0])

        self.ik_solver = SingleArmIK(
            mjcf_path,
            solver_dt=self.solver_dt,
            joint_names=joint_names,
            ee_frame=ee_frame,
        )
        # if is_left_arm:
        #     self.home_q = np.array([0.0, 1.58065, -0.578175, 0.0, -0.912, 0.78])
        # else:
        #     self.home_q = np.array([0.0, 1.58065, -0.578175, 0.0, -0.912, -0.78])

    def init(self):
        # self.piper.reset_to_home()
        time.sleep(1.0)

        # home
        # q = self.home_q.copy()
        # print(f"q_home: {np.round(q, 4)}")
        # cmd = JointState(self.robot_config.joint_dof)
        # cmd.timestamp = self.piper.get_timestamp() + 1.0
        # cmd.pos = q
        # cmd.gripper_pos = 1.0 * GRIPPER_OPEN
        # self.piper.set_joint_cmd(cmd)
        # time.sleep(2.0)

        # q = self.piper.get_joint_state().pos
        q = np.array(self.piper.get_joint_positions())
        print(f"q_reached: {np.round(q, 4)}")
        self.ik_solver.init(q)
        self.target = self.ik_solver.forward_kinematics()

    # def check_timestamp(self, timestamp: int, max_delay: float = 0.1) -> bool:
    #     current_time = time.perf_counter_ns()
    #     delay = (current_time - timestamp) / 1e9
    #     if delay > max_delay or delay < 0:
    #         print(f"Skipping message because of delay: {delay}s")
    #         return False
    #     return True

    def home(self, gripper_target: float = 1.0):
        q = self.home_q.copy()
        # cmd = JointState(self.robot_config.joint_dof)
        # cmd.timestamp = self.piper.get_timestamp() + 1.0
        # cmd.pos = q
        # cmd.gripper_pos = gripper_target * GRIPPER_OPEN
        print("commanding home")
        self.piper.command_joint_positions(q)
        time.sleep(2.0)
        self.update_joint_positions()

    def set_joint_target(
        self, joint_target: np.ndarray, gripper_target: float | None = None, preview_time: float = 0.1
    ):
        # cmd = JointState(self.robot_config.joint_dof)
        # cmd.pos = joint_target
        # cmd.timestamp = self.piper.get_timestamp() + preview_time
        # if gripper_target is not None:
        #     # cmd.gripper_pos = gripper_target * GRIPPER_OPEN
        #     self.piper.command_gripper(position=gripper_target * GRIPPER_OPEN)
        self.piper.command_joint_positions(joint_target)

    def set_ee_target(self, ee_target: mink.SE3, gripper_target: Optional[float] = None, preview_time: float = 0.1):
        self.target = ee_target
        qd, _ = self.ik_solver.solve_ik(self.target)
        # cmd = JointState(self.robot_config.joint_dof)
        # cmd.pos = qd
        # if gripper_target is not None:
        #     cmd.gripper_pos = gripper_target * GRIPPER_OPEN
        # cmd.timestamp = self.piper.get_timestamp() + preview_time
        # self.piper.set_joint_cmd(cmd)
        self.piper.command_joint_positions(qd)

    # def set_gain(self, kp: np.ndarray, kd: np.ndarray):
    #     gain = Gain(kp, kd)
    #     self.piper.set_gain(gain)

    def get_joint_positions(self) -> np.ndarray:
        return np.array(self.piper.get_joint_positions())

    def get_ee_pose(self) -> mink.SE3:
        self.update_joint_positions()
        return self.ik_solver.forward_kinematics()

    def update_joint_positions(self):
        q = np.array(self.piper.get_joint_positions())
        self.ik_solver.update_configuration(q)

    def stop(self):
        print("called stop")
        pass
