import numpy as np
import mujoco
import mink


class SingleArmIK:
    def __init__(
        self,
        mjcf_path: str,
        solver_dt=0.01,
        joint_names: list[str] | None = None,
        ee_frame: str | None = None,
        use_lift: bool = False,
    ):
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.solver_dt = solver_dt

        if joint_names is None:
            joint_names = [
                "left_arm_joint1",
                "left_arm_joint2",
                "left_arm_joint3",
                "left_arm_joint4",
                "left_arm_joint5",
                "left_arm_joint6",
                "left_arm_joint7",
                # "Slider_1",
                # "Slider_2",
            ]
        if ee_frame is None:
            ee_frame = "left_arm_ee"

        # velocity_limits = {k: np.pi / 2 if "joint" in k else 0.05 for k in joint_names}
        self.dof_ids = np.array([self.model.joint(name).id for name in joint_names])
        self.actuator_ids = np.array([self.model.actuator(name + "_pos").id for name in joint_names if "joint" in name])
        if use_lift:
            self.lift_actuator_id = self.model.actuator("Lift").id

        self.configuration = mink.Configuration(self.model)
        self.end_effector_task = mink.FrameTask(
            frame_name=ee_frame,
            frame_type="site",
            position_cost=1.0,
            orientation_cost=0.1,
            lm_damping=1.0,
        )
        lift_cost = [1e-1]
        arm_cost = [1e-3] * 7
        # posture_cost = lift_cost + arm_cost + arm_cost
        self.posture_task = mink.PostureTask(
            self.model, cost=np.array(lift_cost + arm_cost + arm_cost)
        )
        self.tasks = [self.end_effector_task, self.posture_task]
        if use_lift:
            self.lift_equality_task = mink.EqualityConstraintTask(
                self.model,
                cost=1.0,
            )
            self.tasks.append(self.lift_equality_task)
        self.limits = [mink.ConfigurationLimit(self.model)]  # , mink.VelocityLimit(self.model, velocity_limits)]

        # initial setup
        self.initalized_ = False

    def init(self, q):
        current_q = self.configuration.q.copy()
        current_q[self.dof_ids] = q
        self.configuration.update(current_q)
        self.posture_task.set_target_from_configuration(self.configuration)
        self.initalized_ = True

    def get_home_q(self) -> np.ndarray:
        return self.model.key("home").qpos[self.dof_ids]

    def solve_ik(self, T_wt: mink.SE3, max_iter: int = 10, pos_eps: float = 1e-3, rot_eps: float = 1e-3):
        if not self.initalized_:
            raise ValueError("IK solver not initialized")
        self.end_effector_task.set_target(T_wt)
        is_solved = False
        for _ in range(max_iter):
            vel = mink.solve_ik(
                self.configuration, self.tasks, self.solver_dt, solver="daqp", damping=1e-5, limits=self.limits
            )
            self.configuration.integrate_inplace(vel, self.solver_dt)
            err = self.end_effector_task.compute_error(self.configuration)
            if np.linalg.norm(err[:3]) <= pos_eps and np.linalg.norm(err[3:]) <= rot_eps:
                is_solved = True
                break

        return self.configuration.q[self.dof_ids], is_solved

    def update_configuration(self, q: np.ndarray):
        current_q = self.configuration.q.copy()
        current_q[self.dof_ids] = q
        self.configuration.update(current_q)

    def forward_kinematics(self) -> mink.SE3:
        return self.configuration.get_transform_frame_to_world(
            self.end_effector_task.frame_name, self.end_effector_task.frame_type
        )


class BimanualArmIK:
    def __init__(self, mjcf_path: str, solver_dt=0.01):
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.solver_dt = solver_dt

        joint_names = [
            "left/joint1",
            "left/joint2",
            "left/joint3",
            "left/joint4",
            "left/joint5",
            "left/joint6",
            "right/joint1",
            "right/joint2",
            "right/joint3",
            "right/joint4",
            "right/joint5",
            "right/joint6",
        ]

        velocity_limits = {k: np.pi / 2 if "joint" in k else 0.05 for k in joint_names}
        self.dof_ids = np.array([self.model.joint(name).id for name in joint_names])
        self.actuator_ids = np.array([self.model.actuator(name + "_pos").id for name in joint_names])

        self.configuration = mink.Configuration(self.model)

        self.left_ee_task = mink.FrameTask(
            frame_name="left/ee",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=0.1,
            lm_damping=1.0,
        )
        self.right_ee_task = mink.FrameTask(
            frame_name="right/ee",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=0.1,
            lm_damping=1.0,
        )

        self.posture_task = mink.PostureTask(self.model, cost=np.array([1e-3] * 12))
        self.tasks = [self.left_ee_task, self.right_ee_task]  # TODO: add posture task
        self.limits = [mink.ConfigurationLimit(self.model), mink.VelocityLimit(self.model, velocity_limits)]

        # initial setup
        self.initalized_ = False

    def init(self, q):
        self.configuration.update(q)
        self.posture_task.set_target_from_configuration(self.configuration)
        self.initalized_ = True

    def get_home_q(self, home_key: str = "home") -> np.ndarray:
        return self.model.key(home_key).qpos[self.dof_ids]

    def solve_ik(self, T_wL: mink.SE3, T_wR: mink.SE3) -> tuple[np.ndarray, np.ndarray]:
        """
        Solves the IK problem for the left and right arms.
        Returns: (qL, qR)
        """
        if not self.initalized_:
            raise ValueError("IK solver not initialized")
        self.left_ee_task.set_target(T_wL)
        self.right_ee_task.set_target(T_wR)
        vel = mink.solve_ik(
            self.configuration, self.tasks, self.solver_dt, solver="daqp", damping=1e-3, limits=self.limits
        )
        self.configuration.integrate_inplace(vel, self.solver_dt)
        q = self.configuration.q[self.dof_ids]
        return q[:6], q[6:]

    def forward_kinematics(self) -> tuple[mink.SE3, mink.SE3]:
        # self.configuration.update(q)
        return (
            self.configuration.get_transform_frame_to_world(self.left_ee_task.frame_name, self.left_ee_task.frame_type),
            self.configuration.get_transform_frame_to_world(
                self.right_ee_task.frame_name, self.right_ee_task.frame_type
            ),
        )
