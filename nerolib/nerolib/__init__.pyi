from typing import overload

class ControllerConfig:
    interface_name: str
    urdf_path: str
    default_kp: list[float]
    default_kd: list[float]
    joint_vel_max: list[float]
    joint_acc_max: list[float]
    over_current_cnt_max: int
    controller_freq_hz: float
    gravity_compensation: bool
    gripper_on: bool

class Gain:
    kp: list[float]
    kd: list[float]
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(
        self,
        kp: list[float],
        kd: list[float],
    ) -> None: ...
    def __add__(self, other: Gain) -> Gain: ...
    def __mul__(self, scalar: float) -> Gain: ...

class JointState:
    timestamp: float
    pos: list[float]
    vel: list[float]
    torque: list[float]
    gripper_pos: float
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(
        self,
        pos: list[float],
        vel: list[float],
        torque: list[float],
        gripper_pos: float,
    ) -> None: ...
    def __add__(self, other: JointState) -> JointState: ...
    def __mul__(self, scalar: float) -> JointState: ...

class NeroController:
    def __init__(
        self,
        controller_config: ControllerConfig,
    ) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def reset_to_home(self) -> None: ...
    def is_running(self) -> bool: ...
    def get_current_state(self) -> JointState: ...
    def get_current_target(self) -> list[float]: ...
    def set_gain(self, gain: Gain) -> None: ...
    def set_target(
        self,
        new_target_pos: list[float],
        new_target_gripper_pos: float,
        minimum_duration: float = 0.0,
        new_target_vel: list[float] = [0.0] * 7,
        new_target_acc: list[float] = [0.0] * 7,
    ) -> None: ...
    def set_gripper(self, gripper_pos: float) -> None: ...
