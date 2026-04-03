# Nerolib

Nerolib is a library for AgileX Nero arms. It implements a minimal hardware interface layer and a joint controller. Nerolib is written purely in C++ (with Python bindings) with only Pinocchio dependency for gravity compensation and Ruckig for trajectory generation.

## Features

- [x] Simple interface with Nero without ROS and [nero_sdk]()
- [x] Joint stiffness controller implemented through the MIT control mode
- [x] Velocity, acceleration and jerk limits with [Ruckig]() on-the-go trajectory generation.
- [x] Python bindings
- [x] Gripper control
- [ ] nero-cli for easier state management

## Installation

Nerolib is a C++ library that has Python bindings. It has a few C++ dependencies that unfortunately cannot be handled only through `uv`.

### Pre-requisites:

C++ dependencies are installed using conda. However, conda is __ONLY__ used for these dependencies. Nerolib Python package uses the uv Python and venv.
uv is a superior tool and this makes Nerolib easier to use as a dependency in other projects. Follow these steps to install dependencies:

1. Install the conda environment: `mamba env create -f environment.yml`.
2. If your conda environment is not installed in the default path: `$HOME/miniforge3/envs/yor-nero`, set NEROLIB_CONDA_ENV environment variable.

### Installing Python library

1. You can now use pip to build and install the C++ library and Python bindings: `pip install -e .`
2. or you can do `uv sync` to create a virtual environment and install the package.

### Building C++ library

If you want to use the C++ library or test your developments in C++:

```bash
mkdir -p build && cd build
cmake ..
make -j
```


## Usage

You can use Nerolib directly in C++ or through its Python bindings. Here is a very simple example that moves the robot to a desired joint position:

```python
from nerolib import NeroController, ControllerConfig

controller_config = ControllerConfig()
controller_config.urdf_path = "" # set urdf path for gravity compensation, you can use the one in urdf/
nero_controller = NeroController(controller_config)

if not nero_controller.start(): # WARNING: this will power down the robot, and it will fall if not in neutral position.
    raise RuntimeError("nero controller could not be started")

nero_controller.set_target(
    new_target_pos=[0.0] * 6,
    new_target_vel=[0.0] * 6,
    new_target_acc=[0.0] * 6,
)

time.sleep(1)

nero_controller.stop() # WARNING: this puts the robot in damping mode, slowly lowering. Then, it prompts user to press Enter to disable motors after the arm is safe.
```

## Nero Interface

Nero arms use CAN as communication protocol. `nero_interface.h` defines the hardware interface with the robot:

- `reset_arm(NeroInterface &interface)` / `reset_gripper(NeroInterface &interface)` : Disables and enables arm/gripper until they are active again in a timeout loop.`NeroController.start` uses these.

- `enable_arm(NeroInterface &interface)` / `disable_arm(NeroInterface &interface)` : Used to enable/disable Nero actuators. Disable powers off the motors and the arms will fall down if not supported!!!

- `enable_gripper(NeroInterface &interface)` / `disable_gripper(NeroInterface &interface)` : Used to enable/disable Nero gripper.

- `NeroInterface::set_emergency_stop(EmergencyStop es)`: Emergency stop cancels the current command and puts the Nero actuators in a damping-only mode which allows the arm to slowly drop.

- `NeroInterface::set_arm_mode(ControlMode ctrl_mode, MoveMode move_mode, uint8_t speed_rate, ArmController arm_controller)`: Nero supports two control modes: MIT mode for low level PD control of actuators and Joint Control mode that implements a joint controller in the firmware. Since we implement a compliant joint stiffness controller in Nerolib, we use MIT mode. This call allows you to change between modes.

- `NeroInterface::set_joint_pos_vel_torque(JointState &joint_state, Gain& gain)`: Sends motor commands to all the actuators. Requires that the actuators are enabled and in MIT mode. You can command:
    * `desired_pos`: rad
    * `desired_vel`: rad/s
    * `feedforward_torque`: Nm
    * `kp`
    * `kd`
    The controller law here is: `K_p * (desired_pos - current_pos) + K_d * (desired_vel - current_vel) + feedforward_torque`

- `NeroInterface::get_current_state()`: It gets the current `JointState` from nero. 
- `NeroInterface::get_driver_status(int motor_id)`: `DriverStatus` shows error status, overcurrent, enabled/disabled for arm motors.
- `NeroInterface::get_arm_status()`: Check `ArmStatus` enum for possible states. Expected to be `NORMAL`.
- `NeroInterface::get_control_mode()`: Check `ControlMode` enum. Expected to be `STANDBY` or `CAN_COMMAND`
- `NeroInterface::get_gripper_status()`: `GripperStatus` shows gripper error and enabled status.
- `NeroInterface::get_move_mode()`: Check `MoveMode` enum. Expected to be `MIT` in MIT mode or `JOINT` in joint control mode.
- `NeroInterface::is_arm_enabled()`
- `NeroInterface::is_gripper_enabled()`
