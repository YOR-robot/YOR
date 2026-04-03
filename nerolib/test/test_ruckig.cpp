#include "common.h"
#include "nero_interface.h"
#include "utils.h"

#include "spdlog/spdlog.h"
#include <ruckig/ruckig.hpp>

using namespace ruckig;

void run_ruckig(NeroInterface &nero_interface, Ruckig<MOTOR_DOF> &otg,
                int control_freq) {
  // Create nero interface
  if (nero_interface.get_arm_status() == ArmStatus::EMERGENCY_STOP) {
    spdlog::warn("Arm status is emergency stop, first resume emergency stop!");
    return;
  } else {
    nero_interface.enable_arm();
    nero_interface.set_arm_mode(ControlMode::CAN_COMMAND, MoveMode::MIT, 0, ArmController::MIT);
  }

  // Create instances: the Ruckig OTG as well as input and output parameters
  InputParameter<MOTOR_DOF> input;
  OutputParameter<MOTOR_DOF> output;

  JointState initial_joint_state = nero_interface.get_current_state();
  // Set input parameters
  input.current_position = initial_joint_state.pos;
  input.current_velocity = initial_joint_state.vel;

  input.current_acceleration = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  input.target_position = {0.2, 0.2, -0.2, 0.3, -0.2, 0.5};
  input.target_velocity = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  input.target_acceleration = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  input.max_velocity = {3.0, 3.0, 3.0, 3.0, 3.0, 3.0};
  input.max_acceleration = {3.0, 3.0, 3.0, 3.0, 3.0, 3.0};
  // input.max_jerk = {4.0, 4.0, 4.0, 4.0, 4.0, 4.0};

  // input.minimum_duration = 1.0;

  std::cout << "Input position: " << ::join(input.current_position)
            << std::endl;
  std::cout << "Input velocity: " << ::join(input.current_velocity)
            << std::endl;
  std::cout << "Input acceleration: " << ::join(input.current_acceleration)
            << std::endl;
  std::cout << "Input target position: " << ::join(input.target_position)
            << std::endl;
  std::cout << "Input target velocity: " << ::join(input.target_velocity)
            << std::endl;

  // return;

  RateLimiter rate_limiter(control_freq);
  JointState output_joint_state;
  Gain gain({10.0, 10.0, 10.0, 10.0, 10.0, 10.0},
            {0.8, 0.8, 0.8, 0.8, 0.8, 0.8});

  while (otg.update(input, output) == Result::Working) {
    output_joint_state.timestamp = output.time;
    output_joint_state.pos = output.new_position;
    output_joint_state.vel = output.new_velocity;
    output_joint_state.torque = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    output.pass_to_input(input);
    nero_interface.set_joint_pos_vel_torque(output_joint_state, gain);
    rate_limiter.wait();
  }

  sleep_ms(5000);

  input.target_position = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  input.target_velocity = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  input.target_acceleration = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  // input.minimum_duration = 1.0;

  while (otg.update(input, output) == Result::Working) {
    output_joint_state.timestamp = output.time;
    output_joint_state.pos = output.new_position;
    output_joint_state.vel = output.new_velocity;
    output.pass_to_input(input);
    nero_interface.set_joint_pos_vel_torque(output_joint_state, gain);
    rate_limiter.wait();
  }
}

int main() {
  int control_freq = 200;
  NeroInterface nero_interface("can0");
  Ruckig<MOTOR_DOF> otg(1.0 / control_freq); // control cycle
  try {
    run_ruckig(nero_interface, otg, control_freq);
  } catch (const std::exception &e) {
    spdlog::error("Error: {}", e.what());
    return 1;
  }

  nero_interface.set_to_damping_mode();

  spdlog::info("Press Enter to disable arm...");
  std::cin.get();
  nero_interface.disable_arm();
  nero_interface.disable_gripper();
  return 0;
}