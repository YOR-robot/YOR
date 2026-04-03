#include "controller_base.h"
#include "utils.h"

NeroController::NeroController(ControllerConfig controller_config)
    : controller_config_(controller_config),
      nero_interface_(controller_config.interface_name, controller_config.gripper_on),
      solver_(controller_config.urdf_path),
      otg_(1.0 / controller_config.controller_freq_hz),
      gain_(controller_config.default_kp, controller_config.default_kd),
      gravity_compensation_enabled_(controller_config.gravity_compensation) {
  start_time_us_ = get_time_us();
  target_position_ = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  target_velocity_ = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  target_acceleration_ = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  target_gripper_pos_ = 1.0f;
  minimum_duration_ = 0.0f;
}

NeroController::~NeroController() { stop(); }

void NeroController::resetToHome() {
  setTarget(
      controller_config_.home_position,
      1.0f,
      3.0f // minimum duration
  );

  // wait until homed
  while (true) {
    auto current_state = getCurrentState();
    bool is_close = true;
    for (size_t i = 0; i < MOTOR_DOF; ++i) {
      if (std::abs(current_state.pos[i] - controller_config_.home_position[i]) > 0.05) {
        if (is_close) { // Only log once per iteration if any joint is off
            spdlog::debug("Waiting for homing: Joint {} diff {}", i, current_state.pos[i] - controller_config_.home_position[i]);
        }
        is_close = false;
      }
    }

    if (is_close) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(200)); // Increase sleep to reduce log spam
  }
  spdlog::info("Arm reset to home successfully");
}

bool NeroController::start() {
  if (control_loop_running_.load()) {
    spdlog::warn("Controller already running");
    return false;
  }

  reset_arm(nero_interface_);
  reset_gripper(nero_interface_);

  // INITIALIZE TARGET TO CURRENT POSITION TO PREVENT HOMING GLITCH
  target_position_ = nero_interface_.get_current_state().pos;
  target_velocity_ = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  target_acceleration_ = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  target_gripper_pos_ = 1.0f;
  new_target_flag_.store(false);

  should_stop_.store(false);
  control_loop_running_.store(true);
  control_loop_thread_ = std::thread(&NeroController::controlLoop, this);

  return true;
}

void NeroController::stop() {
  if (!control_loop_running_.load()) {
    return;
  }

  should_stop_.store(true);

  if (control_loop_thread_.joinable()) {
    control_loop_thread_.join();
  }

  control_loop_running_.store(false);
  nero_interface_.set_to_damping_mode();
  spdlog::info("NeroController set to damping mode");

  disable_arm(nero_interface_);
  disable_gripper(nero_interface_);
}

void NeroController::setTarget(
    const std::array<double, MOTOR_DOF> &new_target_pos,
    const float new_target_gripper_pos,
    const float minimum_duration,
    const std::array<double, MOTOR_DOF> &new_target_vel,
    const std::array<double, MOTOR_DOF> &new_target_acc) {
  std::lock_guard<std::mutex> lock(target_mutex_);
  target_position_ = new_target_pos;
  target_velocity_ = new_target_vel;
  target_acceleration_ = new_target_acc;
  target_gripper_pos_ = new_target_gripper_pos;
  minimum_duration_ = minimum_duration;
  new_target_flag_.store(true);
  spdlog::debug("Target set to: [{}]", ::join(new_target_pos));
}

bool NeroController::isRunning() const { return control_loop_running_.load(); }

std::array<double, MOTOR_DOF> NeroController::getCurrentTarget() {
  std::lock_guard<std::mutex> lock(target_mutex_);
  return target_position_;
}

JointState NeroController::getCurrentState() {
  return nero_interface_.get_current_state();
}

void NeroController::setGain(const Gain &gain) {
  std::lock_guard<std::mutex> lock(target_mutex_);
  gain_ = gain;
  spdlog::info("Gain updated: kp[0]={:.2f}, kd[0]={:.2f}", gain.kp[0], gain.kd[0]);
}

void NeroController::enableGravityCompensation(bool enable) {
  gravity_compensation_enabled_.store(enable);
  spdlog::info("Gravity compensation {}", enable ? "enabled" : "disabled");
}

void NeroController::setGravityCompScale(float scale) {
  gravity_comp_scale_.store(scale);
  spdlog::info("Gravity compensation scale set to {:.2f}", scale);
}

void NeroController::setMode(ControlMode control_mode, MoveMode move_mode) {
  ArmController arm_controller = (move_mode == MoveMode::MIT) ? ArmController::MIT : ArmController::POSITION_VELOCITY;
  nero_interface_.set_arm_mode(control_mode, move_mode, 100, arm_controller);
  spdlog::info("Mode updated: ControlMode={}, MoveMode={}", (int)control_mode, (int)move_mode);
}

void NeroController::driverProtection() {
  bool over_current = false;
  for (int i = 0; i < MOTOR_DOF; ++i) {
    DriverStatus driver_status = nero_interface_.get_driver_status(i);
    if (driver_status.driver_overcurrent) {
      over_current = true;
      spdlog::warn("Over current detected once on joint {}", i);
      break;
    }
    if (driver_status.driver_error_status) {
      nero_interface_.set_emergency_stop(EmergencyStop::STOP);
      throw std::runtime_error("Driver error detected on joint " +
                               std::to_string(i) +
                               ". Please restart the program.");
    }
    if (driver_status.driver_overheating) {
      nero_interface_.set_emergency_stop(EmergencyStop::STOP);
      throw std::runtime_error("Driver overheating detected on joint " +
                               std::to_string(i) +
                               ". Please restart the program.");
    }
    if (driver_status.collision_status) {
      nero_interface_.set_emergency_stop(EmergencyStop::STOP);
      throw std::runtime_error("Collision detected on joint " +
                               std::to_string(i) +
                               ". Please restart the program.");
    }
    if (driver_status.stall_status) {
      nero_interface_.set_emergency_stop(EmergencyStop::STOP);
      throw std::runtime_error("Stall detected on joint " + std::to_string(i) +
                               ". Please restart the program.");
    }
  }

  if (over_current) {
    over_current_cnt_++;
    if (over_current_cnt_ > controller_config_.over_current_cnt_max) {
      nero_interface_.set_emergency_stop(EmergencyStop::STOP);
      throw std::runtime_error("Over current detected too many times. Please "
                               "restart the program.");
    }
  } else {
    over_current_cnt_ = 0;
  }
}

void NeroController::controlLoop() {
  spdlog::info("Starting Nero controller control loop at {}Hz",
               controller_config_.controller_freq_hz);

  ruckig::InputParameter<MOTOR_DOF> input;
  ruckig::OutputParameter<MOTOR_DOF> output;

  JointState initial_joint_state = nero_interface_.get_current_state();
  input.current_position = initial_joint_state.pos;
  input.current_velocity = initial_joint_state.vel;
  input.current_acceleration = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  {
    std::lock_guard<std::mutex> lock(target_mutex_);
    input.target_position = target_position_;
    input.target_velocity = target_velocity_;
    input.target_acceleration = target_acceleration_;
  }

  input.max_velocity = controller_config_.joint_vel_max;
  input.max_acceleration = controller_config_.joint_acc_max;

  JointState output_joint_cmd;

  RateLimiter rate_limiter(controller_config_.controller_freq_hz);
  while (!should_stop_.load()) {
    if (new_target_flag_.exchange(false)) {
      std::lock_guard<std::mutex> lock(target_mutex_);
      input.target_position = target_position_;
      input.target_velocity = target_velocity_;
      input.target_acceleration = target_acceleration_;
      input.minimum_duration = minimum_duration_;
      trajectory_active_.store(true);
      spdlog::debug("New target received: [{}]", ::join(input.target_position));
    }
    auto current_joint_state = nero_interface_.get_current_state();

    // Check if we're in compliant mode (all kp effectively zero)
    bool all_kp_zero = false;
    {
      std::lock_guard<std::mutex> lock(target_mutex_);
      all_kp_zero = true;
      for (int i = 0; i < MOTOR_DOF; ++i) {
        if (gain_.kp[i] > 0.001) { all_kp_zero = false; break; }
      }
    }

    if (all_kp_zero) {
      // COMPLIANT MODE: Skip trajectory entirely.
      // Track actual arm position so there is no spring-back.
      // Only apply gravity compensation torque + damping.
      trajectory_active_.store(false);
      new_target_flag_.store(false);

      std::array<double, MOTOR_DOF> gravity_compensation = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      if (gravity_compensation_enabled_.load()) {
        gravity_compensation = solver_.inverse_dynamics(current_joint_state.pos,
                                                       current_joint_state.vel,
                                                       {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
        float scale = gravity_comp_scale_.load();
        for (int i = 0; i < MOTOR_DOF; ++i) {
          gravity_compensation[i] *= scale;
        }
      }

      // Update Ruckig state and internal target to track the arm
      input.current_position = current_joint_state.pos;
      input.current_velocity = current_joint_state.vel;

      Gain current_gain;
      {
        std::lock_guard<std::mutex> lock(target_mutex_);
        target_position_ = current_joint_state.pos;
        target_velocity_ = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        output_joint_cmd.pos = current_joint_state.pos;
        output_joint_cmd.vel = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        output_joint_cmd.torque = gravity_compensation;
        current_gain = gain_;
      }

      nero_interface_.set_joint_pos_vel_torque(output_joint_cmd, current_gain);

    } else if (trajectory_active_.load()) {
      ruckig::Result result = otg_.update(input, output);

      if (result == ruckig::Result::Working) {
        output_joint_cmd.pos = output.new_position;
        output_joint_cmd.vel = output.new_velocity;

        Gain current_gain;
        if (gravity_compensation_enabled_.load()) {
          std::array<double, MOTOR_DOF> gravity_compensation =
              solver_.inverse_dynamics(current_joint_state.pos,
                                       current_joint_state.vel,
                                       {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
          float scale = gravity_comp_scale_.load();
          for(int i=0; i<MOTOR_DOF; ++i) {
            gravity_compensation[i] *= scale;
          }
          output_joint_cmd.torque = gravity_compensation;
        } else {
          output_joint_cmd.torque = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        }

        {
          std::lock_guard<std::mutex> lock(target_mutex_);
          current_gain = gain_;
        }
        nero_interface_.set_joint_pos_vel_torque(output_joint_cmd, current_gain);
        output.pass_to_input(input);
      } else if (result == ruckig::Result::Finished) {
        spdlog::debug("Trajectory completed");
        trajectory_active_.store(false);
      }
    } else {
      // No trajectory active, hold the last target position
      std::array<double, MOTOR_DOF> gravity_compensation = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      if (gravity_compensation_enabled_.load()) {
        gravity_compensation = solver_.inverse_dynamics(current_joint_state.pos,
                                                       current_joint_state.vel,
                                                       {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
      }
      
      // Update Ruckig input continuously with sensors so that a new 
      // trajectory starts from the actual physical state.
      input.current_position = current_joint_state.pos;
      input.current_velocity = current_joint_state.vel;

      Gain current_gain;
      {
        std::lock_guard<std::mutex> lock(target_mutex_);
        output_joint_cmd.pos = target_position_;
        output_joint_cmd.vel = target_velocity_;
        output_joint_cmd.torque = gravity_compensation;
        current_gain = gain_;
      }
      nero_interface_.set_joint_pos_vel_torque(output_joint_cmd, current_gain);
    }

    if (controller_config_.gripper_on) {
      nero_interface_.set_gripper(target_gripper_pos_, 0.1f, GripperCode::ENABLE);
    }

    // TODO: check if joint state is within limits
    driverProtection();
    rate_limiter.wait();
  }
  spdlog::info("Control loop stopped");
}
