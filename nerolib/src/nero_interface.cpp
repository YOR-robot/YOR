#include "nero_interface.h"
#include "common.h"
#include "spdlog/spdlog.h"
#include <cstdint>
#include <stdexcept>

#define COMMUNICATION_DELAY 10 // 10us

/********************************
        Helper functions
********************************/

inline uint8_t can_data_to_uint8_t(const uint8_t *data) {
  uint8_t value;
  std::memcpy(&value, data, sizeof(uint8_t));
  return value;
}
inline int16_t can_data_to_int16_t(const uint8_t *data) {
  int16_t value;
  std::memcpy(&value, data, sizeof(int16_t));
  return be16toh(value);
}

inline uint16_t can_data_to_uint16_t(const uint8_t *data) {
  uint16_t value;
  std::memcpy(&value, data, sizeof(uint16_t));
  return be16toh(value);
}

inline uint64_t can_data_to_uint64_t(const uint8_t *data) {
  uint64_t value;
  std::memcpy(&value, data, sizeof(uint64_t));
  return be64toh(value);
}

inline int32_t can_data_to_int32_t(const uint8_t *data) {
  int32_t value;
  std::memcpy(&value, data, sizeof(int32_t));
  return be32toh(value);
}

inline int float_to_int(const float val, const float min, const float max,
                        const int bits) {
  float span = max - min;
  return static_cast<int>((val - min) * ((1 << bits) - 1) / span);
}

/**
 * Helper functions
 */

void disable_arm(NeroInterface &nero_interface, float timeout_sec) {
  // disable arm first
  auto start_time = get_time_ms();
  while (true) {
    nero_interface.disable_arm();
    nero_interface.set_emergency_stop(EmergencyStop::RESUME);
    sleep_ms(100);
    ControlMode cm = nero_interface.get_control_mode();
    ArmStatus as = nero_interface.get_arm_status();
    bool is_enabled = nero_interface.is_arm_enabled();
    spdlog::info("disable_arm poll: ctrl_mode={} arm_status={} is_enabled={}",
                 static_cast<uint8_t>(cm), static_cast<uint8_t>(as), is_enabled);
    
    // Success if we are in STANDBY or if all motors are clearly disabled
    if ((cm == ControlMode::STANDBY && as == ArmStatus::NORMAL) || !is_enabled) {
      break;
    }
    if ((get_time_ms() - start_time).count() > timeout_sec * 1000) {
      spdlog::error("Failed to disable arm: ctrl_mode={} arm_status={}",
                    static_cast<uint8_t>(cm), static_cast<uint8_t>(as));
      throw std::runtime_error("Failed to disable arm");
    }
    sleep_ms(200);
  }
}

void enable_arm(NeroInterface &nero_interface, float timeout_sec) {
  // enable arm
  auto start_time = get_time_ms();
  while (true) {
    nero_interface.enable_arm();
    sleep_ms(100);
    if (nero_interface.is_arm_enabled()) {
      break;
    }
    if ((get_time_ms() - start_time).count() > timeout_sec * 1000) {
      spdlog::error("Failed to enable arm");
      throw std::runtime_error("Failed to enable arm");
    }
    sleep_ms(200);
  }

  // set arm mode
  nero_interface.set_arm_mode(ControlMode::CAN_COMMAND, MoveMode::MIT, 100,
                              ArmController::MIT);
  sleep_ms(100);

  if (nero_interface.get_arm_status() != ArmStatus::NORMAL ||
      nero_interface.get_control_mode() != ControlMode::CAN_COMMAND ||
      nero_interface.get_move_mode() != MoveMode::MIT) {
    spdlog::error("Failed to set arm mode");
    throw std::runtime_error("Failed to set arm mode");
  }
}

void reset_arm(NeroInterface &nero_interface, float timeout_sec) {
  disable_arm(nero_interface, timeout_sec);
  enable_arm(nero_interface, timeout_sec);
  spdlog::info("Arm reset successfully");
}

void disable_gripper(NeroInterface &nero_interface, float timeout_sec) {
  if (!nero_interface.is_gripper_active()) {
    spdlog::warn("Gripper is not active, skipping gripper disable");
    return;
  }
  auto start_time = get_time_ms();
  // disable gripper
  while (true) {
    nero_interface.disable_gripper();
    sleep_ms(100);
    if (!nero_interface.is_gripper_enabled()) {
      break;
    }
    if ((get_time_ms() - start_time).count() > timeout_sec * 1000) {
      spdlog::error("Failed to disable gripper");
      throw std::runtime_error("Failed to disable gripper");
    }
    sleep_ms(200);
  }
  spdlog::info("Gripper disabled successfully");
}

void enable_gripper(NeroInterface &nero_interface, float timeout_sec) {
  if (!nero_interface.is_gripper_active()) {
    spdlog::warn("Gripper is not active, skipping gripper enable");
    return;
  }
  auto start_time = get_time_ms();
  // enable gripper
  while (true) {
    nero_interface.enable_gripper();
    sleep_ms(100);
    if (nero_interface.is_gripper_enabled()) {
      break;
    }
    if ((get_time_ms() - start_time).count() > timeout_sec * 1000) {
      spdlog::error("Failed to enable gripper");
      throw std::runtime_error("Failed to enable gripper");
    }
    sleep_ms(200);
  }
  spdlog::info("Gripper enabled successfully");
}

void reset_gripper(NeroInterface &nero_interface, float timeout_sec) {
  disable_gripper(nero_interface, timeout_sec);
  enable_gripper(nero_interface, timeout_sec);
  spdlog::info("Gripper reset successfully");
}

/********************************
        NeroInterface
********************************/

NeroInterface::NeroInterface(std::string interface_name, bool gripper_active)
    : control_mode_(ControlMode::STANDBY), arm_status_(ArmStatus::NORMAL) {
  socketcan_ = new SocketCAN();
  socketcan_->open(interface_name.c_str());
  socketcan_->start_receiver_thread();
  socketcan_->reception_handler = [this](can_frame_t *frame) {
    can_receive_frame(frame);
  };
  sleep_ms(100);
  if (!can_connection_status_) {
    spdlog::warn("Failed to connect to Nero arm. Check the connection and power supply. Proceeding anyway...");
    // throw std::runtime_error("Failed to connect to Nero arm");
  }
  this->gripper_active_ = gripper_active;
}

void NeroInterface::transmit(can_frame_t &frame) {
  if (socketcan_->is_open()) {
    frame.can_dlc = 8;
    socketcan_->transmit(&frame);
  } else {
    spdlog::error("Cannot transmit CAN frame: CAN bus not open");
  }
}

void NeroInterface::can_receive_frame(const can_frame_t *frame) {
  can_connection_status_ = true;
  if (frame->can_id >= 0x251 &&
      frame->can_id <= 0x257) { // high speed feedback (7 motors)
    int motor_id = frame->can_id - 0x251;
    qvel_[motor_id].store(can_data_to_int16_t(frame->data) / 1000.0f);
    int16_t cur_raw = can_data_to_int16_t(frame->data + 2);
    float torque = 0.0f;
    if (cur_raw != static_cast<int16_t>(0x7FFF) &&
        cur_raw != static_cast<int16_t>(0x8000)) {
      // All motors report current in 0.001A units (per Nero CAN protocol spec)
      float current_a = cur_raw / 1000.0f;
      // pyAgxArm sets Kt = 1.18125 for ALL 7 Nero motors.
      float kt = 1.18125f;
      torque = current_a * kt;
    }
    tau_[motor_id].store(torque);
    qpos_[motor_id].store(can_data_to_int32_t(frame->data + 4) / 1000.0f);
  } else if (frame->can_id >= 0x261 &&
             frame->can_id <= 0x267) { // low speed feedback (7 motors)
    int motor_id = frame->can_id - 0x261;
    driver_status_[motor_id].store(DriverStatus(frame->data[5]));
  } else if (frame->can_id == 0x2A1) {
    control_mode_.store(ControlMode(can_data_to_uint8_t(frame->data)));
    arm_status_.store(ArmStatus(can_data_to_uint8_t(frame->data + 1)));
    move_mode_.store(MoveMode(can_data_to_uint8_t(frame->data + 2)));
  } else if (frame->can_id == 0x2A8) {
    gripper_pos_.store(can_data_to_int32_t(frame->data) /
                       (GRIPPER_ANGLE_MAX * 1000.0f * 1000.0f));
    gripper_effort_.store(can_data_to_uint16_t(frame->data + 4) / 1000.0f);
    gripper_status_.store(GripperStatus(can_data_to_uint8_t(frame->data + 6)));
  }
}

// Public functions

bool NeroInterface::is_arm_enabled() {
  for (int i = 0; i < MOTOR_DOF; i++) {
    if (!driver_status_[i].load().driver_enable_status) {
      return false;
    }
  }
  return true;
}

bool NeroInterface::is_gripper_enabled() {
  if (!gripper_active_) {
    return true; // if gripper is not active, consider it as enabled
  }
  return gripper_status_.load().driver_enable_status;
}

void NeroInterface::enable_arm() {
  // Step 1: Switch to CAN_COMMAND mode with enable_can_push=0x01 so the arm
  // starts broadcasting low-speed feedback (0x261-0x267 driver status packets).
  // Without this, is_arm_enabled() has no status bits to read.
  set_arm_mode(ControlMode::CAN_COMMAND, MoveMode::JOINT, 50,
               ArmController::POSITION_VELOCITY);
  sleep_ms(300);

  // Step 2: Send motor enable command via CAN 0x471:
  //   Byte 0 = 0x08 (all arm joints: joint index 8 = all, per nero protocol)
  //   Byte 1 = 0x02 (enable flag)
  can_frame_t frame;
  frame.can_id = 0x471;
  frame.data[0] = 0x08; // all arm joints (nero: 8 = all joints)
  frame.data[1] = 0x02; // enable
  std::fill(std::begin(frame.data) + 2, std::end(frame.data), 0);
  transmit(frame);
  sleep_ms(600); // wait for low-speed feedback to reflect new enable status
}

void NeroInterface::disable_arm() {
  // Send motor disable command via CAN 0x471:
  //   Byte 0 = 0x08 (all arm joints)
  //   Byte 1 = 0x01 (disable flag)
  can_frame_t frame;
  frame.can_id = 0x471;
  frame.data[0] = 0x08; // all arm joints
  frame.data[1] = 0x01; // disable
  std::fill(std::begin(frame.data) + 2, std::end(frame.data), 0);
  transmit(frame);
  sleep_ms(200);
  // Also switch to STANDBY mode so listeners polling get_control_mode() see STANDBY
  set_arm_mode(ControlMode::STANDBY, MoveMode::POSITION, 0, ArmController::POSITION_VELOCITY);
  sleep_ms(400); // wait for low-speed feedback to reflect new enable status
}

void NeroInterface::enable_gripper() {
  if (!gripper_active_) {
    spdlog::warn("Gripper is not active, cannot enable gripper");
    return;
  }
  set_gripper(0.0f, 0.0f, GripperCode::ENABLE); // enable
  sleep_ms(200);
}

void NeroInterface::disable_gripper() {
  if (!gripper_active_) {
    spdlog::warn("Gripper is not active, cannot disable gripper");
    return;
  }
  set_gripper(0.0f, 0.0f,
              GripperCode::DISABLE_AND_CLEAR_ERROR); // disable and clear error
  sleep_ms(200);
}

void NeroInterface::set_emergency_stop(EmergencyStop emergency_stop) {
  can_frame_t frame;
  frame.can_id = 0x150;
  frame.data[0] = static_cast<uint8_t>(emergency_stop);
  std::fill(std::begin(frame.data) + 1, std::end(frame.data), 0);
  transmit(frame);
  sleep_ms(200);
}

/*
 * This function is intended to be called once to switch control modes.
 * It has a sleep, don't call in main control loop.
 */
void NeroInterface::set_arm_mode(ControlMode ctrl_mode, MoveMode move_mode,
                                 uint8_t speed_rate,
                                 ArmController arm_controller) {
  can_frame_t frame;
  frame.can_id = 0x151;
  frame.data[0] = static_cast<uint8_t>(ctrl_mode);
  frame.data[1] = static_cast<uint8_t>(move_mode);
  frame.data[2] = speed_rate;
  frame.data[3] = static_cast<uint8_t>(arm_controller);
  frame.data[4] = 0x00; // offline trajectory hold time
  frame.data[5] = 0x00; // installation position (invalid)
  // Byte 6: enable_can_push — 0x01 = enable feedback, 0x02 = disable, 0x00 = no-change
  // Send 0x01 when activating CAN control so arm starts broadcasting
  frame.data[6] = (ctrl_mode == ControlMode::CAN_COMMAND) ? 0x01 : 0x02;
  frame.data[7] = 0x00;
  frame.can_dlc = 8;
  transmit(frame);
  sleep_ms(200);
}

void NeroInterface::set_to_damping_mode() {
  JointState joint_state;
  Gain gain;
  gain.kp = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  gain.kd = {0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8};

  for (int i = 0; i < 10; i++) {
    set_joint_pos_vel_torque(joint_state, gain);
    sleep_ms(10);
  }
}

/*
 * This code first converts the float values to integers of varying bit
 * widths, and packs them into a CAN frame for each motor in the array.
 * pos (16-bit)|spd (12-bit)|kp (12-bit)|kd (12-bit)|t (8-bit)|crc (4-bit)
 * crc is calculated by XORing the first 7 bytes of the CAN frame. A simple
 * checksum.
 */
void NeroInterface::set_joint_pos_vel_torque(const JointState &joint_state,
                                             const Gain &gain) {
  if (get_move_mode() != MoveMode::MIT) {
    spdlog::error("Cannot call set_joint_pos_vel_torque in non-MIT mode!");
    return;
  }

  for (uint8_t motor_id = 0; motor_id < MOTOR_DOF; ++motor_id) {
    // Check bounds for all parameters
    if (joint_state.pos[motor_id] < POS_MIN ||
        joint_state.pos[motor_id] > POS_MAX) {
      spdlog::error("Warning: Position {} out of bounds [{}, {}] for motor {}",
                    joint_state.pos[motor_id], POS_MIN, POS_MAX, motor_id);
      continue;
    }
    if (joint_state.vel[motor_id] < VEL_MIN ||
        joint_state.vel[motor_id] > VEL_MAX) {
      spdlog::error("Warning: Velocity {} out of bounds [{}, {}] for motor {}",
                    joint_state.vel[motor_id], VEL_MIN, VEL_MAX, motor_id);
      continue;
    }
    if (gain.kp[motor_id] < KP_MIN || gain.kp[motor_id] > KP_MAX) {
      spdlog::error("Warning: Kp {} out of bounds [{}, {}] for motor {}",
                    gain.kp[motor_id], KP_MIN, KP_MAX, motor_id);
      continue;
    }
    if (gain.kd[motor_id] < KD_MIN || gain.kd[motor_id] > KD_MAX) {
      spdlog::error("Warning: Kd {} out of bounds [{}, {}] for motor {}",
                    gain.kd[motor_id], KD_MIN, KD_MAX, motor_id);
      continue;
    }
    // t_ff expects torque in Nm directly, no conversion to Amperes needed.
    float torque_scaled = joint_state.torque[motor_id];
    // Clamp torque to valid range instead of skipping the command entirely.
    // Skipping leaves the motor uncontrolled (zero torque), which is dangerous
    // for gravity compensation where partial torque is better than none.
    if (torque_scaled < T_MIN || torque_scaled > T_MAX) {
      spdlog::warn("Torque {:.2f} (scaled {:.2f}) clamped to [{}, {}] for motor {}",
                    joint_state.torque[motor_id], torque_scaled, T_MIN, T_MAX, motor_id);
      torque_scaled = std::max(T_MIN, std::min(T_MAX, torque_scaled));
    }

    can_frame_t frame;
    frame.can_id = 0x15A + motor_id;

    int pos_ref = float_to_int(joint_state.pos[motor_id], POS_MIN, POS_MAX, 16);
    int spd_ref = float_to_int(joint_state.vel[motor_id], VEL_MIN, VEL_MAX, 12);
    int kp_int = float_to_int(gain.kp[motor_id], KP_MIN, KP_MAX, 12);
    int kd_int = float_to_int(gain.kd[motor_id], KD_MIN, KD_MAX, 12);
    int t_ref =
        float_to_int(torque_scaled, T_MIN, T_MAX, 8);

    frame.data[0] = (pos_ref >> 8) & 0xFF; // High byte
    frame.data[1] = pos_ref & 0xFF;        // Low byte
    frame.data[2] = (spd_ref >> 4) & 0xFF;
    frame.data[3] = (((spd_ref & 0xF) << 4) & 0xF0) | ((kp_int >> 8) & 0x0F);
    frame.data[4] = kp_int & 0xFF;
    frame.data[5] = (kd_int >> 4) & 0xFF;
    frame.data[6] = (((kd_int & 0xF) << 4) & 0xF0) | ((t_ref >> 4) & 0x0F);

    uint8_t crc =
        (frame.data[0] ^ frame.data[1] ^ frame.data[2] ^ frame.data[3] ^
         frame.data[4] ^ frame.data[5] ^ frame.data[6]) &
        0x0F;
    frame.data[7] = ((t_ref << 4) & 0xF0) | crc;

    transmit(frame);
    sleep_us(COMMUNICATION_DELAY);
  }
}

// Even when the status_code is disable, the gripper moves to the position.
void NeroInterface::set_gripper(float position, float effort,
                                GripperCode status_code) {
  if (!gripper_active_) {
    return; // if gripper is not active, do nothing
  }
  int32_t pos_ref = static_cast<int32_t>((position * 0.07) * 1000 *
                                         1000); // 0-1 to m to mm to micrometer
  uint16_t effort_ref =
      static_cast<uint16_t>(effort * 5000); // 0-1 to Nm to mNm

  can_frame_t frame;
  frame.can_id = 0x159;
  frame.data[0] = (pos_ref >> 24) & 0xFF;
  frame.data[1] = (pos_ref >> 16) & 0xFF;
  frame.data[2] = (pos_ref >> 8) & 0xFF;
  frame.data[3] = pos_ref & 0xFF;
  frame.data[4] = (effort_ref >> 8) & 0xFF;
  frame.data[5] = effort_ref & 0xFF;
  frame.data[6] = static_cast<uint8_t>(status_code);
  frame.data[7] = 0;
  transmit(frame);
  sleep_us(COMMUNICATION_DELAY);
}

JointState NeroInterface::get_current_state() {
  JointState state;
  state.timestamp = get_time_ms().count();
  for (int i = 0; i < MOTOR_DOF; i++) {
    state.pos[i] = qpos_[i].load();
    state.vel[i] = qvel_[i].load();
    state.torque[i] = tau_[i].load();
  }
  state.gripper_pos = gripper_pos_.load();
  state.gripper_effort = gripper_effort_.load();
  return state;
}
