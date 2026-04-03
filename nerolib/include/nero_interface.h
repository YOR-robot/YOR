#ifndef NERO_INTERFACE_H
#define NERO_INTERFACE_H

#include "common.h"
#include "socket_can.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <string>

#define KP_MIN 0.0f
#define KP_MAX 500.0f
#define KD_MIN -5.0f
#define KD_MAX 5.0f
#define POS_MIN -12.5f
#define POS_MAX 12.5f
#define VEL_MIN -45.0f
#define VEL_MAX 45.0f
#define T_MIN -25.0f
#define T_MAX 25.0f

#define GRIPPER_ANGLE_MAX 0.07f
#define GRIPPER_EFFORT_MAX 2.0f

union DriverStatus {
  uint8_t status;
  struct {
    bool voltage_too_low : 1;
    bool motor_overheating : 1;
    bool driver_overcurrent : 1;
    bool driver_overheating : 1;
    bool collision_status : 1;
    bool driver_error_status : 1;
    bool driver_enable_status : 1;
    bool stall_status : 1;
  };
  DriverStatus() : status(0) {};
  DriverStatus(uint8_t status) : status(status) {};
};

union GripperStatus {
  uint8_t status;
  struct {
    bool voltage_too_low : 1;
    bool motor_overheating : 1;
    bool driver_overcurrent : 1;
    bool driver_overheating : 1;
    bool sensor_status : 1;
    bool driver_error_status : 1;
    bool driver_enable_status : 1;
    bool homing_status : 1;
  };
  GripperStatus() : status(0) {};
  GripperStatus(uint8_t status) : status(status) {};
};

enum class ArmStatus : uint8_t {
  NORMAL = 0x00,
  EMERGENCY_STOP = 0x01,
  NO_SOLUTION = 0x02,
  SINGULARITY = 0x03,
  TARGET_ANGLE_EXCEEDS_LIMIT = 0x04,
  JOINT_COMMUNICATION_EXCEPTION = 0x05,
  JOINT_BRAKE_NOT_RELEASED = 0x06,
  COLLISION = 0x07,
  OVERSPEED_DURING_TEACHING = 0x08,
  JOINT_STATUS_ABNORMAL = 0x09,
  OTHER_EXCEPTION = 0x0A,
  TEACHING_RECORD = 0x0B,
  TEACHING_EXECUTION = 0x0C,
  TEACHING_PAUSE = 0x0D,
  MAIN_CONTROLLER_NTC_OVER_TEMPERATURE = 0x0E,
  RELEASE_RESISTOR_NTC_OVER_TEMPERATURE = 0x0F,
};

enum class TeachStatus : uint8_t {
  OFF = 0x00,
  START_RECORD = 0x01,
  END_RECORD = 0x02,
  EXECUTE = 0x03,
  PAUSE = 0x04,
  CONTINUE = 0x05,
  TERMINATE = 0x06,
  MOVE_TO_START = 0x07,
};

enum class MotionStatus : uint8_t {
  REACHED_TARGET = 0x00,
  NOT_YET_REACHED_TARGET = 0x01,
};

enum class GripperCode : uint8_t {
  DISABLE = 0x00,
  ENABLE = 0x01,
  DISABLE_AND_CLEAR_ERROR = 0x02,
  ENABLE_AND_CLEAR_ERROR = 0x03
};

enum class ControlMode : uint8_t {
  STANDBY = 0x00,
  CAN_COMMAND = 0x01,
  TEACH_MODE = 0x02,
  ETHERNET = 0x03,
  WIFI = 0x04,
  REMOTE = 0x05,
  LINKAGE_TEACHING = 0x06,
  OFFLINE_TRAJECTORY = 0x07
};

enum class MoveMode : uint8_t {
  POSITION = 0x00,
  JOINT = 0x01,
  LINEAR = 0x02,
  CIRCULAR = 0x03,
  MIT = 0x04,
};

enum class ArmController : uint8_t {
  POSITION_VELOCITY = 0x00,
  MIT = 0xAD,
  INVALID = 0xFF,
};

enum class EmergencyStop : uint8_t {
  INVALID = 0x00,
  STOP = 0x01,
  RESUME = 0x02
};

class NeroInterface {
public:
  NeroInterface(std::string interface_name, bool gripper_active = true);
  ~NeroInterface() { delete socketcan_; }

  void enable_arm();
  void enable_gripper();
  void disable_arm();
  void disable_gripper();

  void set_to_damping_mode();
  void set_emergency_stop(EmergencyStop emergency_stop);
  void set_arm_mode(ControlMode ctrl_mode, MoveMode move_mode,
                    uint8_t speed_rate, ArmController arm_controller);

  void set_joint_pos_vel_torque(const JointState &joint_state,
                                const Gain &gain);
  void set_gripper(float position, float effort,
                   GripperCode status_code = GripperCode::ENABLE);

  JointState get_current_state();
  DriverStatus get_driver_status(int motor_id) {
    return driver_status_[motor_id].load();
  };
  ArmStatus get_arm_status() { return arm_status_.load(); };
  ControlMode get_control_mode() { return control_mode_.load(); };
  GripperStatus get_gripper_status() { return gripper_status_.load(); };
  MoveMode get_move_mode() { return move_mode_.load(); };

  void standby(MoveMode move_mode, ArmController arm_controller);
  std::string get_nero_interface_name() { return interface_name_; }
  std::string get_nero_firmware_version() {
    return "1.0.0";
  } // TODO: get firmware version

  bool is_gripper_active() { return gripper_active_; }
  bool is_arm_enabled();
  bool is_gripper_enabled();

  void show_status();

private:
  // can communication
  std::string interface_name_;
  SocketCAN *socketcan_;
  bool can_connection_status_ = false;
  bool gripper_active_ = true;
  void can_receive_frame(const can_frame_t *frame);
  void transmit(can_frame_t &frame);

  // state management
  std::array<std::atomic<float>, MOTOR_DOF> qpos_;
  std::array<std::atomic<float>, MOTOR_DOF> qvel_;
  std::array<std::atomic<float>, MOTOR_DOF> tau_;
  std::array<std::atomic<DriverStatus>, MOTOR_DOF> driver_status_;
  std::atomic<ArmStatus> arm_status_;
  std::atomic<MoveMode> move_mode_;
  std::atomic<ControlMode> control_mode_;
  std::atomic<TeachStatus> teach_status_;
  std::atomic<MotionStatus> motion_status_;
  std::atomic<float> gripper_pos_;
  std::atomic<float> gripper_effort_;
  std::atomic<GripperStatus> gripper_status_;
};

void enable_arm(NeroInterface &nero_interface, float timeout_sec = 5.0f);
void disable_arm(NeroInterface &nero_interface, float timeout_sec = 5.0f);
void enable_gripper(NeroInterface &nero_interface, float timeout_sec = 5.0f);
void disable_gripper(NeroInterface &nero_interface, float timeout_sec = 5.0f);
void reset_arm(NeroInterface &nero_interface, float timeout_sec = 5.0f);
void reset_gripper(NeroInterface &nero_interface, float timeout_sec = 5.0f);

#endif
