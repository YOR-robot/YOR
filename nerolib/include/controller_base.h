#ifndef CONTROLLER_BASE_H
#define CONTROLLER_BASE_H
#include "common.h"
#include "config.h"
#include "nero_interface.h"
#include "solver.h"

#include <ruckig/ruckig.hpp>
#include <spdlog/spdlog.h>
#include <thread>

class NeroController {
public:
  NeroController(ControllerConfig controller_config);
  ~NeroController();

  void resetToHome();
  bool start();
  void stop();
  bool isRunning() const;
  JointState getCurrentState();
  std::array<double, MOTOR_DOF> getCurrentTarget();
  void setGain(const Gain &gain);
  void setTarget(
      const std::array<double, MOTOR_DOF> &new_target_pos,
      const float new_target_gripper_pos,
      const float minimum_duration = 0.0f,
      const std::array<double, MOTOR_DOF> &new_target_vel = {0.0, 0.0, 0.0, 0.0,
                                                             0.0, 0.0, 0.0},
      const std::array<double, MOTOR_DOF> &new_target_acc = {0.0, 0.0, 0.0, 0.0,
                                                             0.0, 0.0, 0.0});

  void enableGravityCompensation(bool enable);
  void setGravityCompScale(float scale);
  void setMode(ControlMode control_mode, MoveMode move_mode);

private:
  const ControllerConfig controller_config_;
  ruckig::Ruckig<MOTOR_DOF> otg_;
  InverseDynamicsSolver solver_;

  Gain gain_;
  std::array<double, MOTOR_DOF> target_position_;
  std::array<double, MOTOR_DOF> target_velocity_;
  std::array<double, MOTOR_DOF> target_acceleration_;
  float target_gripper_pos_;
  float minimum_duration_;

  NeroInterface nero_interface_;
  std::thread control_loop_thread_;
  std::mutex target_mutex_;
  std::atomic<bool> control_loop_running_{false};
  std::atomic<bool> should_stop_{false};
  std::atomic<bool> new_target_flag_{false};
  std::atomic<bool> trajectory_active_{false};
  std::atomic<bool> gravity_compensation_enabled_{false};
  std::atomic<float> gravity_comp_scale_{1.0f};

  std::chrono::microseconds start_time_us_;
  int over_current_cnt_ = 0;

  void controlLoop();
  void checkJointStateSanity(const JointState &joint_state);
  void driverProtection();
};

#endif
