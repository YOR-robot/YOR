#ifndef CONFIG_H
#define CONFIG_H

#include "common.h"

class ControllerConfig {
public:
  std::string interface_name;
  std::string urdf_path;
  std::array<double, MOTOR_DOF> default_kp;
  std::array<double, MOTOR_DOF> default_kd;
  std::array<double, MOTOR_DOF> joint_vel_max;
  std::array<double, MOTOR_DOF> joint_acc_max;
  std::array<double, MOTOR_DOF> home_position;
  int over_current_cnt_max;
  double controller_freq_hz;
  bool gravity_compensation;
  float gravity_comp_scale;
  bool gripper_on;

  ControllerConfig(
      std::string interface_name = "can0",
      std::string urdf_path = "../urdf/nero_description.urdf",
      std::array<double, MOTOR_DOF> default_kp = {10.0, 10.0, 10.0, 10.0, 10.0,
                                                   10.0, 10.0},
      // kd=0.8 matches pyAgxArm reference value for nero motors
      std::array<double, MOTOR_DOF> default_kd = {0.8, 0.8, 0.8, 0.8, 0.8, 0.8,
                                                   0.8},
      std::array<double, MOTOR_DOF> joint_vel_max = {1.5, 1.5, 1.5, 1.5,
                                                     1.5, 1.5, 1.5},
      std::array<double, MOTOR_DOF> joint_acc_max = {2.0, 2.0, 2.0, 2.0, 2.0,
                                                     2.0, 2.0},
      std::array<double, MOTOR_DOF> home_position = {0.0, 0.0, 0.0, 0.0, 0.0,
                                                     0.0, 0.0},
      int over_current_cnt_max = 20, double controller_freq_hz = 250.0,
      bool gravity_compensation = true, float gravity_comp_scale = 1.0f, bool gripper_on = false)
      : interface_name(interface_name), urdf_path(urdf_path),
        default_kp(default_kp), default_kd(default_kd),
        joint_vel_max(joint_vel_max), joint_acc_max(joint_acc_max),
        home_position(home_position),
        over_current_cnt_max(over_current_cnt_max),
        controller_freq_hz(controller_freq_hz),
        gravity_compensation(gravity_compensation), gravity_comp_scale(gravity_comp_scale), gripper_on(gripper_on) {}
};

#endif // CONFIG_H
