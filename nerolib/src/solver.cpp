#include "solver.h"

#include <Eigen/Core>

InverseDynamicsSolver::InverseDynamicsSolver(std::string urdf_path) {
  pinocchio::urdf::buildModel(urdf_path, model_);
  data_ = pinocchio::Data(model_);
}

// Eigen dependency is contained here
std::array<double, MOTOR_DOF> InverseDynamicsSolver::inverse_dynamics(
    const std::array<double, MOTOR_DOF> &joint_pos,
    const std::array<double, MOTOR_DOF> &joint_vel,
    const std::array<double, MOTOR_DOF> &joint_acc) {
  Eigen::VectorXd joint_pos_eigen(MOTOR_DOF);
  Eigen::VectorXd joint_vel_eigen(MOTOR_DOF);
  Eigen::VectorXd joint_acc_eigen(MOTOR_DOF);
  for (int i = 0; i < MOTOR_DOF; i++) {
    joint_pos_eigen[i] = joint_pos[i];
    joint_vel_eigen[i] = joint_vel[i];
    joint_acc_eigen[i] = joint_acc[i];
  }
  Eigen::VectorXd joint_torque_eigen = pinocchio::rnea(
      model_, data_, joint_pos_eigen, joint_vel_eigen, joint_acc_eigen);
  std::array<double, MOTOR_DOF> joint_torque;
  for (int i = 0; i < MOTOR_DOF; i++) {
    joint_torque[i] = joint_torque_eigen(i);
  }
  return joint_torque;
}
