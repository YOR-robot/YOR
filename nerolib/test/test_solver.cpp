#include "solver.h"
#include "utils.h"
#include <iostream>

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <urdf_file_path>" << std::endl;
    return 1;
  }

  std::string urdf_path = argv[1];
  InverseDynamicsSolver solver(urdf_path);
  std::array<double, MOTOR_DOF> joint_pos = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  std::array<double, MOTOR_DOF> joint_vel = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  std::array<double, MOTOR_DOF> joint_acc = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  std::array<double, MOTOR_DOF> joint_torque =
      solver.inverse_dynamics(joint_pos, joint_vel, joint_acc);
  std::cout << "Joint torque: " << ::join(joint_torque) << std::endl;
  return 0;
}