#ifndef INVERSE_DYNAMICS_SOLVER_H
#define INVERSE_DYNAMICS_SOLVER_H

#include "common.h"
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/parsers/urdf.hpp>

class InverseDynamicsSolver {
public:
  InverseDynamicsSolver(std::string urdf_path);
  ~InverseDynamicsSolver() = default;
  std::array<double, MOTOR_DOF>
  inverse_dynamics(const std::array<double, MOTOR_DOF> &joint_pos,
                   const std::array<double, MOTOR_DOF> &joint_vel,
                   const std::array<double, MOTOR_DOF> &joint_acc);

  // Pinocchio model and data
  pinocchio::Model model_;
  pinocchio::Data data_;
};

#endif // INVERSE_DYNAMICS_SOLVER_H