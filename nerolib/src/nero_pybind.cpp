#include "common.h"
#include "config.h"
#include "controller_base.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(_nerolib, m) {
  py::class_<JointState>(m, "JointState")
      .def(py::init<>())
      .def(
          py::init<std::array<double, MOTOR_DOF>, std::array<double, MOTOR_DOF>,
                   std::array<double, MOTOR_DOF>, double>())
      .def_readwrite("timestamp", &JointState::timestamp)
      .def_readwrite("pos", &JointState::pos)
      .def_readwrite("vel", &JointState::vel)
      .def_readwrite("torque", &JointState::torque)
      .def_readwrite("gripper_pos", &JointState::gripper_pos)
      .def("__add__", [](const JointState &self,
                         const JointState &other) { return self + other; })
      .def("__mul__", [](const JointState &self, const float &scalar) {
        return self * scalar;
      });
  py::class_<Gain>(m, "Gain")
      .def(py::init<>())
      .def(py::init<std::array<double, MOTOR_DOF>,
                    std::array<double, MOTOR_DOF>>())
      .def_readwrite("kp", &Gain::kp)
      .def_readwrite("kd", &Gain::kd)
      .def("__add__",
           [](const Gain &self, const Gain &other) { return self + other; })
      .def("__mul__",
           [](const Gain &self, const float &scalar) { return self * scalar; });
  py::enum_<ControlMode>(m, "ControlMode")
      .value("STANDBY", ControlMode::STANDBY)
      .value("CAN_COMMAND", ControlMode::CAN_COMMAND)
      .value("TEACH_MODE", ControlMode::TEACH_MODE)
      .value("ETHERNET", ControlMode::ETHERNET)
      .value("WIFI", ControlMode::WIFI)
      .value("REMOTE", ControlMode::REMOTE)
      .value("LINKAGE_TEACHING", ControlMode::LINKAGE_TEACHING)
      .value("OFFLINE_TRAJECTORY", ControlMode::OFFLINE_TRAJECTORY)
      .export_values();

  py::enum_<MoveMode>(m, "MoveMode")
      .value("POSITION", MoveMode::POSITION)
      .value("JOINT", MoveMode::JOINT)
      .value("LINEAR", MoveMode::LINEAR)
      .value("CIRCULAR", MoveMode::CIRCULAR)
      .value("MIT", MoveMode::MIT)
      .export_values();

  py::class_<NeroController>(m, "NeroController")
      .def(py::init<ControllerConfig>())
      .def("reset_to_home", &NeroController::resetToHome)
      .def("start", &NeroController::start)
      .def("stop", &NeroController::stop)
      .def("is_running", &NeroController::isRunning)
      .def("get_current_state", &NeroController::getCurrentState)
      .def("get_current_target", &NeroController::getCurrentTarget)
      .def("set_gain", &NeroController::setGain, py::arg("gain"))
      .def("enable_gravity_compensation", &NeroController::enableGravityCompensation, py::arg("enable"))
      .def("set_gravity_comp_scale", &NeroController::setGravityCompScale, py::arg("scale"))
      .def("set_mode", &NeroController::setMode, py::arg("control_mode"), py::arg("move_mode"))
      .def("set_target", &NeroController::setTarget,
           py::arg("new_target_pos"),
           py::arg("new_target_gripper_pos"),
           py::arg("minimum_duration") = 0.0f,
           py::arg("new_target_vel") =
               std::array<double, MOTOR_DOF>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
           py::arg("new_target_acc") =
               std::array<double, MOTOR_DOF>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
  py::class_<ControllerConfig>(m, "ControllerConfig")
      .def(py::init<>())
      .def_readwrite("interface_name", &ControllerConfig::interface_name)
      .def_readwrite("urdf_path", &ControllerConfig::urdf_path)
      .def_readwrite("default_kp", &ControllerConfig::default_kp)
      .def_readwrite("default_kd", &ControllerConfig::default_kd)
      .def_readwrite("joint_vel_max", &ControllerConfig::joint_vel_max)
      .def_readwrite("joint_acc_max", &ControllerConfig::joint_acc_max)
      .def_readwrite("home_position", &ControllerConfig::home_position)
      .def_readwrite("over_current_cnt_max",
                     &ControllerConfig::over_current_cnt_max)
      .def_readwrite("controller_freq_hz",
                     &ControllerConfig::controller_freq_hz)
      .def_readwrite("gravity_compensation",
                     &ControllerConfig::gravity_compensation)
      .def_readwrite("gravity_comp_scale", &ControllerConfig::gravity_comp_scale)
      .def_readwrite("gripper_on", &ControllerConfig::gripper_on);
}
