#include "nero_interface.h"
#include "solver.h"
#include "utils.h"
#include <signal.h>
#include <spdlog/spdlog.h>
#include <iostream>
#include <string>

static bool running = true;

void signal_handler(int signal) { running = false; }

void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [options]\n"
              << "Options:\n"
              << "  -i, --interface <name>  CAN interface name (default: can_left)\n"
              << "  -u, --urdf <path>       Path to URDF file (default: ../urdf/nero_cone-e_left.urdf)\n"
              << "  -g, --gripper           Enable gripper (default: false)\n"
              << "  -h, --help              Show this help message\n";
}

int main(int argc, char *argv[]) {
  signal(SIGINT, signal_handler);

  // Default configuration
  std::string interface_name = "can_left";
  std::string urdf_path = "/home/cone-e/code/nero_description/urdf/nero_description_left.urdf";
  bool gripper_on = false;

  // Parse arguments
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-h" || arg == "--help") {
        print_usage(argv[0]);
        return 0;
    } else if (arg == "-i" || arg == "--interface") {
        if (i + 1 < argc) {
            interface_name = argv[++i];
        } else {
            std::cerr << "Error: --interface requires an argument\n";
            return 1;
        }
    } else if (arg == "-u" || arg == "--urdf") {
        if (i + 1 < argc) {
            urdf_path = argv[++i];
        } else {
            std::cerr << "Error: --urdf requires an argument\n";
            return 1;
        }
    } else if (arg == "-g" || arg == "--gripper") {
        gripper_on = true;
    } else {
        std::cerr << "Unknown argument: " << arg << "\n";
        print_usage(argv[0]);
        return 1;
    }
  }

  InverseDynamicsSolver solver(urdf_path);
  NeroInterface nero_interface(interface_name, gripper_on);
  
  if (nero_interface.get_arm_status() == ArmStatus::EMERGENCY_STOP) {
    spdlog::error("Arm status is emergency stop, first resume emergency stop!");
    exit(1);
  }

  // reset arm
  reset_arm(nero_interface);
  if (gripper_on) {
    reset_gripper(nero_interface);
  }

  RateLimiter rate_limiter(200);

  JointState output_joint_state_;
  Gain grav_comp_gain =
      Gain({0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
  while (running) {
    JointState joint_state = nero_interface.get_current_state();
    std::array<double, MOTOR_DOF> joint_torque = solver.inverse_dynamics(
        joint_state.pos, joint_state.vel, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});

    output_joint_state_.pos = joint_state.pos;
    output_joint_state_.vel = joint_state.vel;
    output_joint_state_.torque = joint_torque;

    if (rate_limiter.get_iteration_count() % 100 == 0) {
      spdlog::info("Computed gravity comp torques: {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}",
                   joint_torque[0], joint_torque[1], joint_torque[2], joint_torque[3],
                   joint_torque[4], joint_torque[5], joint_torque[6]);
    }

    nero_interface.set_joint_pos_vel_torque(output_joint_state_, grav_comp_gain);
    rate_limiter.wait();
  }

  spdlog::info("Shutting down... Setting damping mode and disabling arm.");
  nero_interface.set_to_damping_mode();
  sleep_ms(100);

  nero_interface.disable_arm();
  if (gripper_on) {
    nero_interface.disable_gripper();
  }
  return 0;
}
