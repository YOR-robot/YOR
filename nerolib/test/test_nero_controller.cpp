#include "controller_base.h"
#include "spdlog/spdlog.h"
#include <iostream>
#include <string>

void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [options]\n"
              << "Options:\n"
              << "  -i, --interface <name>  CAN interface name (default: can_left)\n"
              << "  -u, --urdf <path>       Path to URDF file (default: ../urdf/nero_cone-e_left.urdf)\n"
              << "  -g, --gripper           Enable gripper (default: false)\n"
              << "  -h, --help              Show this help message\n";
}

int main(int argc, char* argv[]) {
  // Default configuration
  std::string interface_name = "can_left";
  std::string urdf_path = "../urdf/nero_cone-e_left.urdf";
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

  try {
    ControllerConfig controller_config;
    controller_config.interface_name = interface_name;
    controller_config.urdf_path = urdf_path;
    controller_config.gripper_on = gripper_on;
    controller_config.gravity_compensation=false;

    spdlog::info("Controller config: interface={}, urdf={}, gripper={}",
                 controller_config.interface_name,
                 controller_config.urdf_path,
                 controller_config.gripper_on);

    NeroController joint_controller(controller_config);

    if (!joint_controller.start()) {
      spdlog::error("Failed to start controller");
      return 1;
    }

    spdlog::info("Controller started. Commanding various joint states...");

    joint_controller.resetToHome();

    joint_controller.setTarget({0.2, 0.2, -0.2, 0.3, -0.2, 0.5}, 0.0f);
    sleep_ms(3000);

    joint_controller.setTarget({-0.1, 0.3, -0.1, -0.2, -0.4, -0.3}, 0.5f);
    sleep_ms(3000);

    joint_controller.setTarget({0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, 1.0f);
    sleep_ms(3000);

    joint_controller.setTarget({0.1, 0.1, -0.2, 0.1, -0.1, 0.2}, 0.5f);
    sleep_ms(3000);

    joint_controller.setTarget({0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, 0.0f);
    sleep_ms(3000);

    spdlog::info("Test completed, stopping controller");
    joint_controller.stop();

  } catch (const std::exception &e) {
    spdlog::error("Error: {}", e.what());
    return 1;
  }

  spdlog::info("Test finished successfully");
  return 0;
}
