#include "nero_interface.h"
#include "utils.h"
#include <iostream>
#include <csignal>
#include <spdlog/spdlog.h>

bool running = true;
void signal_handler(int signum) { running = false; }

int main(int argc, char *argv[]) {
  signal(SIGINT, signal_handler);

  std::string interface_name = "can_left";
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-i" || arg == "--interface") {
        if (i + 1 < argc) interface_name = argv[++i];
    }
  }

  NeroInterface nero_interface(interface_name, false);
  
  if (nero_interface.get_arm_status() == ArmStatus::EMERGENCY_STOP) {
    spdlog::error("Arm status is emergency stop, first resume emergency stop!");
    exit(1);
  }

  // CRITICAL: reset_arm is required to put the firmware into MIT mode
  reset_arm(nero_interface);

  RateLimiter rate_limiter(200);

  // Home position (from armnew.py for left arm)
  std::array<double, MOTOR_DOF> home_pos = {1.57, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  
  // Normal stiff gain to move there safely
  Gain move_gain = Gain({15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0}, 
                        {0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8});

  // Target compliant gain
  Gain compliant_gain = Gain({0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, 
                             {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5});

  JointState initial_state = nero_interface.get_current_state();
  JointState output_state;
  output_state.torque = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  output_state.vel = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  
  spdlog::info("Moving to HOME POSITION...");

  int total_steps = 200 * 3; // 3 seconds at 200Hz
  for (int i = 0; i <= total_steps && running; i++) {
    float t = float(i) / total_steps;
    for (int j = 0; j < MOTOR_DOF; j++) {
      output_state.pos[j] = initial_state.pos[j] + (home_pos[j] - initial_state.pos[j]) * t;
    }
    nero_interface.set_joint_pos_vel_torque(output_state, move_gain);
    rate_limiter.wait();
  }

  spdlog::info("Reached home position! Holding for 2 seconds...");
  for (int i = 0; i < 200 * 2 && running; i++) {
    for (int j = 0; j < MOTOR_DOF; j++) {
      output_state.pos[j] = home_pos[j];
    }
    nero_interface.set_joint_pos_vel_torque(output_state, move_gain);
    rate_limiter.wait();
  }

  spdlog::info("Dropping Kp to 5.0 (Low Stiffness). Entering SPRING-LIKE COMPLIANT MODE!");
  
  // Spring-like compliant gain
  Gain spring_gain = Gain({2.0, 8.0, 2.0, 2.0, 2.0, 2.0, 2.0}, 
                          {0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2});
                             
  while (running) {
    // Keep target at home_pos so it always pulls back like a spring
    for (int j = 0; j < MOTOR_DOF; j++) {
      output_state.pos[j] = home_pos[j];
    }
    nero_interface.set_joint_pos_vel_torque(output_state, move_gain);
    rate_limiter.wait();
  }

  spdlog::info("Shutting down... disabling arm.");
  nero_interface.disable_arm();
  return 0;
}
