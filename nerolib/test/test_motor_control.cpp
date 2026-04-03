#include "catch2/catch_test_macros.hpp"
#include "nero_interface.h"
#include "iostream"
#include "utils.h"

#include <cmath>
#include <spdlog/spdlog.h>

TEST_CASE("Motor control tests", "[nero]") {
  NeroInterface nero_interface("can0");

  reset_arm(nero_interface);
  reset_gripper(nero_interface);

  SECTION("Motor control") {
    JointState initial_joint_state = nero_interface.get_current_state();
    spdlog::info("Initial joint state: {}", initial_joint_state.gripper_pos);

    JointState desired_joint_state = JointState(
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, 1.0);

    Gain gain = Gain({10.0, 10.0, 10.0, 10.0, 10.0, 10.0},
                     {0.8, 0.8, 0.8, 0.8, 0.8, 0.8});

    float control_freq = 200.0;
    float time_to_reach_pos = 1.0; // seconds
    int total_steps = int(time_to_reach_pos * control_freq);

    JointState ref_joint_state;
    for (int i = 0; i < MOTOR_DOF; i++) {
      ref_joint_state.vel[i] =
          (desired_joint_state.pos[i] - initial_joint_state.pos[i]) /
          time_to_reach_pos;
    }
    ref_joint_state.gripper_pos = desired_joint_state.gripper_pos;
    ref_joint_state.torque = {0.0,   0.0, 0.0, 0.0,
                              -0.85, 0.0}; // Gravity compensation for joint 5

    spdlog::info("Arm is going to move. Press Enter to continue...");
    std::cin.get();
    RateLimiter rate_limiter(control_freq);
    for (int i = 0; i < total_steps; i++) {
      for (int j = 0; j < MOTOR_DOF; j++) {
        ref_joint_state.pos[j] =
            initial_joint_state.pos[j] +
            (desired_joint_state.pos[j] - initial_joint_state.pos[j]) *
                (float(i) / total_steps);

        ref_joint_state.gripper_pos =
            initial_joint_state.gripper_pos +
            (desired_joint_state.gripper_pos - initial_joint_state.gripper_pos) *
                (float(i) / total_steps);
      }
      nero_interface.set_joint_pos_vel_torque(ref_joint_state, gain);
      nero_interface.set_gripper(ref_joint_state.gripper_pos, 0.2f);
      rate_limiter.wait();
    }

    sleep_ms(1000); // wait for the arm to settle

    JointState final_joint_state = nero_interface.get_current_state();
    for (int i = 0; i < MOTOR_DOF; i++) {
      float diff =
          std::abs(final_joint_state.pos[i] - desired_joint_state.pos[i]);
      CHECK(diff < 0.05f);
      spdlog::info("✓ Motor {} pos={}, pos_des={}, diff={:.3f}", i,
                   final_joint_state.pos[i], desired_joint_state.pos[i], diff);
    }

    CHECK(std::abs(final_joint_state.gripper_pos - desired_joint_state.gripper_pos) < 0.02);
    spdlog::info("✓ Gripper pos={}, pos_des={}, diff={:.3f}", final_joint_state.gripper_pos, desired_joint_state.gripper_pos, std::abs(final_joint_state.gripper_pos - desired_joint_state.gripper_pos));

    spdlog::info("✓ Motor control test PASSED");
  }
  nero_interface.set_to_damping_mode();

  spdlog::info("Press Enter to disable arm...");
  std::cin.get();
  nero_interface.disable_arm();
  nero_interface.disable_gripper();
}