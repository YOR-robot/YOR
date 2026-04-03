#include "catch2/catch_test_macros.hpp"
#include "catch2/catch_session.hpp"
#include "nero_interface.h"

#include <cmath>
#include <spdlog/spdlog.h>
#include <iostream>
#include <string>

std::string g_interface_name = "can0";
bool g_gripper_active = false;

TEST_CASE("NeroInterface tests", "[nero]") {
  NeroInterface nero_interface(g_interface_name);

  if (nero_interface.get_arm_status() == ArmStatus::EMERGENCY_STOP) {
    spdlog::error("Arm status is emergency stop, first resume emergency stop!");
    exit(1);
  }

  SECTION("Get current state") {
    JointState state = nero_interface.get_current_state();
    spdlog::info("✓ Successfully retrieved current state");

    for (int i = 0; i < MOTOR_DOF; i++) {
      REQUIRE(state.pos[i] >= POS_MIN);
      REQUIRE(state.pos[i] <= POS_MAX);

      REQUIRE(state.vel[i] >= VEL_MIN);
      REQUIRE(state.vel[i] <= VEL_MAX);

      REQUIRE(state.torque[i] >= -50.0f);
      REQUIRE(state.torque[i] <= 50.0f);


      REQUIRE(std::isfinite(state.pos[i]));
      REQUIRE(std::isfinite(state.vel[i]));
      REQUIRE(std::isfinite(state.torque[i]));

      spdlog::info("✓ Motor {} pos={:.3f}, vel={:.3f}, tau={:.3f}", i,
                   state.pos[i], state.vel[i], state.torque[i]);
    }

    REQUIRE(std::isfinite(state.gripper_pos));
    spdlog::info("✓ Gripper pos={:.3f}", state.gripper_pos);
    spdlog::info("✓ Gripper effort={:.3f}", state.gripper_effort);
    spdlog::info("✓ All motor states are within reasonable bounds");
  }

  SECTION("Arm status") {
    ArmStatus arm_status = nero_interface.get_arm_status();
    spdlog::info("Arm status is {}", static_cast<uint8_t>(arm_status));
    REQUIRE(arm_status == ArmStatus::NORMAL);
  }

  SECTION("Arm enable/disable") {
    nero_interface.enable_arm();
    REQUIRE(nero_interface.is_arm_enabled());

    nero_interface.disable_arm();
    sleep_ms(500); // wait for low-speed feedback to reflect new enable status
    REQUIRE(!nero_interface.is_arm_enabled());

    spdlog::info("✓ Motor enable and disable test PASSED");
  }

  SECTION("Control mode switching") {
    // test position-velocity mode
    nero_interface.enable_arm();
    nero_interface.set_arm_mode(ControlMode::CAN_COMMAND, MoveMode::JOINT, 0x30, ArmController::POSITION_VELOCITY);

    REQUIRE(nero_interface.get_control_mode() == ControlMode::CAN_COMMAND);
    REQUIRE(nero_interface.get_move_mode() == MoveMode::JOINT);
    spdlog::info("✓ Set to joint mode successfully!");

    // test MIT mode
    nero_interface.set_arm_mode(ControlMode::CAN_COMMAND, MoveMode::MIT, 0x00, ArmController::MIT);
    REQUIRE(nero_interface.get_control_mode() == ControlMode::CAN_COMMAND);
    REQUIRE(nero_interface.get_move_mode() == MoveMode::MIT);
    spdlog::info("✓ Set to MIT mode successfully!");

    nero_interface.disable_arm();
  }


  if (g_gripper_active) {
    SECTION("Gripper position"){
      nero_interface.enable_gripper();
      if(!nero_interface.is_gripper_enabled()) {
        spdlog::error("Gripper cannot be enabled, trying to disable and enable again");
        nero_interface.disable_gripper();
        nero_interface.enable_gripper();
      }
      REQUIRE(nero_interface.is_gripper_enabled());
      double pos = nero_interface.get_current_state().gripper_pos;
      spdlog::info("Gripper pos: {:.3f}", pos);
      REQUIRE(pos >= -0.05);
      REQUIRE(pos <= 1.15);

      nero_interface.disable_gripper();
      REQUIRE(!nero_interface.is_gripper_enabled());
    }
  }
}

int main( int argc, char* argv[] ) {
  Catch::Session session;

  using namespace Catch::Clara;
  auto cli 
    = session.cli() 
    | Opt( g_interface_name, "name" )
        ["--interface"]
        ("CAN interface name")
    | Opt( g_gripper_active )
        ["--gripper-active"]
        ("Enable gripper tests");

  session.cli( cli );

  int returnCode = session.applyCommandLine( argc, argv );
  if( returnCode != 0 ) 
      return returnCode;

  return session.run();
}
