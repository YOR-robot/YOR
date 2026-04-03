from nerolib import NeroController, ControllerConfig
import time

class TestPythonBindings:
    def test_1(self):
        controller_config = ControllerConfig()
        nero_controller = NeroController(controller_config)
        
        if not nero_controller.start():
            raise RuntimeError("Failed to start Nero controller")
        
        nero_controller.set_target(
            new_target_pos=[0.0] * 6,
            new_target_vel=[0.0] * 6,
            new_target_acc=[0.0] * 6,
        )

        time.sleep(1)

        nero_controller.stop()


if __name__ == "__main__":
    test = TestPythonBindings()
    test.test_1()
        
