import csv
from datetime import datetime
import zmq
import numpy as np
import atexit
import time
import threading
import os
import cv2

from loop_rate_limiters import RateLimiter
import mink

from robot.teleop.oculus_msgs import parse_controller_state

from commlink import RPCClient
import argparse


def apply_deadzone(arr, deadzone_size=0.05):
    return np.where(np.abs(arr) <= deadzone_size, 0, np.sign(arr) * (np.abs(arr) - deadzone_size) / (1 - deadzone_size))


# VR Constants
VR_TCP_HOST = "10.21.88.137"
ZED_POSE_HOST = "194.168.1.11"
VR_TCP_PORT = 5555
VR_CONTROLLER_TOPIC = b"oculus_controller"

BUTTON_DEBOUNCE_TIME = 0.2  # seconds
LOOP_RATE = 30  # Hz


class YORBimanualNoLiftTeleop:
    def __init__(self, zed_pose: bool = False, zed_image: bool = False, reset_base_after_data_collection: bool = False):
        self.reset_base_after_data_collection = reset_base_after_data_collection

        # teleop state
        self.zed_pose = zed_pose
        self.zed_image = zed_image
        print(f"[YORTeleop] ZED base pose recording: {self.zed_pose}, ZED image recording: {self.zed_image}")
        self.start_teleop_left = False
        self.start_teleop_right = False
        self.start_base_control = False
        self.data_list = []
        self.record_data = False

        # Coordinate transform: controller -> robot base frame
        self.H = mink.SE3.from_rotation(mink.SO3.from_matrix(np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])))
        self.X_Cinit_left = None
        self.X_ee_init_left = None
        self.X_Cinit_right = None
        self.X_ee_init_right = None

        # base control state
        self.max_vels = [
            np.array([0.5, 0.5, 1.57]),
            np.array([0.25, 0.25, 1.57]),
            np.array([0.75, 0.75, 1.57]),
            np.array([0.15, 0.15, 1.57 / 2.5])
        ]  # [teleop, slow, fast, recording]
        self.max_vel_setting = 0
        self.vel_alpha = 0.9
        self.last_target_velocity = np.array([0.0, 0.0, 0.0])

        # Robot RPC connection
        print("[YORTeleop] Connecting to Robot RPC at localhost:5557...")
        self.robot = RPCClient("localhost", 5557)
        self.robot.init()
        self.has_arms = True
        try:
            print("[YORTeleop] Homing arms...")
            self.robot.home_left_arm()
            self.robot.home_right_arm()
        except (ConnectionError, AttributeError, RuntimeError, Exception) as e:
            print(f"[YORTeleop] Warning: Could not home arms: {e}")
            self.has_arms = False

        if self.zed_pose or self.zed_image:
            from robot.teleop.zed_reader import ZedSub
            self.pose_stream = ZedSub(host=ZED_POSE_HOST)
            if self.zed_image:
                self.image_list = []
        self.stop_event = threading.Event()

        self.latest_controller_state = None
        self.controller_state_lock = threading.Lock()
        self.thread = threading.Thread(target=self._oculus_worker, daemon=True)
        self.thread.start()

    def _oculus_worker(self):
        print(f"[YORTeleop] Connecting to Oculus at {VR_TCP_HOST}:{VR_TCP_PORT}...")
        zmq_context = zmq.Context()
        stick_socket = zmq_context.socket(zmq.SUB)
        stick_socket.connect("tcp://{}:{}".format(VR_TCP_HOST, VR_TCP_PORT))
        stick_socket.subscribe(VR_CONTROLLER_TOPIC)

        while not self.stop_event.is_set():
            try:
                _, message = stick_socket.recv_multipart()
                controller_state = parse_controller_state(message.decode())
                with self.controller_state_lock:
                    self.latest_controller_state = controller_state
            except zmq.ZMQError:
                time.sleep(0.1)
            except Exception:
                pass

        stick_socket.close()
        zmq_context.destroy()

    def _compute_ee_target(self, X_Cinit, X_ee_init, X_Ctarget):
        """Compute the EE target pose using YOR-style decomposition (translate + rotate separately)."""
        X_Cdelta = X_Cinit.inverse().multiply(X_Ctarget)
        X_Rdelta = self.H.inverse() @ X_Cdelta @ self.H

        target_pos = X_ee_init.translation() + X_Rdelta.translation()
        target_rot = X_ee_init.rotation() @ X_Rdelta.rotation()
        return mink.SE3(np.concatenate([target_rot.wxyz, target_pos]))

    def control_loop(self):
        rate = RateLimiter(LOOP_RATE, name="yor_bimanual_no_lift")
        reset_rate = RateLimiter(LOOP_RATE, name="yor_bimanual_no_lift_reset")

        print("[YORTeleop] Ready. Left Stick: Rotate. Right Stick: Move. Both Grips: Toggle Base.")

        while not self.stop_event.is_set():
            with self.controller_state_lock:
                controller_state = self.latest_controller_state
            if controller_state is None:
                rate.sleep()
                continue

            # --- Arm teleop state machine ---
            if controller_state.left_x and self.has_arms:
                if not self.start_teleop_left:
                    self.X_ee_init_left = self.robot.get_left_ee_pose()
                    self.X_Cinit_left = controller_state.left_SE3
                    self.start_teleop_left = True
                    time.sleep(BUTTON_DEBOUNCE_TIME)
                    print("[Left] engaged")
                else:
                    self.start_teleop_left = False
                    time.sleep(BUTTON_DEBOUNCE_TIME)
                    print("[Left] disengaged")

            if controller_state.left_y and self.has_arms:
                print("[Left] Homing")
                self.robot.home_left_arm()
                self.start_teleop_left = False
                time.sleep(BUTTON_DEBOUNCE_TIME)

            if controller_state.right_a and self.has_arms:
                if not self.start_teleop_right:
                    self.X_ee_init_right = self.robot.get_right_ee_pose()
                    self.X_Cinit_right = controller_state.right_SE3
                    self.start_teleop_right = True
                    time.sleep(BUTTON_DEBOUNCE_TIME)
                    print("[Right] engaged")
                else:
                    self.start_teleop_right = False
                    time.sleep(BUTTON_DEBOUNCE_TIME)
                    print("[Right] disengaged")

            if controller_state.right_b and self.has_arms:
                print("[Right] Homing")
                self.robot.home_right_arm()
                self.start_teleop_right = False
                time.sleep(BUTTON_DEBOUNCE_TIME)

            # --- Compute arm targets ---
            x_Re_L, gripper_L, preview_time_L = None, 0, 1 / 15
            x_Re_R, gripper_R, preview_time_R = None, 0, 1 / 15

            if self.start_teleop_left:
                if self.X_Cinit_left is None or self.X_ee_init_left is None:
                    print("WARN: no initial pose yet")
                    time.sleep(0.01)
                    continue
                x_Re_L = self._compute_ee_target(self.X_Cinit_left, self.X_ee_init_left, controller_state.left_SE3)
                gripper_L = 1 if controller_state.left_index_trigger < 0.5 else 0

            if self.start_teleop_right:
                if self.X_Cinit_right is None or self.X_ee_init_right is None:
                    print("WARN: no initial pose yet")
                    time.sleep(0.01)
                    continue
                x_Re_R = self._compute_ee_target(self.X_Cinit_right, self.X_ee_init_right, controller_state.right_SE3)
                gripper_R = 1 if controller_state.right_index_trigger < 0.5 else 0

            # --- Send arm commands ---
            if self.start_teleop_left and not self.start_teleop_right:
                self.robot.set_left_ee_target(
                    ee_target=x_Re_L,
                    gripper_target=gripper_L,
                    preview_time=preview_time_L,
                )
            if self.start_teleop_right and not self.start_teleop_left:
                self.robot.set_right_ee_target(
                    ee_target=x_Re_R,
                    gripper_target=gripper_R,
                    preview_time=preview_time_R,
                )
            if self.start_teleop_left and self.start_teleop_right:
                self.robot.set_bimanual_ee_target(
                    L_ee_target=x_Re_L,
                    L_gripper_target=gripper_L,
                    L_preview_time=preview_time_L,
                    R_ee_target=x_Re_R,
                    R_gripper_target=gripper_R,
                    R_preview_time=preview_time_R,
                )

            # --- Base control ---
            vy = -controller_state.right_thumbstick_axes[0]  # Right stick horizontal
            vx = controller_state.right_thumbstick_axes[1]   # Right stick vertical
            w = -controller_state.left_thumbstick_axes[0]    # Left stick horizontal

            target_velocity = apply_deadzone(np.array([vx, vy, w]))
            target_velocity = self.max_vels[self.max_vel_setting] * target_velocity
            target_velocity = (1 - self.vel_alpha) * target_velocity + self.vel_alpha * self.last_target_velocity
            self.last_target_velocity = target_velocity

            if controller_state.left_hand_trigger > 0.5 and controller_state.right_hand_trigger > 0.5:
                self.start_base_control = not self.start_base_control
                print(f"[Base] Control {'ENABLED' if self.start_base_control else 'DISABLED'}")
                time.sleep(BUTTON_DEBOUNCE_TIME)

            if self.start_base_control:
                if sum(np.abs(target_velocity)) > 1e-2:
                    self.robot.set_base_velocity(target_velocity)
            else:
                target_velocity = np.zeros(3)

            # --- Data recording ---
            if self.record_data:
                curr_pose = self.robot.get_bimanual_state()
                if self.start_teleop_left and x_Re_L is not None:
                    L_ee_target_list = x_Re_L.wxyz_xyz.tolist()
                else:
                    L_ee_target_list = curr_pose[1:8]
                if self.start_teleop_right and x_Re_R is not None:
                    R_ee_target_list = x_Re_R.wxyz_xyz.tolist()
                else:
                    R_ee_target_list = curr_pose[16:23]

                record_row = curr_pose[:-1] + L_ee_target_list + R_ee_target_list
                record_row += target_velocity.tolist()

                if self.zed_pose:
                    base_wxyz_xyz, cam_wxyz_xyz, ts = self.pose_stream.get_quat_pose()
                    record_row += [ts]
                    record_row += base_wxyz_xyz
                    record_row += cam_wxyz_xyz
                if self.zed_image:
                    image, _ = self.pose_stream.get_image()
                    self.image_list.append(image)
                self.data_list.append(record_row)

            # --- Toggle recording ---
            if controller_state.left_menu:
                if not self.record_data:
                    self.record_data = True
                    self.max_vel_setting = 3
                    timestamp_str = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
                    filename = f'teleop_data/yor_bimanual_no_lift_log_{timestamp_str}.csv'
                    os.makedirs('teleop_data', exist_ok=True)
                    header = ['timestamp']
                    header += [f'left_ee_{i}' for i in ['w', 'qx', 'qy', 'qz', 'x', 'y', 'z']]
                    header += [f'left_joint_{i}' for i in range(7)]
                    header += ['left_gripper_pose']
                    header += [f'right_ee_{i}' for i in ['w', 'qx', 'qy', 'qz', 'x', 'y', 'z']]
                    header += [f'right_joint_{i}' for i in range(7)]
                    header += ['right_gripper_pose']
                    header += [f'left_ee_{i}_target' for i in ['w', 'qx', 'qy', 'qz', 'x', 'y', 'z']]
                    header += [f'right_ee_{i}_target' for i in ['w', 'qx', 'qy', 'qz', 'x', 'y', 'z']]
                    header += ['base_vx_target', 'base_vy_target', 'base_w_target']
                    if self.zed_pose:
                        assert self.pose_stream.ready(), "ZED pose stream not ready!"
                        header += ['zed_timestamp']
                        header += [f'base_{i}' for i in ['w', 'qx', 'qy', 'qz', 'x', 'y', 'z']]
                        header += [f'zed_{i}' for i in ['w', 'qx', 'qy', 'qz', 'x', 'y', 'z']]
                    if self.zed_image:
                        assert self.pose_stream.ready(), "ZED pose stream not ready!"
                        image_file_name = f'teleop_data/images_{timestamp_str}'
                        os.makedirs(image_file_name, exist_ok=True)
                    self.data_list = [header]
                    self.image_list = []
                    time.sleep(BUTTON_DEBOUNCE_TIME)
                    print(f"[Record] STARTED: {filename}")
                else:
                    self.record_data = False
                    self.start_teleop_left = False
                    self.start_teleop_right = False
                    self.max_vel_setting = 0
                    with open(filename, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerows(self.data_list)
                        f.flush()
                    print(f"[Record] STOPPED. Saved to {filename}")
                    if self.zed_image:
                        video_path = os.path.join(image_file_name, "images.mp4")
                        h, w = self.image_list[0].shape[:2]
                        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
                        for img in self.image_list:
                            out.write(img)
                        out.release()
                        print(f"[Record] Video saved: {video_path}")
                    time.sleep(BUTTON_DEBOUNCE_TIME)

                    if self.reset_base_after_data_collection:
                        reset_start_time = time.time()
                        for i in range(len(self.data_list) - 1, 0, -1):
                            *_, vx_r, vy_r, w_r = self.data_list[i]
                            if sum(np.abs([vx_r, vy_r, w_r])) > 1e-2:
                                self.robot.set_base_velocity(np.array([-vx_r, -vy_r, -w_r]))
                            reset_rate.sleep()
                        print(f"[Reset] Base reset in {time.time() - reset_start_time:.2f}s")

                    if self.has_arms:
                        print("[YORTeleop] Homing both arms after recording")
                        self.robot.home_left_arm()
                        self.robot.home_right_arm()

                    self.data_list = []

            rate.sleep()

    def stop(self):
        self.stop_event.set()
        self.thread.join()


def main(zed_pose, zed_image, reset_base):
    teleop = YORBimanualNoLiftTeleop(
        reset_base_after_data_collection=reset_base,
        zed_pose=zed_pose,
        zed_image=zed_image
    )
    atexit.register(teleop.stop)
    teleop.control_loop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--reset_base', '-r', action='store_true', help='Reset base after data collection')
    parser.add_argument('--zed_pose', '-b', action='store_true', help='Record ZED base pose data')
    parser.add_argument('--zed_image', '-i', action='store_true', help='Record ZED image and depth data')
    args = parser.parse_args()
    main(args.zed_pose, args.zed_image, args.reset_base)
