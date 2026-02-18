#!/usr/bin/env python3
import time
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
from commlink import Publisher
from loop_rate_limiters import RateLimiter

from robot.utils.utils import pose_to_matrix, theta_y_from_R

import pyzed.sl as sl



ZED_PUB_PORT = 6000

POSE_TOPIC  = "zed/pose"
IMAGE_TOPIC = "zed/image"
DEPTH_TOPIC = "zed/depth"
PCD_TOPIC = "zed/pcd"
QUAT_XYZ_TOPIC = "zed/quat_xyz"

class ZEDCamReader:
    def __init__(
        self,
        resolution: sl.RESOLUTION = sl.RESOLUTION.HD720,
        fps: int = 60,
        depth_mode: sl.DEPTH_MODE = sl.DEPTH_MODE.NEURAL,
        depth_max_range_m: float = 8.0,
        confidence_threshold: int = 90,
        texture_confidence: int = 98,
        set_floor_as_origin: bool = True,
    ):
        
        # Zed Init Params
        self._init_params = sl.InitParameters()
        self._init_params.depth_mode = depth_mode
        self._init_params.coordinate_units = sl.UNIT.METER
        self._init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
        self._init_params.depth_maximum_distance = float(depth_max_range_m)
        self._init_params.camera_resolution = resolution
        self._init_params.camera_fps = fps
        self._init_params.depth_stabilization = True

        self.area_file_path = "saved_map.area"

        self.set_floor_as_origin = set_floor_as_origin
        self.sl_memory_type = sl.MEM.CPU

        self._runtime = sl.RuntimeParameters()
        
        self._runtime.confidence_threshold = int(confidence_threshold)
        self._runtime.texture_confidence_threshold = int(texture_confidence)
        
        # Camera & mats
        self._zed: sl.Camera

        self._left_rgba = sl.Mat()
        self._depth_m = sl.Mat()
        self._conf_f = sl.Mat()
        self._pcd = sl.Mat()
        self._pose = sl.Pose()

        # Intrinsics cache
        self._fx = self._fy = None
        self._W = self._H = None
        self._cx = self._cy = None

        # Commlink Publisher
        self.pub = Publisher("*", port=ZED_PUB_PORT)

        # Rate Limiting
        self.rate = RateLimiter(fps, name="Zed sub")


    def stop(self):
        self._zed.disable_positional_tracking(self.area_file_path)
        self._zed.close()
        self._zed = None
    
    def open_and_start(self) -> sl.Camera:
        cam = sl.Camera()
        err = cam.open(self._init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED open failed: {repr(err)}")

        # Positional tracking
        pt_params = sl.PositionalTrackingParameters(_enable_imu_fusion = True, _mode = sl.POSITIONAL_TRACKING_MODE.GEN_3)
        pt_params.set_floor_as_origin = self.set_floor_as_origin
        pt_params.set_gravity_as_origin = True
        pt_params.enable_area_memory = True
        pt_params.enable_pose_smoothing = True

        if os.path.exists(self.area_file_path):
            pt_params.area_file_path = self.area_file_path

        err = cam.enable_positional_tracking(pt_params)
        
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Zed enable_positional_tracking failed: {repr(err)}")

        # Cache intrinsics
        ci = cam.get_camera_information()
        calib = ci.camera_configuration.calibration_parameters.left_cam
        self.fx, self.fy = float(calib.fx), float(calib.fy)
        self.cx, self.cy = float(calib.cx), float(calib.cy)
        self._W = int(ci.camera_configuration.resolution.width)
        self._H = int(ci.camera_configuration.resolution.height)

        return cam

    def is_pose_good(self):
        good_pose = True
        bad_states = {"INITIALIZING", "SEARCHING", "LOST"}

        spatial_status = self._zed.get_positional_tracking_status().spatial_memory_status.name.split(".")[-1]

        tracking_state = self._zed.get_position(self._pose, sl.REFERENCE_FRAME.WORLD)

        if spatial_status in bad_states or (not tracking_state == sl.POSITIONAL_TRACKING_STATE.OK):
            good_pose = False
        else:
            good_pose = True            

        print("Positional Tracking status: " + str(tracking_state))
        print("Spatial Mapping Status:" + spatial_status)

        return good_pose      
        

    def run(self):
        self._zed = self.open_and_start()
        print("Zed Publisher Started. Publishing: " + IMAGE_TOPIC + DEPTH_TOPIC + POSE_TOPIC + PCD_TOPIC)

        # Preallocate Mats once
        self._left_rgba = sl.Mat(self._W, self._H, sl.MAT_TYPE.U8_C4, memory_type=self.sl_memory_type)
        self._depth_m = sl.Mat(self._W, self._H, sl.MAT_TYPE.F32_C1, memory_type=self.sl_memory_type)
        self._conf_f = sl.Mat(self._W, self._H, sl.MAT_TYPE.F32_C1, memory_type=self.sl_memory_type)
        self._pcd = sl.Mat(self._W, self._H, sl.MAT_TYPE.F32_C4, memory_type=self.sl_memory_type)

        try:
            while True:

                if self._zed.grab(self._runtime) != sl.ERROR_CODE.SUCCESS or (not self.is_pose_good()):
                    time.sleep(0.002)
                    continue

                # Retrieve all measures (only when pose is good)
        
                self._zed.retrieve_image(self._left_rgba, sl.VIEW.LEFT, mem_type=self.sl_memory_type)
                self._zed.retrieve_measure(self._depth_m, sl.MEASURE.DEPTH, mem_type=self.sl_memory_type)
                self._zed.retrieve_measure(self._conf_f, sl.MEASURE.CONFIDENCE, mem_type=self.sl_memory_type)
                self._zed.retrieve_measure(self._pcd, sl.MEASURE.XYZRGBA, mem_type=self.sl_memory_type)

                # Convert to numpy
                img_rgb = self._left_rgba.get_data(self.sl_memory_type)[..., :3]  # (H,W,4) uint8 RGBA

                depth_m = self._depth_m.get_data(self.sl_memory_type).astype(np.float32, copy=False)

                conf = self._conf_f.get_data(self.sl_memory_type).astype(np.float32, copy=False)
                # Map confidence [0..100] -> {0,2}; use runtime threshold if set
                
                thresh = int(self._runtime.confidence_threshold)
                
                conf_u8 = np.zeros_like(conf, dtype=np.uint8)
                conf_u8[conf >= thresh] = 2

                # Pose to [qx,qy,qz,qw, tx,ty,tz]
                q = self._pose.get_orientation(sl.Orientation()).get()  # [ox, oy, oz, ow]
                t = self._pose.get_translation(sl.Translation()).get()  # [tx, ty, tz]
                pose_qt = np.array([q[0], q[1], q[2], q[3], t[0], t[1], t[2]], dtype=np.float32)
                # pose_qt = np.array([*q, *t], dtype=np.float32)

                Twc = pose_to_matrix(pose_qt)

                angle = np.deg2rad(23.8) # -23.8
                # Top
                # x = 0.0603
                # z = 0.2143
                Tcb = np.array(
                    [
                        [1,             0,             0, 0.0],
                        [0, np.cos(angle),-np.sin(angle), 0.0],
                        [0, np.sin(angle), np.cos(angle), 0.0],
                        [0, 0, 0, 1.0],
                    ],
                    dtype=np.float64)
                
                Tcb_t = np.array(
                    [
                        [1, 0, 0, 0.0603],
                        [0, 1, 0, 0.0   ],
                        [0, 0, 1, 0.2143],
                        [0, 0, 0, 1.0],
                    ],
                    dtype=np.float64)
                
                Tcb_2 = np.array(
                    [
                        [ 0,-1, 0, 0.0],
                        [ 0, 0, 1, 0.0], 
                        [-1, 0, 0, 0.0],
                        [ 0, 0, 0, 1.0],
                    ],
                    dtype=np.float64)

                Twb = Twc @ Tcb @ Tcb_t @ Tcb_2

                translation = Twb[:3, 3]
                rotation_y = theta_y_from_R(Twb)

                pose_base = np.array([translation[0], translation[1], translation[2], rotation_y], dtype=np.float32).tolist()

                # Pointcloud
                pcd_arr = self._pcd.get_data(self.sl_memory_type)

                ts = time.time_ns()
                # 1) RGB image
                img_msg = {
                    "timestamp": ts,
                    "image": img_rgb,
                }

                # 2) Depth + intrinsics
                depth_msg = {
                    "timestamp": ts,
                    "depth": depth_m,                    
                }

                # 3) Camera pose WORLD_T_CAM
                # pose_msg = {
                #     "timestamp": ts,
                #     "camera_pose": pose_qt,
                #     "base_pose": pose_base,
                #     "base_pose_6DOF": Twb,
                # }

                # 4) Pointcloud in camera frame
                pcd_msg = {
                    "timestamp": ts,
                    "points": pcd_arr,
                }

                # 5) pose in list format
                base_quat_xyz = R.from_matrix(Twb[:3, :3]).as_quat()  # [qx, qy, qz, qw]
                base_quat_xyz = base_quat_xyz.tolist() + Twb[:3, 3].tolist() # [qx, qy, qz, qw, x, y, z]
                cam_quat_xyz = R.from_matrix(Twc[:3, :3]).as_quat()  # [qx, qy, qz, qw]
                cam_quat_xyz = cam_quat_xyz.tolist() + Twc[:3, 3].tolist() # [qx, qy, qz, qw, x, y, z]
                pose_quat_xyz = base_quat_xyz + cam_quat_xyz + pose_base + [ts]

                # Base pose: 0:7    Camera pose: 7:14   Base pose(tx, ty, tz, theta_y): 14:18

                # 5) OPTIONAL: Camera intrinsics and info
                # camera_info = {
                #     "timestamp": ts,
                #     "confidence": conf_u8,
                #     "focal": [int(self.fx), int(self.fy), int(self.cx), int(self.cy)],
                #     "resolution": [int(self._W), int(self._H)],
                #     "width": int(self._W),
                #     "height": int(self._H),
                # }

                # Publish
                self.pub.publish(IMAGE_TOPIC, img_msg)
                self.pub.publish(DEPTH_TOPIC, depth_msg)
                # self.pub.publish(POSE_TOPIC, pose_msg)
                self.pub.publish(PCD_TOPIC, pcd_msg)
                self.pub.publish(POSE_TOPIC, pose_quat_xyz)

                # Rate limiter
                self.rate.sleep()

        except KeyboardInterrupt:
            print("[zed_pub] Shutting down...")
        finally:
            self.stop()


def main():    
    fps = 60
    cam = ZEDCamReader(
        resolution=sl.RESOLUTION.VGA,
        fps=fps,
        depth_mode=sl.DEPTH_MODE.NEURAL,
        depth_max_range_m=4.0,
        confidence_threshold=87,
        texture_confidence=95,
        set_floor_as_origin=True,
    )

    cam.run()


if __name__ == "__main__":
    main()
    
