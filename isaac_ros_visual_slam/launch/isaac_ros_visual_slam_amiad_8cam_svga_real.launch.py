# SPDX-License-Identifier: Apache-2.0
"""cuVSLAM, real AMIAD rig, 8 cameras (front, rear, left, right ZED X), svga: isaac_ros_visual_slam_amiad_real.launch.py
resolution:=svga num_cameras:=8 (kept for existing commands; all other arguments of that launch,
e.g. imu:=true, publish_tf:=false, are passed through).

Camera extrinsics: CAMERA_POSES of isaac_ros_visual_slam_amiad_real.launch.py, copied from basalt_ros
config/<res>/zedx_4stereo_calib_zed_real.json by amiad_nav/scripts/tools/gen_isaac_camera_poses.py.

    ros2 launch isaac_ros_visual_slam isaac_ros_visual_slam_amiad_8cam_svga_real.launch.py [imu:=true]
"""
import os

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    return LaunchDescription([IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   'isaac_ros_visual_slam_amiad_real.launch.py')),
        launch_arguments={'resolution': 'svga', 'num_cameras': '8'}.items())])
