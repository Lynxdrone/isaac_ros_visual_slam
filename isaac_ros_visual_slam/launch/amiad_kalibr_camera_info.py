#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Publish a CameraInfo per image, built from a (merged) Kalibr camchain-imucam yaml.

The ZED bags have no camera_info for the left eyes nor for the gray images, and cuVSLAM needs one
per camera. For each camN of the yaml this node subscribes to camN.rostopic (or the remapped
topic given in --image-topics) and publishes on --info-topics[N] a CameraInfo with the image stamp,
frame_id --frames[N], the Kalibr intrinsics and plumb_bob distortion [k1, k2, p1, p2, 0].

Started by isaac_ros_visual_slam_amiad_kalibr.launch.py.
"""
import argparse
import sys

import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


class KalibrCameraInfo(Node):

    def __init__(self, calib, image_topics, info_topics, frames):
        super().__init__('kalibr_camera_info')
        self._subs = []
        for idx, (image_topic, info_topic, frame) in enumerate(zip(image_topics, info_topics, frames)):
            cam = calib[f'cam{idx}']
            fx, fy, cx, cy = cam['intrinsics']
            info = CameraInfo()
            info.width, info.height = cam['resolution']
            info.distortion_model = 'plumb_bob'
            info.d = [float(v) for v in cam['distortion_coeffs']] + [0.0]
            info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
            info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
            info.header.frame_id = frame
            pub = self.create_publisher(CameraInfo, info_topic, 10)
            # sensor-data QoS (best effort) matches both reliable bag playback and the ZED wrapper
            self._subs.append(self.create_subscription(
                Image, image_topic, self._callback(info, pub), qos_profile_sensor_data))
            self.get_logger().info(f'cam{idx}: {image_topic} -> {info_topic} ({frame})')

    @staticmethod
    def _callback(info, pub):
        def cb(msg):
            info.header.stamp = msg.header.stamp
            pub.publish(info)
        return cb


def main():
    argv = rclpy.utilities.remove_ros_args(sys.argv)[1:]
    p = argparse.ArgumentParser()
    p.add_argument('--calib', required=True)
    p.add_argument('--image-topics', nargs='+', required=True)
    p.add_argument('--info-topics', nargs='+', required=True)
    p.add_argument('--frames', nargs='+', required=True)
    args = p.parse_args(argv)
    with open(args.calib) as f:
        calib = yaml.safe_load(f)

    rclpy.init()
    node = KalibrCameraInfo(calib, args.image_topics, args.info_topics, args.frames)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
