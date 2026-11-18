# SPDX-License-Identifier: Apache-2.0
"""cuVSLAM with the 2 cameras of ONE ZED X Wide stereo pair of the AMIAD rig in Isaac Sim.

camera:=front (default), rear, left or right picks the ZED; the extrinsics come from its
zed_<camera>_<eye>_camera_optical_frame TF (URDF), so no per-camera calibration file is needed.

Keep the sim's ground truth off base_link, which cuVSLAM publishes (--gt-child-frame base_link_gt).
cuVSLAM only takes mono8/rgb8 images, so the ZED wrapper's bgra8 color topics cannot be used.

source:=ros (default, launch_sim_teleop.sh default), ideal pinhole cameras published by the sim
(rgb8, identical stamps, /zed_<arm>/zed_node/<eye>/image_rect_color):
    <isaac>/python.sh amiad_description/isaac/run_sim.py --camera-mode ros --gt-child-frame base_link_gt
    ros2 launch isaac_ros_visual_slam isaac_ros_visual_slam_isaac_sim_amiad_2cam.launch.py
source:=zed (launch_sim_teleop.sh CAMERA_MODE=zed), ZED X streams through the ZED wrapper (gray rectified, mono8):
    <isaac>/python.sh amiad_description/isaac/run_sim.py --gt-child-frame base_link_gt
    ros2 launch amiad_description zed_sim_cameras.launch.py
    ros2 launch isaac_ros_visual_slam isaac_ros_visual_slam_isaac_sim_amiad_2cam.launch.py source:=zed
    ros2 launch isaac_ros_visual_slam isaac_ros_visual_slam_isaac_sim_amiad_2cam.launch.py source:=zed camera:=rear

Inputs (camera index -> topic, optical frame), <arm> = the camera argument:
    0,1 <arm>   (left, right; one stereo pair; the other ZEDs unused)
    ros: /zed_<arm>/zed_node/<eye>/image_rect_color, /zed_<arm>/zed_node/<eye>/camera_info
    zed: /zed_<arm>/zed_node/<eye>/gray/rect/image, /zed_<arm>/zed_node/<eye>/gray/rect/camera_info
    zed_<arm>_<eye>_camera_optical_frame (from the amiad_description URDF, robot_state_publisher)
imu:=true: VIO mode (tracking_mode 1), cuVSLAM also fuses imu_topic (default /pulse80/imu, frame
pulse80_imu_link; base_link -> pulse80_imu_link from the URDF TF). NVIDIA documents stereo-inertial
with one stereo pair: check the result with several pairs.
Outputs: map -> odom -> base_link TF (publish_tf:=false to disable), /visual_slam/tracking/odometry, ...
"""

import launch
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch_ros.substitutions import FindPackageShare

# ZEDs of the rig; this launch uses the one given by the camera argument
ARMS = ('front', 'rear', 'left', 'right')
# Order matters: cuVSLAM treats cameras (0,1), (2,3), ... as stereo pairs
EYES = ['left', 'right']


TOPICS = {
    'ros': ('/zed_{arm}/zed_node/{eye}/image_rect_color', '/zed_{arm}/zed_node/{eye}/camera_info'),
    'zed': ('/zed_{arm}/zed_node/{eye}/gray/rect/image', '/zed_{arm}/zed_node/{eye}/gray/rect/camera_info'),
}


# imu:=true. Noise = basalt calibration of the sim pulse80 IMU (zedx_*_calib*.json), so that
# both estimators model the same IMU (the node defaults are those of a RealSense)
IMU_PARAMS = {
    'imu_frame': 'pulse80_imu_link',   # '' = header.frame_id of the IMU messages
    'calibration_frequency': 200.0,    # Hz, run_sim.py --imu-hz
    'gyro_noise_density': 0.002,       # basalt gyro_noise_std
    'gyro_random_walk': 0.0007,        # basalt gyro_bias_std
    'accel_noise_density': 0.008,      # basalt accel_noise_std
    'accel_random_walk': 0.01,         # basalt accel_bias_std
    'imu_jitter_threshold_ms': 10.0,   # 200 Hz = 5 ms between messages
}


def launch_setup(context):
    source = LaunchConfiguration('source').perform(context)
    if source not in TOPICS:
        raise RuntimeError(f"source must be one of {list(TOPICS)}, got '{source}'")
    image_topic, info_topic = TOPICS[source]
    # '' when an including launch (vio_algorithms) declares camera without setting it
    arm = LaunchConfiguration('camera').perform(context).strip() or 'front'
    if arm not in ARMS:
        raise RuntimeError(f"camera must be one of {list(ARMS)}, got '{arm}'")
    cameras = [(arm, eye) for eye in EYES]
    remappings, optical_frames = [], []
    for idx, (arm, eye) in enumerate(cameras):
        remappings += [(f'visual_slam/image_{idx}', image_topic.format(arm=arm, eye=eye)),
                       (f'visual_slam/camera_info_{idx}', info_topic.format(arm=arm, eye=eye))]
        optical_frames.append(f'zed_{arm}_{eye}_camera_optical_frame')

    # imu:=true: VIO mode, cuVSLAM fuses the IMU (waits for its first message before starting)
    use_imu = LaunchConfiguration('imu').perform(context).lower() == 'true'
    if use_imu:
        remappings.append(('visual_slam/imu', LaunchConfiguration('imu_topic').perform(context)))

    visual_slam_node = ComposableNode(
        name='visual_slam_node',
        package='isaac_ros_visual_slam',
        plugin='nvidia::isaac_ros::visual_slam::VisualSlamNode',
        remappings=remappings,
        parameters=[{
            'use_sim_time': True,
            'tracking_mode': 1 if use_imu else 0,  # 1 = VIO (IMU fusion), 0 = multi camera
            'num_cameras': len(cameras),
            'min_num_images': len(cameras),
            'camera_optical_frames': optical_frames,
            'base_frame': 'base_link',
            'odom_frame': 'odom',
            'map_frame': 'map',
            # false: odometry topics only, e.g. when basalt owns odom/base_link on TF
            'publish_odom_to_base_tf': LaunchConfiguration('publish_tf').perform(context).lower() == 'true',
            'publish_map_to_odom_tf': LaunchConfiguration('publish_tf').perform(context).lower() == 'true',
            # first pose = TF odom -> align_frame (Isaac Sim ground truth): same reference
            'align_frame': LaunchConfiguration('align_frame').perform(context),
            'rectified_images': True,
            'enable_image_denoising': False,
            # ros: all cameras rendered on the same tick (identical stamps);
            # zed: one ZED stream, its left/right share a stamp
            'sync_matching_threshold_ms': float(LaunchConfiguration('sync_ms').perform(context)
                                                or 5.0),
            'image_jitter_threshold_ms': float(LaunchConfiguration('jitter_ms').perform(context)),
            'enable_slam_visualization': True,
            'enable_observations_view': True,
            'enable_landmarks_view': True,
            **(IMU_PARAMS if use_imu else {}),
        }])

    return [ComposableNodeContainer(
        name='visual_slam_launch_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[visual_slam_node],
        output='screen',
    )]


def generate_launch_description():

    # base_link -> camera optical frames for cuVSLAM's extrinsics
    robot_description = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [FindPackageShare('amiad_description'), 'launch', 'scout_amiad_display.launch.py'])),
        launch_arguments={'use_sim_time': 'true', 'rviz': LaunchConfiguration('rviz')}.items(),
        condition=IfCondition(LaunchConfiguration('robot_description')),
    )

    return launch.LaunchDescription([
        DeclareLaunchArgument('robot_description', default_value='true',
                              description='start robot_state_publisher with the AMIAD URDF '
                                          '(false if it already runs)'),
        DeclareLaunchArgument('rviz', default_value='false',
                              description='also start the amiad_description RViz'),
        DeclareLaunchArgument('publish_tf', default_value='true',
                              description='broadcast map -> odom -> base_link (false when another '
                                          'estimator, e.g. basalt, publishes base_link)'),
        DeclareLaunchArgument('camera', default_value='front',
                              description='ZED X stereo pair: front, rear, left or right'),
        DeclareLaunchArgument('source', default_value='ros',
                              description='ros: sim pinhole cameras, zed: ZED wrapper gray topics'),
        DeclareLaunchArgument('imu', default_value='false',
                              description='true: VIO mode (tracking_mode 1), also fuse imu_topic'),
        DeclareLaunchArgument('imu_topic', default_value='/pulse80/imu',
                              description='sensor_msgs/Imu topic fused with imu:=true'),
        DeclareLaunchArgument('sync_ms', default_value='',
                              description='max stamp difference between the 2 images of one frame '
                                          '(default 5)'),
        DeclareLaunchArgument('align_frame', default_value='',
                              description='TF frame (e.g. base_link_gt, Isaac Sim ground truth) whose '
                                          'pose in odom sets the first cuVSLAM pose: same reference '
                                          'as that odometry. Empty: cuVSLAM starts at identity'),
        DeclareLaunchArgument('jitter_ms', default_value='60.0',
                              description='max delta between frames before a warning '
                                          '(> 1000 / camera rate)'),
        robot_description,
        OpaqueFunction(function=launch_setup),
    ])
