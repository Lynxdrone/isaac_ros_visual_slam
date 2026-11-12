# SPDX-License-Identifier: Apache-2.0
"""cuVSLAM with the 2 cameras of the front ZED X Wide stereo pair of the AMIAD rig in Isaac Sim.

Keep the sim's ground truth off base_link, which cuVSLAM publishes (--gt-child-frame base_link_gt).
cuVSLAM only takes mono8/rgb8 images, so the ZED wrapper's bgra8 color topics cannot be used.

source:=ros, ideal pinhole cameras published by the sim (rgb8, identical stamps):
    <isaac>/python.sh amiad_description/isaac/run_sim.py --camera-mode ros --gt-child-frame base_link_gt
    ros2 launch isaac_ros_visual_slam isaac_ros_visual_slam_isaac_sim_amiad_2cam.launch.py source:=ros
source:=zed (default, launch_sim_teleop.sh default), ZED X streams through the ZED wrapper (gray rectified, mono8):
    <isaac>/python.sh amiad_description/isaac/run_sim.py --gt-child-frame base_link_gt
    ros2 launch amiad_description zed_sim_cameras.launch.py
    ros2 launch isaac_ros_visual_slam isaac_ros_visual_slam_isaac_sim_amiad_2cam.launch.py

Inputs (camera index -> topic, optical frame):
    0,1 front   (left, right; one stereo pair; rear/left/right ZEDs unused)
    ros: /zed_<arm>/<eye>/image_rect_color, /zed_<arm>/<eye>/camera_info
    zed: /zed_<arm>/zed_node/<eye>/gray/rect/image, /zed_<arm>/zed_node/<eye>/gray/rect/camera_info
    zed_<arm>_<eye>_camera_optical_frame (from the amiad_description URDF, robot_state_publisher)
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

# Order matters: cuVSLAM treats cameras (0,1), (2,3), ... as stereo pairs
ARMS = ['front']
EYES = ['left', 'right']


TOPICS = {
    'ros': ('/zed_{arm}/{eye}/image_rect_color', '/zed_{arm}/{eye}/camera_info'),
    'zed': ('/zed_{arm}/zed_node/{eye}/gray/rect/image', '/zed_{arm}/zed_node/{eye}/gray/rect/camera_info'),
}


def launch_setup(context):
    source = LaunchConfiguration('source').perform(context)
    if source not in TOPICS:
        raise RuntimeError(f"source must be one of {list(TOPICS)}, got '{source}'")
    image_topic, info_topic = TOPICS[source]
    cameras = [(arm, eye) for arm in ARMS for eye in EYES]
    remappings, optical_frames = [], []
    for idx, (arm, eye) in enumerate(cameras):
        remappings += [(f'visual_slam/image_{idx}', image_topic.format(arm=arm, eye=eye)),
                       (f'visual_slam/camera_info_{idx}', info_topic.format(arm=arm, eye=eye))]
        optical_frames.append(f'zed_{arm}_{eye}_camera_optical_frame')

    visual_slam_node = ComposableNode(
        name='visual_slam_node',
        package='isaac_ros_visual_slam',
        plugin='nvidia::isaac_ros::visual_slam::VisualSlamNode',
        remappings=remappings,
        parameters=[{
            'use_sim_time': True,
            'tracking_mode': 0,  # multi camera
            'num_cameras': len(cameras),
            'min_num_images': len(cameras),
            'camera_optical_frames': optical_frames,
            'base_frame': 'base_link',
            'odom_frame': 'odom',
            'map_frame': 'map',
            # false: odometry topics only, e.g. when basalt owns odom/base_link on TF
            'publish_odom_to_base_tf': LaunchConfiguration('publish_tf').perform(context).lower() == 'true',
            'publish_map_to_odom_tf': LaunchConfiguration('publish_tf').perform(context).lower() == 'true',
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
        DeclareLaunchArgument('source', default_value='zed',
                              description='ros: sim pinhole cameras, zed: ZED wrapper gray topics'),
        DeclareLaunchArgument('sync_ms', default_value='',
                              description='max stamp difference between the 2 images of one frame '
                                          '(default 5)'),
        DeclareLaunchArgument('jitter_ms', default_value='60.0',
                              description='max delta between frames before a warning '
                                          '(> 1000 / camera rate)'),
        robot_description,
        OpaqueFunction(function=launch_setup),
    ])
