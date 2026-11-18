# SPDX-License-Identifier: Apache-2.0
"""cuVSLAM with the front and rear ZED X stereo pairs of the real AMIAD rig (Isaac Sim version:
isaac_ros_visual_slam_isaac_sim_amiad_4cam.launch.py).

Camera extrinsics from the Kalibr camera-IMU calibrations of the front and rear ZED X with the Epson IMU
(front_cameras_imu.bag_0-camchain-imucam.yaml, rear_cameras_imu.bag_0-camchain-imucam.yaml, same values
as basalt_ros zedx_2stereo_calib_zed_real.json):
static TFs imu_link -> zed_<arm>_{left,right}_camera_optical_frame_epson. Separate frame names, so they
do not clash with the CAD optical frames of the rover URDF; base_link -> imu_link comes from that URDF
(rover_description rover_amiad.urdf.xacro, started by amiad_nav onboard.launch.py).
Intrinsics: camera_info of the ZED wrapper (rectified images, no distortion).

    ros2 launch amiad_nav onboard.launch.py        # ZED X, Epson IMU (/imu/data), rover URDF
    ros2 launch isaac_ros_visual_slam isaac_ros_visual_slam_amiad_4cam_real.launch.py [imu:=true]

Inputs (camera index -> topic, optical frame):
    0,1 front  2,3 rear   (left, right of each ZED; pairs = stereo; left/right ZEDs unused)
    /zed_<arm>/zed_node/<eye>/gray/rect/image, /zed_<arm>/zed_node/<eye>/gray/rect/camera_info
    zed_<arm>_<eye>_camera_optical_frame_epson
    imu:=true: /imu/data (frame imu_link), cuVSLAM VIO mode
Outputs: map -> odom -> base_link TF (publish_tf:=false to disable), /visual_slam/tracking/odometry, ...
"""

import os

import launch
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode

IMU_FRAME = 'imu_link'

# Kalibr T_imu_cam (camera optical frame pose in the Epson IMU frame, x forward, y left, z up):
# inverse of T_cam_imu of the front / rear camchain-imucam.yaml.
# Order matters: cuVSLAM treats cameras (0,1), (2,3) as stereo pairs.
# Kalibr timeshift_cam_imu: front 4.74 / 4.79 ms, rear -2.35 / 0.27 ms: not compensated by cuVSLAM.
# cuVSLAM (rectified_images) needs the 2 cameras of a pair with the exact same rotation: the ZED
# images are rectified, so right = left rotation, shifted by the Kalibr baseline (front 0.119326 m,
# rear 0.119176 m) along the left optical x axis (< 0.5 mm from the Kalibr right camera positions;
# Kalibr's ~0.03 deg left/right rotation is calibration noise on rectified images).
CAMERAS = [
    # (arm, eye, x, y, z, qx, qy, qz, qw)
    ('front', 'left', 0.212808, 0.058007, 0.081465, -0.49609879, 0.503399112, -0.499256903, 0.50121639),
    ('front', 'right', 0.212171, -0.061312, 0.08036, -0.49609879, 0.503399112, -0.499256903, 0.50121639),
    ('rear', 'left', -0.201752, -0.072387, 0.034756, -0.505409369, -0.492488832, 0.48838072, 0.513322893),
    ('rear', 'right', -0.197238, 0.046695, 0.03618, -0.505409369, -0.492488832, 0.48838072, 0.513322893),
]
IMAGE_TOPIC = '/zed_{arm}/zed_node/{eye}/gray/rect/image'
INFO_TOPIC = '/zed_{arm}/zed_node/{eye}/gray/rect/camera_info'


def optical_frame(arm, eye):
    return f'zed_{arm}_{eye}_camera_optical_frame_epson'


def launch_setup(context):
    use_imu = LaunchConfiguration('imu').perform(context).lower() == 'true'
    publish_tf = LaunchConfiguration('publish_tf').perform(context).lower() == 'true'

    actions = [
        Node(package='tf2_ros', executable='static_transform_publisher',
             name=f'epson_{IMU_FRAME}_to_zed_{arm}_{eye}',
             arguments=['--x', str(x), '--y', str(y), '--z', str(z),
                        '--qx', str(qx), '--qy', str(qy), '--qz', str(qz), '--qw', str(qw),
                        '--frame-id', IMU_FRAME, '--child-frame-id', optical_frame(arm, eye)])
        for arm, eye, x, y, z, qx, qy, qz, qw in CAMERAS]

    remappings = []
    for idx, (arm, eye, *_) in enumerate(CAMERAS):
        remappings += [(f'visual_slam/image_{idx}', IMAGE_TOPIC.format(arm=arm, eye=eye)),
                       (f'visual_slam/camera_info_{idx}', INFO_TOPIC.format(arm=arm, eye=eye))]
    parameters = {
        'use_sim_time': False,
        'tracking_mode': 1 if use_imu else 0,  # 1: VIO (IMU fusion), 0: multi camera
        'num_cameras': len(CAMERAS),
        'min_num_images': len(CAMERAS),
        'camera_optical_frames': [optical_frame(arm, eye) for arm, eye, *_ in CAMERAS],
        'base_frame': 'base_link',
        'odom_frame': 'odom',
        'map_frame': 'map',
        # false: odometry topics only, e.g. when basalt owns odom/base_link on TF
        'publish_odom_to_base_tf': publish_tf,
        'publish_map_to_odom_tf': publish_tf,
        'rectified_images': True,
        'enable_image_denoising': False,
        # the 2 ZED streams are stamped independently: looser than within one ZED
        'sync_matching_threshold_ms': float(LaunchConfiguration('sync_ms').perform(context)),
        'image_jitter_threshold_ms': float(LaunchConfiguration('jitter_ms').perform(context)),
        'enable_slam_visualization': True,
        'enable_observations_view': True,
        'enable_landmarks_view': True,
    }
    if use_imu:
        remappings.append(('visual_slam/imu', LaunchConfiguration('imu_topic').perform(context)))
        parameters.update({
            'imu_frame': IMU_FRAME,
            # Epson G366 noise model, Kalibr front_cameras_imu.bag_0-imu.yaml (imu0, /imu/data,
            # update_rate 400 Hz = rate the noise model was identified at)
            'calibration_frequency': 400.0,
            'gyro_noise_density': 2.5168513991925153e-05,   # rad/s/sqrt(Hz)
            'gyro_random_walk': 4.878091752029107e-07,      # rad/s^2/sqrt(Hz)
            'accel_noise_density': 0.0005457397459599346,   # m/s^2/sqrt(Hz)
            'accel_random_walk': 6.113996506614726e-05,     # m/s^3/sqrt(Hz)
            'imu_jitter_threshold_ms': 2000.0 / float(LaunchConfiguration('imu_rate').perform(context)),
        })

    visual_slam_node = ComposableNode(
        name='visual_slam_node',
        package='isaac_ros_visual_slam',
        plugin='nvidia::isaac_ros::visual_slam::VisualSlamNode',
        remappings=remappings,
        parameters=[parameters])

    actions.append(ComposableNodeContainer(
        name='visual_slam_launch_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[visual_slam_node],
        output='screen',
    ))
    return actions


def generate_launch_description():

    # base_link -> imu_link (and the rig frames); onboard.launch.py already runs it
    robot_description = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('rover_description'), 'launch', 'state_publisher.launch.py')),
        launch_arguments={'model': os.path.join(get_package_share_directory('rover_description'),
                                                'urdf', 'rover_amiad.urdf.xacro'),
                          'jsp': 'false', 'use_sim_time': 'false'}.items(),
        condition=IfCondition(LaunchConfiguration('robot_description')),
    )

    return launch.LaunchDescription([
        DeclareLaunchArgument('robot_description', default_value='false',
                              description='start robot_state_publisher with the rover + AMIAD short rig '
                                          'URDF (true if onboard.launch.py is not running)'),
        DeclareLaunchArgument('publish_tf', default_value='true',
                              description='broadcast map -> odom -> base_link (false when another '
                                          'estimator, e.g. basalt, publishes base_link)'),
        DeclareLaunchArgument('imu', default_value='false',
                              description='true: VIO mode with the Epson IMU (Kalibr extrinsics)'),
        DeclareLaunchArgument('imu_topic', default_value='/imu/data',
                              description='Epson IMU topic (imu:=true)'),
        DeclareLaunchArgument('imu_rate', default_value='400.0',
                              description='IMU output rate [Hz], for the IMU jitter check '
                                          '(onboard.launch.py imu_dout_rate 8 = 400 Hz)'),
        DeclareLaunchArgument('sync_ms', default_value='20.0',
                              description='max stamp difference between the 4 images of one frame'),
        DeclareLaunchArgument('jitter_ms', default_value='60.0',
                              description='max delta between frames before a warning '
                                          '(> 1000 / camera rate)'),
        robot_description,
        OpaqueFunction(function=launch_setup),
    ])
