# SPDX-License-Identifier: Apache-2.0
"""cuVSLAM with 1 to 4 ZED X stereo pairs of the real AMIAD rig, SVGA or HD1080 (one launch for the
isaac_ros_visual_slam_amiad_{2,4,6,8}cam_{svga,1080p}_real.launch.py variants;
Isaac Sim versions: isaac_ros_visual_slam_isaac_sim_amiad_{2,4,6,8}cam.launch.py).

    ros2 launch amiad_nav onboard.launch.py        # ZED X, Epson IMU (/imu/data), rover URDF
    ros2 launch isaac_ros_visual_slam isaac_ros_visual_slam_amiad_real.launch.py \\
        resolution:=1080p num_cameras:=8 [imu:=true]

Arguments:
    resolution   svga (ZED SVGA 960x600) | 1080p (ZED HD1080 1920x1080), default 1080p: selects the
                 Kalibr calibration set; must match the ZED wrapper grab_resolution (intrinsics come
                 from its camera_info).
    num_cameras  2 | 4 | 6 | 8 (2 images per ZED), default 2:
                 2: front   4: + rear   6: + left   8: + right
    cameras      any set of ZED, comma separated, overrides num_cameras, e.g. cameras:=left,right
                 or cameras:=rear,left,right (sorted front, rear, left, right; default '': num_cameras)

Camera extrinsics: one Epson IMU (imu_link) and, per resolution, the Kalibr camera-IMU calibration of each
ZED X (<arm>_cameras_imu[_1080p].bag_0-camchain-imucam.yaml, same values as basalt_ros
config/<resolution>/zedx_*_calib_zed_real.json): static TFs imu_link -> zed_<arm>_{left,right}_camera_optical_frame_epson.
Separate frame names, so they do not clash with the CAD optical frames of the rover URDF; base_link -> imu_link
comes from that URDF (rover_description rover_amiad.urdf.xacro, started by amiad_nav onboard.launch.py,
or robot_description:=true).

Inputs (camera index -> topic, optical frame):
    0,1 front  2,3 rear  4,5 left  6,7 right   (left, right of each ZED; pairs = stereo)
    /zed_<arm>/zed_node/<eye>/gray/rect/image, /zed_<arm>/zed_node/<eye>/gray/rect/camera_info
    zed_<arm>_<eye>_camera_optical_frame_epson
    imu:=true: /imu/data (frame imu_link), cuVSLAM VIO mode
Outputs: map -> odom -> base_link TF (publish_tf:=false to disable), /visual_slam/tracking/odometry, ...
"""

import os

import launch
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode

IMU_FRAME = 'imu_link'

# Stereo pairs in camera index order; num_cameras takes the first num_cameras / 2.
# Order matters: cuVSLAM treats cameras (0,1), (2,3), ... as stereo pairs.
ARMS = ['front', 'rear', 'left', 'right']
NUM_CAMERAS = ['2', '4', '6', '8']

# T_imu_cam per resolution and camera: (x, y, z, qx, qy, qz, qw) of the camera optical frame in the
# Epson IMU frame (x forward, y left, z up) = inverse of the Kalibr T_cam_imu.
# cuVSLAM (rectified_images) needs the 2 cameras of a pair with the exact same rotation: the ZED images
# are rectified, so right = left rotation, shifted by the Kalibr baseline along the left optical x axis
# (<= 1 mm from the Kalibr right camera positions; Kalibr's ~0.03-0.11 deg left/right rotation is
# calibration noise on rectified images).
# Kalibr timeshift_cam_imu (left / right), not compensated by cuVSLAM:
#   svga:  front 4.74 / 4.79 ms, rear -2.35 / 0.27 ms, left 2.92 / 2.89 ms
#   1080p: front 3.09 / 3.28 ms, rear 4.56 / 4.58 ms, left 3.57 / 3.99 ms, right 4.82 / 5.09 ms
CAMERA_POSES = {
    # front_/rear_/left_cameras_imu.bag_0-camchain-imucam.yaml; baselines front 0.119326 m,
    # rear 0.119176 m, left 0.119269 m. The right ZED was not calibrated at SVGA: left ZED pose turned
    # 180 deg about the vertical axis x=0, y=0.01 of imu_link (CAD symmetry of the rig).
    'svga': {
        ('front', 'left'): (0.212808, 0.058007, 0.081465, -0.49609879, 0.503399112, -0.499256903, 0.50121639),
        ('front', 'right'): (0.212171, -0.061312, 0.08036, -0.49609879, 0.503399112, -0.499256903, 0.50121639),
        ('rear', 'left'): (-0.201752, -0.072387, 0.034756, -0.505409369, -0.492488832, 0.48838072, 0.513322893),
        ('rear', 'right'): (-0.197238, 0.046695, 0.03618, -0.505409369, -0.492488832, 0.48838072, 0.513322893),
        ('left', 'left'): (-0.074542, 0.218419, 0.081774, -0.702414107, 0.000562529, -0.001163425, 0.711767344),
        ('left', 'right'): (0.044726, 0.218127, 0.081873, -0.702414107, 0.000562529, -0.001163425, 0.711767344),
        ('right', 'left'): (0.074542, -0.198419, 0.081774, -0.000562529, -0.702414107, 0.711767344, 0.001163425),
        ('right', 'right'): (-0.044726, -0.198127, 0.081873, -0.000562529, -0.702414107, 0.711767344, 0.001163425),
    },
    # <arm>_cameras_imu_1080p.bag_0-camchain-imucam.yaml (all 4 ZED); baselines front and rear
    # 0.120197 m, left 0.119848 m, right 0.120150 m.
    '1080p': {
        ('front', 'left'): (0.242076, 0.054676, 0.0811, -0.49717892, 0.50349595, -0.49745386, 0.50184122),
        ('front', 'right'): (0.241843, -0.065514, 0.079814, -0.49717892, 0.50349595, -0.49745386, 0.50184122),
        ('rear', 'left'): (-0.2726, -0.040107, 0.082669, -0.49957127, -0.49993846, 0.4928755, 0.50750746),
        ('rear', 'right'): (-0.270884, 0.080063, 0.084471, -0.49957127, -0.49993846, 0.4928755, 0.50750746),
        ('left', 'left'): (-0.073175, 0.231045, 0.081887, -0.70247967, 0.00173901, -0.00300891, 0.71169533),
        ('left', 'right'): (0.046671, 0.230239, 0.082097, -0.70247967, 0.00173901, -0.00300891, 0.71169533),
        ('right', 'left'): (0.044062, -0.241657, 0.063742, -0.00368679, -0.71440975, 0.69971596, 0.00164164),
        ('right', 'right'): (-0.076085, -0.240748, 0.063404, -0.00368679, -0.71440975, 0.69971596, 0.00164164),
    },
}

IMAGE_TOPIC = '/zed_{arm}/zed_node/{eye}/gray/rect/image'
INFO_TOPIC = '/zed_{arm}/zed_node/{eye}/gray/rect/camera_info'

# sync_matching_threshold_ms when sync_ms is empty: the 2 images of one ZED share a stamp, several ZED
# streams are stamped independently
SYNC_MS_ONE_ZED = 5.0
SYNC_MS_MULTI_ZED = 20.0


def optical_frame(arm, eye):
    return f'zed_{arm}_{eye}_camera_optical_frame_epson'


def launch_setup(context):
    resolution = LaunchConfiguration('resolution').perform(context).strip().lower()
    if resolution not in CAMERA_POSES:
        raise RuntimeError(f"resolution must be one of {list(CAMERA_POSES)}, got '{resolution}'")
    camera_arg = LaunchConfiguration('cameras').perform(context).strip()
    if camera_arg:
        names = {a.strip() for a in camera_arg.split(',') if a.strip()}
        if not names or names - set(ARMS):
            raise RuntimeError(f"cameras must be a comma separated list of {ARMS}, got '{camera_arg}'")
        arms = [a for a in ARMS if a in names]
        num_cameras = str(2 * len(arms))
    else:
        num_cameras = LaunchConfiguration('num_cameras').perform(context).strip()
        if num_cameras not in NUM_CAMERAS:
            raise RuntimeError(f"num_cameras must be one of {NUM_CAMERAS}, got '{num_cameras}'")
        arms = ARMS[:int(num_cameras) // 2]
    poses = CAMERA_POSES[resolution]
    cameras = [(arm, eye) for arm in arms for eye in ('left', 'right')]

    use_imu = LaunchConfiguration('imu').perform(context).lower() == 'true'
    publish_tf = LaunchConfiguration('publish_tf').perform(context).lower() == 'true'
    sync_ms = LaunchConfiguration('sync_ms').perform(context).strip()
    sync_ms = float(sync_ms) if sync_ms else (SYNC_MS_ONE_ZED if len(arms) == 1 else SYNC_MS_MULTI_ZED)

    actions = [LogInfo(msg=f'[visual_slam] {resolution}, {num_cameras} cameras ({", ".join(arms)}), '
                           f'sync {sync_ms} ms, {"VIO" if use_imu else "multi camera"}')]

    actions += [
        Node(package='tf2_ros', executable='static_transform_publisher',
             name=f'epson_{IMU_FRAME}_to_zed_{arm}_{eye}',
             arguments=['--x', str(x), '--y', str(y), '--z', str(z),
                        '--qx', str(qx), '--qy', str(qy), '--qz', str(qz), '--qw', str(qw),
                        '--frame-id', IMU_FRAME, '--child-frame-id', optical_frame(arm, eye)])
        for arm, eye in cameras
        for x, y, z, qx, qy, qz, qw in [poses[(arm, eye)]]]

    remappings = []
    for idx, (arm, eye) in enumerate(cameras):
        remappings += [(f'visual_slam/image_{idx}', IMAGE_TOPIC.format(arm=arm, eye=eye)),
                       (f'visual_slam/camera_info_{idx}', INFO_TOPIC.format(arm=arm, eye=eye))]
    parameters = {
        # true when replaying a bag with ros2 bag play --clock
        'use_sim_time': LaunchConfiguration('use_sim_time').perform(context).lower() == 'true',
        'tracking_mode': 1 if use_imu else 0,  # 1: VIO (IMU fusion), 0: multi camera
        'num_cameras': len(cameras),
        'min_num_images': len(cameras),
        'camera_optical_frames': [optical_frame(arm, eye) for arm, eye in cameras],
        'base_frame': 'base_link',
        'odom_frame': 'odom',
        'map_frame': 'map',
        # false: odometry topics only, e.g. when basalt owns odom/base_link on TF
        'publish_odom_to_base_tf': publish_tf,
        'publish_map_to_odom_tf': publish_tf,
        'rectified_images': True,
        'enable_image_denoising': False,
        'sync_matching_threshold_ms': sync_ms,
        'image_jitter_threshold_ms': float(LaunchConfiguration('jitter_ms').perform(context)),
        'enable_slam_visualization': True,
        'enable_observations_view': True,
        'enable_landmarks_view': True,
    }
    if use_imu:
        remappings.append(('visual_slam/imu', LaunchConfiguration('imu_topic').perform(context)))
        parameters.update({
            'imu_frame': IMU_FRAME,
            # Epson G366 noise model, Kalibr <arm>_cameras_imu[_1080p].bag_0-imu.yaml (imu0, /imu/data,
            # update_rate 400 Hz = rate the noise model was identified at; same for all calibrations)
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
        DeclareLaunchArgument('resolution', default_value='1080p',
                              description=f'ZED grab resolution {list(CAMERA_POSES)}: selects the '
                                          'Kalibr camera-IMU extrinsics'),
        DeclareLaunchArgument('num_cameras', default_value='2',
                              description='2 (front), 4 (+ rear), 6 (+ left) or 8 (+ right): '
                                          'ZED X images, 2 per stereo pair'),
        DeclareLaunchArgument('cameras', default_value='',
                              description=f'ZED set, comma separated among {ARMS} (e.g. left,right); '
                                          'overrides num_cameras. Empty: num_cameras'),
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
        DeclareLaunchArgument('sync_ms', default_value='',
                              description='max stamp difference [ms] between the images of one frame '
                                          f'(empty: {SYNC_MS_ONE_ZED} for 1 ZED, {SYNC_MS_MULTI_ZED} for several)'),
        DeclareLaunchArgument('use_sim_time', default_value='false',
                              description='true: /clock time, to replay a bag (ros2 bag play --clock)'),
        DeclareLaunchArgument('jitter_ms', default_value='60.0',
                              description='max delta between frames before a warning '
                                          '(> 1000 / camera rate)'),
        robot_description,
        OpaqueFunction(function=launch_setup),
    ])
