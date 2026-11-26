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

Comparison with FAST-LIVO2 (output_frame:=livox_frame [odom_frame:=camera_init]):
    cuVSLAM tracks the URDF livox_frame (Livox Mid-360, child of base_link through the rig URDF) instead of
    base_link: the rig extrinsics are taken from TF livox_frame -> camera optical frames / imu_link, so the
    odometry is the pose of livox_frame in an odom frame that starts on livox_frame (identity at the first
    tracked image), the convention of FAST-LIVO2 (world = first lidar pose). /visual_slam/tracking/odometry
    then has child_frame_id livox_frame and is directly comparable with /lio_odom (evo / ov_eval).
    The odom -> livox_frame TF is never broadcast (livox_frame already has its URDF parent); map -> odom
    is still published with publish_tf:=true.

    align_frame:=livox_frame: instead of starting at identity, the odometry (and the SLAM pose) starts
    on the TF pose odom_frame -> livox_frame at the first tracked image, e.g. the FAST-LIVO2 pose with
    odom_frame:=camera_init when FAST-LIVO2 publishes camera_init -> ... -> livox_frame on TF. Same
    physical frame as output_frame, so the two trajectories share their start. cuVSLAM publishes nothing
    until that TF exists ("Waiting for TF ... to align the odometry").
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
# are rectified. Values copied from basalt_ros (gen_isaac_camera_poses.py), where both cameras of a ZED
# have the mean of the two Kalibr rotations and the right one sits exactly one Kalibr baseline along
# the common optical x axis, centred on the Kalibr pair (Kalibr's ~0.03-0.11 deg left/right rotation is
# calibration noise on rectified images). Basalt, OKVIS2-X and cuVSLAM use the same numbers.
# The right ZED was not calibrated at SVGA: its basalt_ros pose is the left ZED one turned 180 deg about
# the vertical axis x=0, y=0.01 of imu_link (CAD symmetry of the rig).
# Kalibr timeshift_cam_imu (left / right), not compensated by cuVSLAM:
#   svga:  front 3.05 / 3.46 ms, rear 3.46 / 3.13 ms, left 2.92 / 2.89 ms
#   1080p: front 3.09 / 3.28 ms, rear 4.56 / 4.58 ms, left 3.57 / 3.99 ms, right 4.82 / 5.09 ms
CAMERA_POSES = {
    # GENERATED by amiad_nav/scripts/tools/gen_isaac_camera_poses.py from
    # basalt_ros/config/<res>/zedx_4stereo_calib_zed_real.json (T_imu_cam, same calibration
    # as Basalt and OKVIS2-X): do not edit by hand.
    'svga': {
        ('front', 'left'): (0.210156983, 0.052740848, 0.077318018, -0.497475674, 0.504641189, -0.497615781, 0.500233704),
        ('front', 'right'): (0.209611017, -0.066761848, 0.076149982, -0.497475674, 0.504641189, -0.497615781, 0.500233704),
        ('rear', 'left'): (-0.237718300, -0.062693018, 0.067560739, -0.499944339, -0.503870247, 0.493643222, 0.502480648),
        ('rear', 'right'): (-0.237135700, 0.057116018, 0.069092261, -0.499944339, -0.503870247, 0.493643222, 0.502480648),
        ('left', 'left'): (-0.074541845, 0.218479901, 0.081742644, -0.702462557, 0.000620283, -0.001289166, 0.711719263),
        ('left', 'right'): (0.044726845, 0.218157099, 0.081853356, -0.702462557, 0.000620283, -0.001289166, 0.711719263),
        ('right', 'left'): (0.074541845, -0.198479901, 0.081742644, -0.000620283, -0.702462557, 0.711719263, 0.001289166),
        ('right', 'right'): (-0.044726845, -0.198157099, 0.081853356, -0.000620283, -0.702462557, 0.711719263, 0.001289166),
    },
    '1080p': {
        ('front', 'left'): (0.241934176, 0.054676443, 0.081090411, -0.497372003, 0.503256003, -0.497339278, 0.502004113),
        ('front', 'right'): (0.241786824, -0.065513443, 0.079822589, -0.497372003, 0.503256003, -0.497339278, 0.502004113),
        ('rear', 'left'): (-0.272457918, -0.040108946, 0.082659091, -0.499333835, -0.500135220, 0.493036344, 0.507391020),
        ('rear', 'right'): (-0.270828082, 0.080061946, 0.084479909, -0.499333835, -0.500135220, 0.493036344, 0.507391020),
        ('left', 'left'): (-0.073173940, 0.231463178, 0.081779272, -0.702504071, 0.001368680, -0.002707625, 0.711673257),
        ('left', 'right'): (0.046672940, 0.230770822, 0.082001728, -0.702504071, 0.001368680, -0.002707625, 0.711673257),
        ('right', 'left'): (0.044065393, -0.241319250, 0.063708115, -0.003488685, -0.714299822, 0.699829761, 0.001377615),
        ('right', 'right'): (-0.076082393, -0.240488750, 0.063357885, -0.003488685, -0.714299822, 0.699829761, 0.001377615),
    },
}

IMAGE_TOPIC = '/zed_{arm}/zed_node/{eye}/gray/rect/image'
INFO_TOPIC = '/zed_{arm}/zed_node/{eye}/gray/rect/camera_info'

# sync_matching_threshold_ms when sync_ms is empty: the 2 images of one ZED share a stamp, several ZED
# streams are stamped independently
SYNC_MS_ONE_ZED = 5.0
SYNC_MS_MULTI_ZED = 20.0


def parse_overrides(text, parameters):
    """'name=value;name=value' -> {name: value} (YAML-typed); an int given for a float parameter of
    `parameters` becomes a float (ROS 2 rejects an int for a double parameter)."""
    import yaml
    out = {}
    for item in [i.strip() for i in text.replace(',', ';').split(';') if i.strip()]:
        if '=' not in item:
            raise RuntimeError(f"overrides: expected name=value, got '{item}'")
        name, value = (x.strip() for x in item.split('=', 1))
        value = yaml.safe_load(value)
        if isinstance(parameters.get(name), float) and isinstance(value, int) and not isinstance(value, bool):
            value = float(value)
        out[name] = value
    return out


def optical_frame(arm, eye, suffix='_epson'):
    return f'zed_{arm}_{eye}_camera_optical_frame{suffix}'


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
    # optical frame names zed_<arm>_<eye>_camera_optical_frame<suffix>: a bag recorded while this
    # launch ran has these static TFs in /tf_static (calibration of that time); a replay with the
    # same names mixes them with the current CAMERA_POSES (cuVSLAM: "Rectified stereo camera mode
    # requires rectified stereo cameras"), so replays use another suffix
    suffix = LaunchConfiguration('frame_suffix').perform(context).strip()
    publish_tf = LaunchConfiguration('publish_tf').perform(context).lower() == 'true'
    sync_ms = LaunchConfiguration('sync_ms').perform(context).strip()
    sync_ms = float(sync_ms) if sync_ms else (SYNC_MS_ONE_ZED if len(arms) == 1 else SYNC_MS_MULTI_ZED)
    output_frame = LaunchConfiguration('output_frame').perform(context).strip() or 'base_link'
    odom_frame = LaunchConfiguration('odom_frame').perform(context).strip() or 'odom'
    # a frame of the URDF other than base_link (e.g. livox_frame) already has a parent: cuVSLAM must not
    # broadcast odom -> <output_frame> on TF, the pose only goes to the odometry topics
    publish_odom_tf = publish_tf and output_frame == 'base_link'
    align_frame = LaunchConfiguration('align_frame').perform(context).strip()
    if align_frame and publish_odom_tf:
        # odom -> align_frame would go through cuVSLAM's own odom -> base_link TF, which it only
        # publishes once aligned: it would wait forever
        raise RuntimeError(f'align_frame:={align_frame} needs the TF {odom_frame} -> {align_frame} from '
                           f'another estimator: use publish_tf:=false or output_frame:=livox_frame')

    actions = [LogInfo(msg=f'[visual_slam] {resolution}, {num_cameras} cameras ({", ".join(arms)}), '
                           f'sync {sync_ms} ms, {"VIO" if use_imu else "multi camera"}, '
                           f'pose of {output_frame} in {odom_frame}'
                           + ('' if publish_odom_tf or not publish_tf else
                              f' (no TF {odom_frame} -> {output_frame}: frame of the URDF)')
                           + (f', start aligned on TF {odom_frame} -> {align_frame}' if align_frame else ''))]

    actions += [
        Node(package='tf2_ros', executable='static_transform_publisher',
             name=f'epson_{IMU_FRAME}_to_zed_{arm}_{eye}',
             arguments=['--x', str(x), '--y', str(y), '--z', str(z),
                        '--qx', str(qx), '--qy', str(qy), '--qz', str(qz), '--qw', str(qw),
                        '--frame-id', IMU_FRAME, '--child-frame-id', optical_frame(arm, eye, suffix)])
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
        'camera_optical_frames': [optical_frame(arm, eye, suffix) for arm, eye in cameras],
        # rig frame of cuVSLAM: base_link, or livox_frame to compare with FAST-LIVO2 (see output_frame)
        'base_frame': output_frame,
        'odom_frame': odom_frame,
        'map_frame': 'map',
        # '': odometry starts at identity; else starts on the TF pose odom_frame -> align_frame
        'align_frame': align_frame,
        # false: odometry topics only, e.g. when basalt owns odom/base_link on TF
        'publish_odom_to_base_tf': publish_odom_tf,
        'publish_map_to_odom_tf': publish_tf,
        'rectified_images': True,
        'enable_image_denoising': True,
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

    # overrides: cuVSLAM parameters replacing the ones above (parameter sweeps, record_bag.sh -p)
    overrides = LaunchConfiguration('overrides').perform(context).strip().strip('\'"')
    if overrides and overrides.lower() != 'none':
        changes = parse_overrides(overrides, parameters)
        parameters.update(changes)
        actions.append(LogInfo(msg=f'[cuVSLAM] overrides {changes}'))

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
        DeclareLaunchArgument('output_frame', default_value='livox_frame',
                              description='frame whose pose cuVSLAM estimates: base_link, or livox_frame '
                                          '(URDF Livox Mid-360) to compare with FAST-LIVO2 /lio_odom; '
                                          'other than base_link: no odom -> output_frame TF'),
        DeclareLaunchArgument('odom_frame', default_value='odom',
                              description='odometry frame name (e.g. camera_init, the FAST-LIVO2 world, '
                                          'with output_frame:=livox_frame)'),
        DeclareLaunchArgument('align_frame', default_value='',
                              description='e.g. livox_frame: start the odometry on the TF pose odom_frame -> '
                                          'align_frame published by another estimator (FAST-LIVO2 with '
                                          'odom_frame:=camera_init); empty: start at identity'),
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
        DeclareLaunchArgument('frame_suffix', default_value='_epson',
                              description='camera optical frames zed_<arm>_<eye>_camera_optical_frame<suffix>; '
                                          'bag replays: another one (e.g. _epson_replay), the bag '
                                          '/tf_static already has the _epson ones of the recording'),
        DeclareLaunchArgument('overrides', default_value='',
                              description='cuVSLAM node parameters replacing the launch ones, '
                                          '"name=value;..." e.g. "multicam_mode=2;'
                                          'enable_ground_constraint_in_odometry=true" (none: no change)'),
        DeclareLaunchArgument('jitter_ms', default_value='70.0',
                              description='max delta between frames before a warning '
                                          '(> 1000 / camera rate)'),        
        robot_description,
        OpaqueFunction(function=launch_setup),
    ])
