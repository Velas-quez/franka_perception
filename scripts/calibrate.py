import os

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Find the zed_wrapper package
    zed_wrapper_dir = FindPackageShare('zed_wrapper').find('zed_wrapper')
    zed_launch_file = os.path.join(
        zed_wrapper_dir, 'launch', 'zed_camera.launch.py')

    # ZED Right Camera (zedr)
    zedr = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(zed_launch_file),
        launch_arguments={
            'camera_name': 'zedr',
            'camera_model': 'zed2',
            'serial_number': '21177909',
            'ros_params_override_path':
                'src/action/configs/custom_zed_configs_right.yaml',
            'publish_urdf': 'true',
            'publish_tf': 'false',
            'publish_map_tf': 'false',
            'publish_imu_tf': 'false'
        }.items()
    )

    right_camera_tansform = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher',
        output='screen',
        arguments=[
            '--x', '0.8490203511285661',
            '--y', '0.5174056515518837',
            '--z', '0.5714417397720613',
            '--qx', '0.37194401469152666',
            '--qy', '0.07157716575364947',
            '--qz', '-0.9172819763062685',
            '--qw', '0.12299648458996869',
            '--frame-id', 'fr3_link0',
            '--child-frame-id', 'zedr_camera_link'
        ])


    # ZED Left Camera (zedl)
    zedl = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(zed_launch_file),
        launch_arguments={
            'camera_name': 'zedl',
            'camera_model': 'zed2',
            'serial_number': '29934236',
            'ros_params_override_path':
                'src/action/configs/custom_zed_configs_left.yaml',
            'publish_urdf': 'true',
            'publish_tf': 'false',
            'publish_map_tf': 'false',
            'publish_imu_tf': 'false'
        }.items()
    )

    left_camera_transform = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher_2',
        output='screen',
        arguments=[
            '--x', '0.14701289805698045',
            '--y', '-0.49165521178756444',
            '--z', '0.5270105802649703',
            '--qx', '-0.13145696353538433',
            '--qy', '0.40233828143381567',
            '--qz', '0.30943671064551204',
            '--qw', '0.8515232798554752',
            '--frame-id', 'fr3_link0',
            '--child-frame-id', 'zedl_camera_link'
        ])

    return LaunchDescription([zedr, zedl,
        right_camera_tansform, left_camera_transform])