"""Depth-camera SLAM odometry, standalone.

    ros2 launch dsim_slam slam.launch.py
    ros2 launch dsim_slam slam.launch.py use_sim_time:=true publish_tf:=true
    ros2 launch dsim_slam slam.launch.py cloud_topic:=/camera/depth/points imu_topic:=/imu

Not included by dsim_bringup's sim.launch.py on purpose: wiring /slam/odom
into the controller is an integration decision, and this file must also work
on a real drone, where there is no simulator to launch.

Needs: a PointCloud2 in an optical frame, an Imu in base_link, and a static
TF base_link -> <cloud frame> (realsense-ros publishes camera_link ->
camera_depth_optical_frame; something must publish base_link ->
camera_link, the vehicle's URDF on hardware).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    config = os.path.join(get_package_share_directory('dsim_slam'), 'config', 'slam.yaml')
    args = {
        'use_sim_time': 'false',
        'cloud_topic': '/camera/camera/depth/color/points',
        'imu_topic': '/drone/imu',
        'publish_tf': 'false',
        'judge_wide_normals': 'false',
    }
    decls = [DeclareLaunchArgument(k, default_value=v) for k, v in args.items()]
    overrides = {
        'use_sim_time': ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool),
        'cloud_topic': LaunchConfiguration('cloud_topic'),
        'imu_topic': LaunchConfiguration('imu_topic'),
        'publish_tf': ParameterValue(LaunchConfiguration('publish_tf'), value_type=bool),
        'judge_wide_normals': ParameterValue(LaunchConfiguration('judge_wide_normals'),
                                             value_type=bool),
    }
    node = Node(
        package='dsim_slam',
        executable='slam_node',
        name='dsim_slam',
        output='screen',
        parameters=[config, overrides],
    )
    return LaunchDescription(decls + [node])
