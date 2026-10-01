"""The simulated D435 depth camera, on its own.

    ros2 launch dsim_depthcam depthcam.launch.py
    ros2 launch dsim_depthcam depthcam.launch.py world:=fence3d
    ros2 launch dsim_depthcam depthcam.launch.py world:=wall \
        landmarks_file:=/ws/config/landmarks_example.yaml pitch_deg:=10 noise:=false

Not part of sim.launch.py: it needs only /drone/truth and /clock, so it runs
beside a simulator that is already up. world:= picks the referee's obstacle
file, config/obstacles_<world>.yaml -- the SAME file the referee is given, so
the camera sees the obstacles exactly where they are scored, on the same
scenario clock (scenario.start_s comes from that file). A world with no
obstacle file renders the ground and the landmarks only.

Topics and frames default to what realsense-ros 4.x publishes for a D435
(namespace `camera`, name `camera`); see docs/INTERFACE.md.
"""
import math
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

WS = os.environ.get('DSIM_WS', '/ws')
CONFIG_DIR = os.path.join(WS, 'config')


def _setup(context):
    arg = lambda name: LaunchConfiguration(name).perform(context)  # noqa: E731
    params = [{
        'use_sim_time': arg('use_sim_time').lower() == 'true',
        'landmarks_file': arg('landmarks_file'),
        'rate_hz': float(arg('rate_hz')),
        'mount.x': float(arg('mount_x')),
        'mount.pitch_rad': math.radians(float(arg('pitch_deg'))),
        'noise.enable': arg('noise').lower() == 'true',
        'noise.dropout': float(arg('dropout')),
        'noise.seed': int(arg('seed')),
    }]
    world = arg('world')
    obstacles_file = os.path.join(CONFIG_DIR, f'obstacles_{world}.yaml')
    if world and os.path.exists(obstacles_file):
        params.append(obstacles_file)
    elif world:
        print(f'[depthcam.launch] no {obstacles_file}: ground and landmarks only')
    return [Node(
        package='dsim_depthcam', executable='depthcam_node', name='dsim_depthcam',
        output='screen', parameters=params,
    )]


def generate_launch_description():
    args = [
        ('world', '', 'scenario whose config/obstacles_<world>.yaml to render; empty = none'),
        ('landmarks_file', '', 'static landmarks YAML; empty = none'),
        ('use_sim_time', 'true', 'stamp and pace on /clock'),
        ('rate_hz', '30.0', 'frame rate in sim time'),
        ('mount_x', '0.10', 'camera_link forward of base_link, m'),
        ('pitch_deg', '0.0', 'mount pitch, positive DOWN'),
        ('noise', 'true', 'stereo depth noise on/off'),
        ('dropout', '0.0', 'fraction of valid pixels dropped'),
        ('seed', '1', 'noise seed; 0 = fresh every run'),
    ]
    return LaunchDescription(
        [DeclareLaunchArgument(n, default_value=d, description=h) for n, d, h in args]
        + [OpaqueFunction(function=_setup)])
