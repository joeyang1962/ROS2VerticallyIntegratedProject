import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    slam_params_file = LaunchConfiguration('slam_params_file')

    declare_slam_params_file_cmd = DeclareLaunchArgument(
        'slam_params_file',
        default_value=os.path.join(get_package_share_directory("dcsl_f1tenth"),
                                   'config', 'exp_slam_online_sync.yaml'),
        description='Full path to the ROS2 parameters file to use for the slam_toolbox node')

    start_sync_slam_toolbox_node = Node(
        parameters=[
          slam_params_file,
        ],
        package='slam_toolbox',
        executable='sync_slam_toolbox_node',
        name='slam_toolbox',
        output='screen'
        )

    ld = LaunchDescription()

    ld.add_action(declare_slam_params_file_cmd)
    ld.add_action(start_sync_slam_toolbox_node)

    return ld
