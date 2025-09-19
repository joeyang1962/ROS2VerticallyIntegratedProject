from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='dcsl_f1tenth',
            executable='hand_follower',
            name='hand_follower',
            output='screen',
            parameters=[{
                # Topics
                'scan_topic': '/scan',
                # use the topic your stack subscribes to:
                'drive_topic': '/ackermann_cmd',

                # Behavior
                'engage_distance': 0.60,       # start moving only if something < 0.6 m
                'target_distance': 0.35,       # maintain this distance
                'kp_speed': 1.2,
                'kp_steer': 1.5,
                'max_speed': 0.30,
                'max_steering_angle': 0.34,
                'front_angle_window_deg': 90.0,
                'allow_reverse': False,
                'angle_offset_deg': 0.0,       # tweak if LiDAR 0° isn’t straight ahead
            }],
        ),
    ])
