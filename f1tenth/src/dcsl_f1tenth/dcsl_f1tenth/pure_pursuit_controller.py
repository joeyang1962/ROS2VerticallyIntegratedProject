import numpy as np
import os
import sys
import pickle
global BASEDIR
BASEDIR = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))

import rclpy
from rclpy.node import Node
from time import time

from geometry_msgs.msg import PoseStamped, TransformStamped, PointStamped
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry, Path
from tf2_ros import TransformListener, Buffer
from tf2_ros.transform_broadcaster import TransformBroadcaster
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
from autoware_auto_planning_msgs.msg import Trajectory, TrajectoryPoint


import tf_transformations
import math
from math import cos,sin
from scipy.interpolate import splprep, splev,CubicSpline,interp1d
from builtin_interfaces.msg import Time

class PurePursuitController(Node):
    ''' actuate throttle/steering to follow published trajectory'''
    def __init__(self):
        super().__init__('pure_pursuit_controller')
        # TODO put this in ros_param
        self.map_frame = 'sim_map'

        # TF buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(Trajectory, '/planner_traj', self.traj_callback, 1)
        # TODO we the control should update at the same frequency as /odom, but use corrected
        # pose from localization, we need to check the rate of localizied pose publication
        # I'm not sure if we can accomplish this with lookup_transoform now
        # for now we just use 50Hz
        self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 1)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 1)
        # for visualization in rviz
        self.visual_control_pub = self.create_publisher(PointStamped, '/visual_control', 1)
        self.timer = self.create_timer(1/50, self.update_control)  # 50 Hz
        self.planner_traj = None
        self.odom = None

    def traj_callback(self, msg: Trajectory):
        self.planner_traj = msg
    def odom_callback(self, msg: Odometry):
        self.odom = msg

    def update_control(self):
        if (self.planner_traj is None or self.odom is None):
            self.get_logger().info('planner_traj and odom not ready')
            return
        try:
            t0 = time()
            # Lookup transform from map to base_link
            now = rclpy.time.Time()
            trans: TransformStamped = self.tf_buffer.lookup_transform(
                self.map_frame, 'ego_racecar/base_link', now)
            x = trans.transform.translation.x
            y = trans.transform.translation.y

            q = trans.transform.rotation
            _, _, yaw = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
            # pose: x: forward, y:left, z:up
            speed = self.odom.twist.twist.linear.x

            # pure pursuit
            # find point closest to vehicle
            def decode(point):
                x = point.pose.position.x
                y = point.pose.position.y
                qx = point.pose.orientation.x
                qy = point.pose.orientation.y
                qz = point.pose.orientation.z
                qw = point.pose.orientation.w
                _, _, yaw = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
                speed = point.longitudinal_velocity_mps
                return (x,y,yaw,speed)

            traj = np.array([ decode(point) for point in self.planner_traj.points ])
            dist = np.abs(traj[:,0] - x) + np.abs(traj[:,1] - y)
            min_idx = np.argmin(dist)


            # NOTE very heuristic
            target_speed = np.min( [ traj[min_idx,3] * 0.3, 2.0])
            lookahead_dist = np.max( [1.0, speed*2] )
            max_idx = min_idx
            while (dist[max_idx] < lookahead_dist):
                max_idx += 1
            #raceline_s, u = splprep(r.T, u=ss,s=0,per=1)
            # NOTE very coarse
            # transform point from map frame(global) into car frame(local)
            local_x = cos(yaw) * (traj[max_idx,0]-x) + sin(yaw) * (traj[max_idx, 1] - y)
            local_y = -sin(yaw) *(traj[max_idx,0]-x) + cos(yaw) * (traj[max_idx, 1] - y)
            R = (local_x**2 + local_y**2) / ( 2 * local_y )
            wheelbase = 0.206
            steering = np.arctan(wheelbase/R)

            visual_msg = PointStamped()
            visual_msg.header
            visual_msg.header.frame_id = self.map_frame
            visual_msg.header.stamp = self.get_clock().now().to_msg()
            visual_msg.point.x = traj[max_idx,0]
            visual_msg.point.y = traj[max_idx,1]
            visual_msg.point.z = 0.0
            self.visual_control_pub.publish(visual_msg)

            ackermann_msg = AckermannDriveStamped()
            ackermann_msg.header.frame_id = ''
            ackermann_msg.header.stamp = self.get_clock().now().to_msg()
            ackermann_msg.drive.steering_angle = steering
            ackermann_msg.drive.speed = target_speed
            self.drive_pub.publish(ackermann_msg)

            dt = time() - t0
            self.get_logger().info(f'steering = {steering/np.pi*180:.2f} deg  v = {target_speed} freq = {1/dt:.1f}')

        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f'Failed to get transform: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

