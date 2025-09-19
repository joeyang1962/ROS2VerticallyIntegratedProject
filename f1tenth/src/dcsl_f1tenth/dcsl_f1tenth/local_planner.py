import numpy as np
import os
import sys
import pickle
global BASEDIR
BASEDIR = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))

import rclpy
from rclpy.node import Node

from nav_msgs.msg import OccupancyGrid, Path, Odometry
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformListener, Buffer
from tf2_ros.transform_broadcaster import TransformBroadcaster
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
from autoware_auto_planning_msgs.msg import Trajectory, TrajectoryPoint


import tf_transformations
import math
from builtin_interfaces.msg import Time

from VipPathOptimization.MapTrack import MapTrack


class LocalPlanner(Node):
    ''' plan local trajectory '''
    def __init__(self):
        super().__init__('local_planner')

        # TODO put these as ros Parameters
        self.map_frame = 'sim_map'
        self.create_subscription(OccupancyGrid, '/sim_map', self.map_callback, 1)
        self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 1)

        # TF buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # for visualization in rviz
        self.visual_path_pub = self.create_publisher(Path, '/visual_path', 1)
        # for use by path tracking controller, include speed, heading
        self.planned_traj_pub = self.create_publisher(Trajectory, '/planner_traj', 1)

        # Timer to periodically publish path
        self.timer = self.create_timer(1/10, self.update_traj)
        self.map = None
        self.odom = None

        with open(os.path.join(BASEDIR,'asset/track.p'), 'rb') as f:
            self.track = pickle.load(f)

    def map_callback(self, msg: OccupancyGrid):
        self.map = msg
    def odom_callback(self, msg: Odometry):
        self.odom = msg

    def update_traj(self):
        if (self.odom is None):
            self.get_logger().info('odom not ready')
            return
        try:
            # Lookup transform from map to base_link
            now = self.get_clock().now()
            trans: TransformStamped = self.tf_buffer.lookup_transform(
                self.map_frame, 'ego_racecar/base_link', rclpy.time.Time() )

            # Extract position and orientation
            x = trans.transform.translation.x
            y = trans.transform.translation.y

            q = trans.transform.rotation
            _, _, yaw = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
            # pose: x: forward, y:left, z:up

            # For now, we’ll assume speed to a constant (e.g., 1.0). Replace as needed.
            speed = self.odom.twist.twist.linear.x

            traj_points = self.plan(x, y, yaw, speed)

            # Convert to autoware_auto_planning_msgs/Trajectory
            traj_msg = Trajectory()
            traj_msg.header.frame_id = self.map_frame
            traj_msg.header.stamp = self.get_clock().now().to_msg()
            for px, py, heading, speed in traj_points:
                point = TrajectoryPoint()
                point.pose.position.x = px
                point.pose.position.y = py
                qx, qy, qz, qw = tf_transformations.quaternion_from_euler(0, 0, heading, axes='sxyz')
                point.pose.orientation.x = qx
                point.pose.orientation.y = qy
                point.pose.orientation.z = qz
                point.pose.orientation.w = qw
                point.longitudinal_velocity_mps = speed
                traj_msg.points.append(point)

            self.planned_traj_pub.publish(traj_msg)

            # Convert to nav_msgs/Path
            path_msg = Path()
            path_msg.header.frame_id = self.map_frame
            path_msg.header.stamp = self.get_clock().now().to_msg()
            for px, py, heading, speed in traj_points:
                pose = PoseStamped()
                pose.header = path_msg.header
                pose.pose.position.x = px
                pose.pose.position.y = py
                pose.pose.position.z = 0.0
                pose.pose.orientation.w = 1.0  # Default orientation
                path_msg.poses.append(pose)
            self.visual_path_pub.publish(path_msg)

        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f'Failed to get transform: {e}')

    def plan(self, x, y, heading, speed):
        """
        Given current vehicle state, provide planning trajectory
        Should return a list of (x, y, heading, speed) tuples representing the trajectory.
        The trajectory does not have to start at vehicle current location
        """
        dist = np.abs(self.track.r[:,0] - x) + np.abs(self.track.r[:,1] - y)
        N = self.track.r.shape[0]
        min_idx = (np.argmin(dist) + N - 25) % N
        if (min_idx + 500 > self.track.r.shape[0]):
            remain = 500 - (self.track.r.shape[0] - min_idx) + 1
            xx = np.hstack([ self.track.r[min_idx:-1,0], self.track.r[:remain,0] ])
            yy = np.hstack([ self.track.r[min_idx:-1,1], self.track.r[:remain,1] ])
            vv = np.hstack([ self.track.vv[min_idx:-1], self.track.vv[:remain] ])
            hh = np.hstack([ self.track.raceline_headings[min_idx:-1], self.track.raceline_headings[:remain] ])
            traj = np.vstack([xx,yy,hh, vv]).T
            return traj
        else:
            xx = self.track.r[min_idx:min_idx+500,0]
            yy = self.track.r[min_idx:min_idx+500,1]
            vv = self.track.vv[min_idx:min_idx+500]
            hh = self.track.raceline_headings[min_idx:min_idx+500]
            traj = np.vstack([xx,yy,hh, vv]).T
            return traj


def main(args=None):
    rclpy.init(args=args)
    node = LocalPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

