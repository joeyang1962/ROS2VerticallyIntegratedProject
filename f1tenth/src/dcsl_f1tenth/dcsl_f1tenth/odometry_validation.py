# validate odometry model (system identification model) against recorded data in rosbag
# this node will cache 5 seconds of Odometry and vicon data, then publish two trajectories
# for users to visualize in rviz

import numpy as np
import pdb

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Quaternion, Point, PoseStamped, Vector3
from collections import deque
from math import acos,sin,cos
import matplotlib.pyplot as plt

from tf_transformations import quaternion_matrix

def to_sec(stamp):
    ''' from ROS2 msg time stamp to seconds '''
    return stamp.sec + stamp.nanosec*1e-9

class OdometryValidation(Node):
    def __init__(self):
        super().__init__('odometry_validation')
        # NOTE for debugging
        self.set_parameters([rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])
        self.get_logger().set_level(rclpy.logging.LoggingSeverity.DEBUG)


        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(Odometry, '/vicon/car_5', self.vicon_callback, 10)
        self.odom_visual_publisher = self.create_publisher(Path, '/odom_visual', 10)
        self.vicon_visual_publisher = self.create_publisher(Path, '/vicon_visual', 10)

        self.timer = self.create_timer(0.1, self.update_prediction)

        self.odom_hist = deque()
        self.vicon_hist = deque()
        self.duration = 2

        # flags that we have enough
        self.odom_hist_ready = False
        self.vicon_hist_ready = False
        self.longitudinal_scale_vec = []

    def odom_callback(self, msg):
        # due to unknown reasons, the time stamp for odom on f1tenth is wrong (likely because it had no access to internet)
        ts = self.get_clock().now().nanoseconds * 1e-9
        msg.header.stamp = self.get_clock().now().to_msg()
        sim_time = to_sec(msg.header.stamp)
        self.odom_hist.append(msg)
        while (len(self.odom_hist) > 0 and to_sec(self.odom_hist[0].header.stamp) < ts - self.duration - 1):
            self.odom_hist.popleft()
            self.odom_hist_ready = True
        #self.get_logger().debug(f' -> {sim_time} odom_callback, odom_hist len: {len(self.odom_hist)}')

        return

    def vicon_callback(self, msg):
        ts = self.get_clock().now().nanoseconds * 1e-9
        sim_time = to_sec(msg.header.stamp)
        self.vicon_hist.append(msg)
        while (len(self.vicon_hist)> 0 and to_sec(self.vicon_hist[0].header.stamp) < ts - self.duration - 1):
            self.vicon_hist.popleft()
            self.vicon_hist_ready = True
        #self.get_logger().debug(f' -> {sim_time} vicon_callback, current vicon_hist len: {len(self.vicon_hist)}')
        return

    def map(self, ia, ib, oa, ob, val):
        ''' linear interpolation '''
        if (val < ia or val > ib):
            self.get_logger().warning('extrapolating, this should not happen')
        return (val - ia)/(ib-ia)*(ob-oa) + oa

    def interpolate_pos(self, p0, p1, t0, t1, t):
        ''' use linear interpolation '''
        if (t < t0 or t > t1):
            self.get_logger().warning('extrapolating, this should not happen')
        t = (t-t0) / (t1-t0)
        result = Point()
        result.x = p0.x * t + p1.x * (1-t)
        result.y = p0.y * t + p1.y * (1-t)
        result.z = p0.z * t + p1.z * (1-t)
        return result

    def interpolate_vec3(self, p0, p1, t0, t1, t):
        ''' use linear interpolation '''
        if (t < t0 or t > t1):
            self.get_logger().warning('extrapolating, this should not happen')
        t = (t-t0) / (t1-t0)
        result = Vector3()
        result.x = p0.x * t + p1.x * (1-t)
        result.y = p0.y * t + p1.y * (1-t)
        result.z = p0.z * t + p1.z * (1-t)
        return result

    def interpolate_quat(self, q0, q1, t0, t1, t):
        ''' use spherical linear interpolation '''
        if (t < t0 or t > t1):
            self.get_logger().warning('extrapolating, this should not happen')
        t = (t-t0) / (t1-t0)
        if (False):
            # spherical interpolation, too numerically unstabe
            print( q0.w * q1.w + q0.x * q1.x +q0.y * q1.y +q0.z * q1.z )
            theta = acos( q0.w * q1.w + q0.x * q1.x +q0.y * q1.y +q0.z * q1.z )
            C0 = sin( (1-t) * theta ) / sin(theta)
            C1 = sin(t*theta) / sin(theta)
            result = Quaternion()
            result.w = C0 * q0.w + C1 * q1.w
            result.x = C0 * q0.x + C1 * q1.x
            result.y = C0 * q0.y + C1 * q1.y
            result.z = C0 * q0.z + C1 * q1.z
        else:
            result = Quaternion()
            result.w = q0.w * t + q1.w * (1-t)
            result.x = q0.x * t + q1.x * (1-t)
            result.y = q0.y * t + q1.y * (1-t)
            result.z = q0.z * t + q1.z * (1-t)
        return result

    def get_interpolated_odom_at_t(self, t, history):
        before = None
        after = None
        for i in range(len(history)):
            if (to_sec(history[i].header.stamp) < t):
                before = history[i]
            else:
                after = history[i]
                break
        if (after is None):
            after = history[-1]
        msg = Odometry()
        msg.header.stamp.sec = int(t)
        msg.header.stamp.nanosec = int((t - msg.header.stamp.sec) * 1e9)
        msg.header.frame_id = before.header.frame_id
        msg.child_frame_id = before.child_frame_id
        msg.pose.pose.position = self.interpolate_pos(before.pose.pose.position,
                after.pose.pose.position, to_sec(before.header.stamp), to_sec(after.header.stamp), t)
        msg.pose.pose.orientation = self.interpolate_quat(before.pose.pose.orientation,
                after.pose.pose.orientation, to_sec(before.header.stamp), to_sec(after.header.stamp), t)
        msg.twist.twist.linear = self.interpolate_vec3(before.twist.twist.linear,
                after.twist.twist.linear, to_sec(before.header.stamp), to_sec(after.header.stamp), t)
        msg.twist.twist.angular = self.interpolate_vec3(before.twist.twist.angular,
                after.twist.twist.angular, to_sec(before.header.stamp), to_sec(after.header.stamp), t)
        return msg

    def find_transformation(self, target, base):
        ''' get the transformation T that will lineup target to base, T @ p_target = p_base '''
        t_pos = target.pose.pose.position
        t_quat = target.pose.pose.orientation
        t_T = quaternion_matrix( (t_quat.x, t_quat.y, t_quat.z, t_quat.w) )
        t_T[0:3,3] = (t_pos.x, t_pos.y, t_pos.z)
        b_pos = base.pose.pose.position
        b_quat = base.pose.pose.orientation
        b_T = quaternion_matrix( (b_quat.x, b_quat.y, b_quat.z, b_quat.w) )
        b_T[0:3,3] = (b_pos.x, b_pos.y, b_pos.z)
        T = np.dot(b_T, np.linalg.inv(t_T))
        return T

    def update_prediction(self):
        # find playback point (PP), this is now - duration
        t = self.get_clock().now()
        # get true position from vicon, interpolate
        if (not self.vicon_hist_ready or not self.odom_hist_ready):
            self.get_logger().info('waiting ...')
            return
        time_margin = 0.1
        tt = np.linspace(t.nanoseconds*1e-9-self.duration-time_margin, t.nanoseconds*1e-9-time_margin)
        odom_vec = [self.get_interpolated_odom_at_t(x, self.odom_hist) for x in tt]
        vicon_vec = [self.get_interpolated_odom_at_t(x, self.vicon_hist) for x in tt]

        # publish vicon path
        path_msg = Path()
        path_msg.header.frame_id = "vicon"
        path_msg.header.stamp = t.to_msg()
        for msg in vicon_vec:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position = msg.pose.pose.position
            pose.pose.orientation = msg.pose.pose.orientation
            path_msg.poses.append(pose)
        self.vicon_visual_publisher.publish(path_msg)

        # play odometry path, converted to vicon frame, aligned at targectory beginning
        T = self.find_transformation(odom_vec[0], vicon_vec[0])

        path_msg = Path()
        path_msg.header.frame_id = "vicon"
        path_msg.header.stamp = t.to_msg()
        for msg in odom_vec:
            pose = PoseStamped()
            pose.header = msg.header
            pos = msg.pose.pose.position
            pos_homo = np.hstack([pos.x, pos.y, pos.z, 1])
            pose_trans = T @ pos_homo.T

            pose.pose.position.x = pose_trans[0]
            pose.pose.position.y = pose_trans[1]
            pose.pose.position.z = pose_trans[2]
            path_msg.poses.append(pose)
        self.odom_visual_publisher.publish(path_msg)

        # check longitudinal model, is encoder accurate
        s_vicon = 0.0
        last_point = vicon_vec[0].pose.pose.position
        for msg in vicon_vec:
            this_point = msg.pose.pose.position
            ds = (last_point.x-this_point.x)**2 + (last_point.y-this_point.y)**2 + (last_point.z-this_point.z)**2
            s_vicon += ds
            last_point = this_point
        s_odom = 0.0
        last_point = odom_vec[0].pose.pose.position
        for msg in odom_vec:
            this_point = msg.pose.pose.position
            ds = (last_point.x-this_point.x)**2 + (last_point.y-this_point.y)**2 + (last_point.z-this_point.z)**2
            s_odom += ds
            last_point = this_point
        s_odom /= 1.331
        ratio = s_odom/s_vicon
        if (ratio < 2 and ratio > 0.5):
            self.longitudinal_scale_vec.append(ratio)
            self.get_logger().info(f'{s_odom/s_vicon, np.mean(self.longitudinal_scale_vec), np.std(self.longitudinal_scale_vec)}')
        return

def main(args=None):
    rclpy.init(args=args)
    node = OdometryValidation()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

