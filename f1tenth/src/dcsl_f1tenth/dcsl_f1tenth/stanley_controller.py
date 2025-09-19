# TODO instead of using discretized raceline, maybe just send in a polynomial
import pdb
import numpy as np
import os
import sys
import pickle
global BASEDIR
BASEDIR = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from time import time,sleep

from geometry_msgs.msg import PoseStamped, TransformStamped, PointStamped, PoseWithCovarianceStamped
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformListener, Buffer
from tf2_ros.transform_broadcaster import TransformBroadcaster
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
from autoware_auto_planning_msgs.msg import Trajectory, TrajectoryPoint

from scipy.interpolate import splprep, splev,CubicSpline,interp1d
from scipy.optimize import minimize


import tf_transformations
from math import cos,sin,atan2,degrees,radians,atan,isnan
from scipy.interpolate import splprep, splev,CubicSpline,interp1d
from builtin_interfaces.msg import Time
from VipPathOptimization.MapTrack import MapTrack
import bisect

import jax
import jax.numpy as jnp

from TimeUtil import TimeUtil
tu = TimeUtil(True)
DEBUG = False

class StanleyController(Node):
    def __init__(self):
        super().__init__('stanley_controller')
        # TODO put these in ros_param
        # tf frame name for the map/global frame, 'sim_map' for simulation amd 'map' for experiment
        self.map_frame = 'map'
        # when we adjust a local planned path that's violation lidar boundary, how much margin do we give the adjusted path
        self.adjust_margin = 0.5
        # when we adjust local trajectory, how far do we look ahead
        self.lookahead = 3.0
        # when determining left/right boundary from lidar point cloud, how far apart do consecutive lidar hit need to be
        # for us to consider the boundary is broken
        self.gap_threshold = 0.5

        self.last_error_ts = time()
        self.last_lateral_error = 0
        self.last_heading_error = 0
        # load assets
        with open(os.path.join(BASEDIR,'asset/track.p'), 'rb') as f:
            self.track = pickle.load(f)

        '''
        with open(os.path.join(BASEDIR,'asset','stanley_gains.p'), 'rb') as f:
            vv = np.linspace(1.0,6.0,600)
            gains = pickle.load(f)
            self.lateral_gain =  interp1d(vv, np.clip(gains[:,0],0,1e5))
            self.dlateral_gain = interp1d(vv, np.clip(gains[:,1],0,1e5))
            self.heading_gain =  interp1d(vv, np.clip(gains[:,2],0,1e5))
            self.dheading_gain = interp1d(vv, np.clip(gains[:,3],0,1e5))
        '''

        # TF buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            qos
        )
        self.create_subscription(Odometry, '/odom', self.odom_callback, 1)
        self.create_subscription(PoseWithCovarianceStamped, '/initialpose', self.initial_pose_callback,1)

        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 1)
        # for visualization in rviz
        self.visual_control_pub = self.create_publisher(PointStamped, '/visual_control', 1)
        # for visualizing planned path in rviz
        self.visual_path_pub = self.create_publisher(Path, '/visual_path', 1)
        self.timer = self.create_timer(1/20, self.update_control)  # 50 Hz
        self.planner_traj = None
        self.odom = None
        self.scan = None
        self.guess_s = None

        if (DEBUG):
            self.fig, self.ax = plt.subplots()
            self.fig_lines = None
            self.ax.set_aspect('equal', adjustable='datalim')
        self.debug_dict = {'lateral':[], 'dlateral':[],'heading':[],'dheading':[]}

    def scan_callback(self, msg: LaserScan):
        self.scan = msg
    def odom_callback(self, msg: Odometry):
        self.odom = msg
    def initial_pose_callback(self, msg:PoseWithCovarianceStamped):
        x,y,heading,vx,vy,yaw_rate = self.get_current_states()
        self.guess_s = self.initial_guess((x,y))

    def get_current_states(self):
        # Lookup latest transform from global frame to base_link
        trans: TransformStamped = self.tf_buffer.lookup_transform(
            self.map_frame, 'base_link', rclpy.time.Time())
        x = trans.transform.translation.x
        y = trans.transform.translation.y
        q = trans.transform.rotation
        _, _, heading = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        # pose: x: forward, y:left, z:up
        # longitudinal speed
        vx = self.odom.twist.twist.linear.x
        vy = self.odom.twist.twist.linear.y
        yaw_rate = self.odom.twist.twist.angular.z
        return x,y,heading,vx,vy,yaw_rate

    def publish_adjusted_path(self, adjusted_raceline):
        path_msg = Path()
        path_msg.header.frame_id = self.map_frame
        path_msg.header.stamp = self.get_clock().now().to_msg()
        def traj_to_pose(u):
            r = np.array(splev(u,adjusted_raceline))
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = r[0]
            pose.pose.position.y = r[1]
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0  # Default orientation
            return pose
        uu = np.linspace(0,1)
        path_msg.poses = [traj_to_pose(u) for u in uu]
        self.visual_path_pub.publish(path_msg)
        return

    def publish_planned_path(self, min_idx):
        # publish "ADJUSTED" planned path
        tu.s('gather')
        N = self.track.r.shape[0]
        lookahead_steps = 40
        if (min_idx + lookahead_steps > self.track.r.shape[0]):
            remain = lookahead_steps - (self.track.r.shape[0] - min_idx) + 1
            xx = np.hstack([ self.track.r[min_idx:-1,0], self.track.r[:remain,0] ])
            yy = np.hstack([ self.track.r[min_idx:-1,1], self.track.r[:remain,1] ])
            vv = np.hstack([ self.track.vv[min_idx:-1], self.track.vv[:remain] ])
            hh = np.hstack([ self.track.raceline_headings[min_idx:-1], self.track.raceline_headings[:remain] ])
            traj = np.vstack([xx,yy,hh, vv]).T
        else:
            xx = self.track.r[min_idx:min_idx+lookahead_steps,0]
            yy = self.track.r[min_idx:min_idx+lookahead_steps,1]
            vv = self.track.vv[min_idx:min_idx+lookahead_steps]
            hh = self.track.raceline_headings[min_idx:min_idx+lookahead_steps]
            traj = np.vstack([xx,yy,hh, vv]).T
        tu.e('gather')
        # Convert to nav_msgs/Path
        path_msg = Path()
        path_msg.header.frame_id = self.map_frame
        path_msg.header.stamp = self.get_clock().now().to_msg()
        def traj_to_pose(traj_point):
            px, py, heading, speed = traj_point
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = px
            pose.pose.position.y = py
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0  # Default orientation
            return pose
        path_msg.poses = [traj_to_pose(p) for p in traj]
        self.visual_path_pub.publish(path_msg)

    def centerline_path(self, x0, car_s):
        # max distance between two consecutive points to be considered in a segment
        # this is scaled based on angle and range
        # d_limit = d_angle * range * nt
        nt = self.neighbor_threshold = 20.0

        scan = self.scan
        d_angle = (scan.angle_max - scan.angle_min) / (len(scan.ranges)-1)
        angles = np.linspace(scan.angle_min, scan.angle_max, len(scan.ranges))
        ranges = np.array(scan.ranges)
        mask = np.logical_and(ranges > 0.05, ranges < 3.0)
        ranges = ranges[mask]; angles = angles[mask]
        # -- identify segments, with thresholds
        xx_map = x0[0] + ranges * np.cos(angles + x0[2])
        yy_map = x0[1] + ranges * np.sin(angles + x0[2])
        points = np.stack((xx_map, yy_map), axis=1)
        gap = np.hstack([0,np.linalg.norm(np.diff(points,axis=0),axis=1)])
        biggest_split_angle = np.clip(angles[np.argmax(gap)], -radians(70), radians(70))

        #limit = nt * d_angle * ranges
        limit = 0.1
        valid = gap < limit
        # label segments [1,1,0,0,2,2,2], etc.
        starts = np.logical_and(valid , np.logical_not(np.roll(valid,1)))
        starts[0] = valid[0]
        labels = np.cumsum(starts)*valid
        total_labels = np.max(labels)

        # find left and right boundary
        # start searching from right side (angle= -np.pi/2), find longest qualifying segment
        # qualification: must START from right side
        # NOTE the idx for label is not the same as the actual label, since idx=0 means label=1, label=0 is unpicked points
        # it's ok if there's none (one side open)
        smallest_idx_for_label = [np.argmax(labels==i) for i in range(1,total_labels+1)]
        smallest_angle_for_label = angles[smallest_idx_for_label]
        # starts from right
        split_angle = biggest_split_angle
        right_qualified_labels = np.arange(1,total_labels+1)[smallest_angle_for_label < split_angle]
        right_angle_mask = angles < split_angle
        right_counts = np.array([np.sum(labels*right_angle_mask==i) for i in right_qualified_labels])
        if (len(right_qualified_labels) == 0):
            self.get_logger().warn(f'no right segment qualified')
            right_label = None
        else:
            # find right-most segment that is long enough, and qualified
            #smallest_angle_segment_idx = np.argmin(smallest_angle_for_label[right_qualified_labels-1])
            #right_label = right_qualified_labels[smallest_angle_segment_idx]
            right_label = right_qualified_labels[np.argmax(right_counts)]
            if (np.max(right_counts) < 100):
                self.get_logger().warn(f'no right segment long enough')
                right_label = None

        biggest_idx_for_label = [labels.shape[0] -1 - np.argmax(labels[::-1]==i) for i in range(1,total_labels+1)]
        biggest_angle_for_label = angles[biggest_idx_for_label]
        # start from left
        left_qualified_labels = np.arange(1,total_labels+1)[biggest_angle_for_label > split_angle]
        left_angle_mask = angles > split_angle
        left_counts = np.array([np.sum(labels*left_angle_mask==i) for i in left_qualified_labels])
        if (len(left_qualified_labels) == 0):
            self.get_logger().warn(f'no left segment qualified')
            left_label = None
        else:
            # find left-most segment that is long enough, and qualified
            #largest_angle_segment_idx = np.argmax(biggest_angle_for_label[left_qualified_labels-1])
            #left_label = left_qualified_labels[largest_angle_segment_idx]
            left_label = left_qualified_labels[np.argmax(left_counts)]
            if (np.max(left_counts) < 100):
                self.get_logger().warn(f'no left segment long enough')
                left_label = None

        # --- find centerline
        expected_track_width = 0.8
        xx_local = ranges * np.cos(angles)
        yy_local = ranges * np.sin(angles)
        right_line_xx_local = xx_local[labels==right_label]
        right_line_yy_local = yy_local[labels==right_label] + expected_track_width/2
        left_line_xx_local = xx_local[labels==left_label]
        left_line_yy_local = yy_local[labels==left_label] - expected_track_width/2

        centerline_xx_local = np.hstack([right_line_xx_local, left_line_xx_local])
        centerline_yy_local = np.hstack([right_line_yy_local, left_line_yy_local])

        idx_vec = np.argsort(centerline_xx_local)
        centerline_xx_local = centerline_xx_local[idx_vec]
        centerline_yy_local = centerline_yy_local[idx_vec]
        coeffs = np.polyfit(centerline_xx_local, centerline_yy_local, deg = 4)
        poly_centerline_xx_local = np.linspace(0, np.max(centerline_xx_local))
        poly_centerline_yy_local = np.poly1d(coeffs)(poly_centerline_xx_local)

        # visualize segments
        if (DEBUG):
            if (self.fig_lines is None):
                self.fig_lines = []
            for line in self.fig_lines:
                if line in self.ax.lines:
                    line.remove()
            self.fig_lines.append( self.ax.plot(x0[0],x0[1], 'ko')[0] )
            self.fig_lines.append( self.ax.plot(xx_map,yy_map, 'k*')[0] )
            if (right_label is not None):
                line_xx = xx_map[labels==right_label]
                line_yy = yy_map[labels==right_label]
                self.fig_lines.append( self.ax.plot(line_xx,line_yy, 'ro')[0] )
            if (left_label is not None):
                line_xx = xx_map[labels==left_label]
                line_yy = yy_map[labels==left_label]
                self.fig_lines.append( self.ax.plot(line_xx,line_yy, 'bo')[0] )

            poly_centerline_xx_map = x0[0] + poly_centerline_xx_local * np.cos(x0[2]) - poly_centerline_yy_local * np.sin(x0[2])
            poly_centerline_yy_map = x0[1] + poly_centerline_xx_local * np.sin(x0[2]) + poly_centerline_yy_local * np.cos(x0[2])
            self.fig_lines.append( self.ax.plot(poly_centerline_xx_map, poly_centerline_yy_map, '-')[0] )

            # plot split_angle
            self.fig_lines.append( self.ax.plot([x0[0], x0[0] + 1.0*np.cos(x0[2]+split_angle)], [x0[1], x0[1] + 1.0*np.sin(x0[2]+split_angle)], '-')[0] )

            #centerline_xx_map = x0[0] + centerline_xx_local * np.cos(x0[2]) - centerline_yy_local * np.sin(x0[2])
            #centerline_yy_map = x0[1] + centerline_xx_local * np.sin(x0[2]) + centerline_yy_local * np.cos(x0[2])
            #self.fig_lines.append( self.ax.plot(centerline_xx_map, centerline_yy_map, 'go')[0] )

            self.ax.relim()
            self.ax.autoscale_view()
            plt.draw()
            plt.pause(0.001)


        # do same for left side (angle = np.pi/2)
        # find centerline using both boundaries, with expected track half width
        # find centerline

    def adjust_path(self, x0, car_s):
        ''' x0: x,y,heading '''
        adjust_margin = self.adjust_margin
        lookahead = self.lookahead
        gap_threshold = self.gap_threshold

        tu.s('bdry')
        scan = self.scan
        angles = np.linspace(scan.angle_min, scan.angle_max, len(scan.ranges))
        ranges = np.array(scan.ranges)
        #mask = np.isfinite(ranges)
        mask = np.logical_and(ranges < 10.0, ranges > 0.05)
        ranges = ranges[mask]; angles = angles[mask]
        # transform to map frame
        xx = x0[0] + ranges * np.cos(angles + x0[2])
        yy = x0[1] + ranges * np.sin(angles + x0[2])
        points = np.stack((xx, yy), axis=1)
        left_idx = np.argmin(np.abs(angles - np.pi / 2))
        right_idx = np.argmin(np.abs(angles + np.pi / 2))
        assert(left_idx > right_idx)
        gap = np.linalg.norm(np.diff(points,axis=0),axis=1)
        max_gap_idx = np.argmax(gap)

        right_gap = np.roll(gap, -right_idx)
        right_end_idx = np.argmax(right_gap > gap_threshold)
        if (right_end_idx == 0):
            right_end_idx = (max_gap_idx - right_idx)%len(gap)
        right_boundary = np.roll(points,-right_idx, axis=0)[:right_end_idx-1]

        left_gap = np.roll(gap[::-1], -len(gap)+left_idx)
        left_end_idx = np.argmax(left_gap > gap_threshold)
        if (left_end_idx == 0):
            left_end_idx = ( - max_gap_idx + left_idx)%len(gap)
        left_boundary = np.roll(points[::-1], -len(gap)+left_idx, axis=0)[:left_end_idx]
        R_ccw = np.array([[0,-1],[1,0]])
        R_cw = np.array([[0,1],[-1,0]])

        def closest_point_on_spline(pt, tck, u_samples):
            spline_pts = np.array(splev(u_samples, tck)).T
            dists = np.linalg.norm(spline_pts - pt, axis=1)
            idx = np.argmin(dists)
            r = spline_pts[idx]
            return u_samples[idx], r

        left_spline, _ = splprep(left_boundary.T, s=0)
        right_spline, _ = splprep(right_boundary.T, s=0)
        tu.e('bdry')

        def adjust_point(point, dr):
            ss = np.linspace(0,1, 20)

            left_s, left_r = closest_point_on_spline(point, left_spline, ss)
            ref_to_bdry = left_r - point
            cross = np.cross(dr, ref_to_bdry)
            if (cross < adjust_margin):
                point = point + R_cw @ dr * (adjust_margin - cross)

            right_s, right_r = closest_point_on_spline(point, right_spline, ss)
            ref_to_bdry = right_r - point
            cross = np.cross(dr, ref_to_bdry)
            if (cross > -adjust_margin):
                point = point + R_ccw @ dr * (adjust_margin + cross)
            return point

        def get_adjustment(point, dr):
            ss = np.linspace(0,1, 50)
            adjustment = 0 # R_ccw @ dr -> positive

            left_s, left_r = closest_point_on_spline(point, left_spline, ss)
            ref_to_bdry = left_r - point
            cross = np.cross(dr, ref_to_bdry)
            if (cross < adjust_margin):
                adjustment = - (adjust_margin - cross)

            right_s, right_r = closest_point_on_spline(point, right_spline, ss)
            ref_to_bdry = right_r - point
            cross = np.cross(dr, ref_to_bdry)
            if (cross > -adjust_margin):
                adjustment =  (adjust_margin + cross)
            return adjustment

        ss = np.linspace(car_s, car_s+lookahead)
        path_point = np.array(splev(ss%self.track.raceline_len_m, self.track.raceline_s)).T
        d_path_point = np.diff(path_point, axis=0)
        d_path_point = d_path_point / np.linalg.norm(d_path_point, axis=1)[:,None]
        tu.s('pre-adjustment')
        adjustment = np.array([get_adjustment(p,dr) for p,dr in zip(path_point[:-1],d_path_point)])
        tu.e('pre-adjustment')
        # adjustment is like a sawtooth, we don't like that
        tu.s('convexify')
        for i in range(1,len(adjustment)-2):
            if (adjustment[i] > 0 and adjustment[i] < adjustment[i-1] and adjustment[i] < adjustment[i+1]):
                adjustment[i] = (adjustment[i-1] + adjustment[i+1])/2
            elif (adjustment[i] < 0 and adjustment[i] > adjustment[i-1] and adjustment[i] > adjustment[i+1]):
                adjustment[i] = (adjustment[i-1] + adjustment[i+1])/2
        tu.e('convexify')

        tu.s('post-adjustment')
        tck, u = splprep([adjustment], u=np.arange(len(adjustment)), s=1)
        adjustment = splev(u,tck)[0]
        adjusted_path_point = (path_point[:-1].T + R_ccw @ d_path_point.T * adjustment).T
        adjusted_raceline, _ = splprep(adjusted_path_point.T, s=0)
        tu.e('post-adjustment')

        '''
        pdb.set_trace()
        plt.plot(points[:,0], points[:,1])
        plt.plot(left_boundary[:,0], left_boundary[:,1], 'ro')
        plt.plot(right_boundary[:,0], right_boundary[:,1], 'go')
        plt.show()
        '''
        if (DEBUG):
            if (self.fig_lines is None):
                self.fig_lines = []
                self.fig_lines.append( self.ax.plot(points[:,0], points[:,1], 'k*')[0] )
                self.fig_lines.append( self.ax.plot(left_boundary[:,0], left_boundary[:,1], 'ro')[0] )
                self.fig_lines.append( self.ax.plot(right_boundary[:,0], right_boundary[:,1], 'bo')[0] )
                self.fig_lines.append( self.ax.plot(path_point[:,0], path_point[:,1], '--')[0] )
                self.fig_lines.append( self.ax.plot(adjusted_path_point[:,0], adjusted_path_point[:,1], '-')[0] )
            else:
                self.fig_lines[0].set_data(points[:,0], points[:,1] )
                self.fig_lines[1].set_data(left_boundary[:,0], left_boundary[:,1] )
                self.fig_lines[2].set_data(right_boundary[:,0], right_boundary[:,1] )
                self.fig_lines[3].set_data(path_point[:,0], path_point[:,1])
                self.fig_lines[4].set_data(adjusted_path_point[:,0], adjusted_path_point[:,1])
            self.ax.relim()
            self.ax.autoscale_view()
            plt.draw()
            plt.pause(0.001)

        return adjusted_raceline

    def update_control(self):
        if (self.odom is None):
            self.get_logger().info('odom not ready')
            return
        if (self.scan is None):
            self.get_logger().info('scan not ready')
            return
        try:
            tu.s()
            tu.s('prep')
            t0 = time()
            x,y,heading,vx,vy,yaw_rate = self.get_current_states()
            # add in a slight lookahead distance
            lookahead = 0.2
            x_lookahead = x + cos(heading) * lookahead
            y_lookahead = y + sin(heading) * lookahead
            coord = np.array((x_lookahead, y_lookahead))
            tu.e('prep')

            if (self.guess_s is None):
                self.guess_s = self.initial_guess(coord)

            d_fun = lambda u: np.linalg.norm(np.array(splev(u%self.track.raceline_len_m, self.track.raceline_s)).flatten() - np.array(coord))
            res = minimize(d_fun, x0 = self.guess_s, tol = 1e-5)
            self.guess_s = car_s = res.x[0] % self.track.raceline_len_m

            # adjust path for laser scan
            tu.s('path')
            # TODO check if reference path is worth adjusting, if too far, ignore and just follow centerline
            adjusted_raceline = self.centerline_path((x,y,heading), car_s)
            adjusted_raceline = self.track.raceline_s
            local_s = car_s

            '''
            try:
                #adjusted_raceline = self.adjust_path((x,y,heading), car_s)
                adjusted_raceline = self.adjust_path_wrapper((x,y,heading), car_s)
                local_s = 0
            except TypeError as e:
                # for when there's not enough points to fit spline
                adjusted_raceline = self.track.raceline_s
                local_s = car_s
                self.get_logger().warn(e)
                pdb.set_trace()
            '''
            tu.e('path')

            # closest point on ref path
            path_point = np.array(splev(local_s, adjusted_raceline)).flatten()

            r =   np.array(splev(local_s, adjusted_raceline, der=0))
            dr =  np.array(splev(local_s, adjusted_raceline, der=1))
            ddr = np.array(splev(local_s, adjusted_raceline, der=2))
            _norm = lambda x:np.linalg.norm(x)
            dr_norm = _norm(dr)
            # 1/R
            curvature = 1.0/(dr_norm**3/(dr_norm**2*_norm(ddr)**2 - np.sum(dr*ddr,axis=0)**2)**0.5)

            tu.s('control')
            path_tangent = dr.flatten()
            path_orientation = atan2(path_tangent[1], path_tangent[0])
            path_to_car = (x_lookahead - path_point[0], y_lookahead - path_point[1])
            # positive error require negative steering (right) for correction
            lateral_error = np.cross(path_tangent/dr_norm, path_to_car).item()
            heading_error = (heading - path_orientation + np.pi) %(2*np.pi) - np.pi

            error_ts = time()
            d_lateral_error_dt = (lateral_error - self.last_lateral_error) / (error_ts - self.last_error_ts)
            d_heading_error_dt = (heading_error - self.last_heading_error) / (error_ts - self.last_error_ts)

            self.last_error_ts = time()
            self.last_lateral_error = lateral_error
            self.last_heading_error = heading_error

            target_yaw_rate = vx * curvature

            # apply gains found with pole placement in scripts/dynamics.py
            #vx_gain = np.clip(vx,1.0,6.0)
            #steering_feedback = self.lateral_gain(vx_gain) * lateral_error + self.dlateral_gain(vx_gain) * d_lateral_error_dt + self.heading_gain(vx_gain) * heading_error + self.dheading_gain(vx_gain) * d_heading_error_dt
            # slow speed
            #steering_feedback = 0.0 * lateral_error + 0.0 * d_lateral_error_dt + 0.2 * heading_error + 0.0 * d_heading_error_dt
            steering_feedback = 0.3 * lateral_error + 0.00 * d_lateral_error_dt + 0.3 * heading_error + 0.0 * d_heading_error_dt
            # use kinematic model for feed forward steering
            wheelbase = 0.32
            steering_feedforward = (path_orientation - heading)

            # "traditional" control
            #steering = (path_orientation - heading) - ( np.clip( lateral_error * self.Pfun(abs(speed)) , -radians(10), radians(10)))
            steering = steering_feedforward - steering_feedback
            steering = (steering+np.pi)%(2*np.pi) -np.pi
            tu.e('control')

            # TODO dynamically do 3 pass alg
            target_speed = float(self.track.raceline_speed_s(car_s))
            target_speed = min(0.5,target_speed) # FIXME

            print(f'vx={vx:.1f}->{target_speed:.1f} steering: {degrees(steering):.2f}\t lateral/dlateral: {lateral_error:.2f}/{d_lateral_error_dt:.2f}, heading/dheading: {heading_error:.2f}/{d_heading_error_dt:.2f} deg freq = {1/(time()-t0):.0f} {target_speed}')
            self.debug_dict['lateral'].append(lateral_error)
            self.debug_dict['dlateral'].append(d_lateral_error_dt)
            self.debug_dict['heading'].append(heading_error)
            self.debug_dict['dheading'].append(d_heading_error_dt)

            tu.s('publish ctrl')
            # publish command
            ackermann_msg = AckermannDriveStamped()
            ackermann_msg.header.frame_id = ''
            ackermann_msg.header.stamp = self.get_clock().now().to_msg()
            ackermann_msg.drive.steering_angle = steering
            ackermann_msg.drive.speed = target_speed
            #FIXME
            #self.drive_pub.publish(ackermann_msg)
            tu.e('publish ctrl')

            # publish debug visualization
            tu.s('publish debug')
            visual_msg = PointStamped()
            visual_msg.header.frame_id = self.map_frame
            visual_msg.header.stamp = self.get_clock().now().to_msg()
            visual_msg.point.x = path_point[0]
            visual_msg.point.y = path_point[1]
            visual_msg.point.z = 0.0
            self.visual_control_pub.publish(visual_msg)
            tu.e('publish debug')

            ss = np.linspace(0,self.track.raceline_len_m,self.track.discretized_raceline_len)
            #self.publish_planned_path(bisect.bisect_left(ss,self.guess_s))
            self.publish_adjusted_path(adjusted_raceline)
            tu.e()


        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f'Failed to get transform: {e}')

    def initial_guess(self,coord):
        ss = np.linspace(0,self.track.raceline_len_m,self.track.discretized_raceline_len)
        r0 = coord
        dist_vec = np.linalg.norm(self.track.r - r0, axis=1)
        idx = np.argmin(dist_vec)
        return ss[idx]

    def stop(self):
        # publish zero speed command
        ackermann_msg = AckermannDriveStamped()
        ackermann_msg.header.frame_id = ''
        ackermann_msg.header.stamp = self.get_clock().now().to_msg()
        ackermann_msg.drive.steering_angle = 0.0
        ackermann_msg.drive.speed = 0.0
        self.drive_pub.publish(ackermann_msg)
        self.get_logger().info('send stop command')
        return

    def adjust_path_wrapper(self, x0, car_s):
        adjust_margin = self.adjust_margin
        lookahead = self.lookahead
        gap_threshold = self.gap_threshold

        scan = self.scan
        angles = np.linspace(scan.angle_min, scan.angle_max, len(scan.ranges))
        ranges = np.array(scan.ranges)
        #mask = np.isfinite(ranges)
        mask = np.logical_and(ranges < 10.0, ranges > 0.05)
        ranges = ranges[mask]; angles = angles[mask]
        ss = np.linspace(car_s, car_s+lookahead)
        path_point = np.array(splev(ss%self.track.raceline_len_m, self.track.raceline_s)).T
        adjustment, left_boundary, right_boundary = jax_adjust_path(gap_threshold,adjust_margin, lookahead, angles, ranges, path_point, x0, car_s)
        adjustment = adjustment.block_until_ready()
        left_boundary = left_boundary.block_until_ready()
        right_boundary = right_boundary.block_until_ready()

        tck, u = splprep([adjustment], u=np.arange(len(adjustment)), s=1)
        adjustment = splev(u,tck)[0]
        R_ccw = np.array([[0,-1],[1,0]])
        d_path_point = np.diff(path_point, axis=0)
        d_path_point = d_path_point / np.linalg.norm(d_path_point, axis=1)[:,None]
        adjusted_path_point = (path_point[:-1].T + R_ccw @ d_path_point.T * adjustment).T
        adjusted_raceline, _ = splprep(adjusted_path_point.T, s=0)

        '''
        pdb.set_trace()
        plt.plot(points[:,0], points[:,1])
        plt.plot(left_boundary[:,0], left_boundary[:,1], 'ro')
        plt.plot(right_boundary[:,0], right_boundary[:,1], 'go')
        plt.show()
        '''
        if (DEBUG):
            xx = x0[0] + ranges * np.cos(angles + x0[2])
            yy = x0[1] + ranges * np.sin(angles + x0[2])
            points = np.stack((xx, yy), axis=1)
            if (self.fig_lines is None):
                self.fig_lines = []
                self.fig_lines.append( self.ax.plot(points[:,0], points[:,1], 'k*')[0] )
                self.fig_lines.append( self.ax.plot(left_boundary[:,0], left_boundary[:,1], 'ro')[0] )
                self.fig_lines.append( self.ax.plot(right_boundary[:,0], right_boundary[:,1], 'bo')[0] )
                self.fig_lines.append( self.ax.plot(path_point[:,0], path_point[:,1], '--')[0] )
                self.fig_lines.append( self.ax.plot(adjusted_path_point[:,0], adjusted_path_point[:,1], '-')[0] )
            else:
                self.fig_lines[0].set_data(points[:,0], points[:,1] )
                self.fig_lines[1].set_data(left_boundary[:,0], left_boundary[:,1] )
                self.fig_lines[2].set_data(right_boundary[:,0], right_boundary[:,1] )
                self.fig_lines[3].set_data(path_point[:,0], path_point[:,1])
                self.fig_lines[4].set_data(adjusted_path_point[:,0], adjusted_path_point[:,1])
            self.ax.relim()
            self.ax.autoscale_view()
            plt.draw()
            plt.pause(0.001)

        return adjusted_raceline

@jax.jit
def jax_adjust_path(gap_threshold, adjust_margin, lookahead, angles, ranges, path_point, x0, car_s):
    ''' x0: x,y,heading '''
    # transform to map frame
    xx = x0[0] + ranges * jnp.cos(angles + x0[2])
    yy = x0[1] + ranges * jnp.sin(angles + x0[2])
    points = jnp.stack((xx, yy), axis=1)
    left_idx = jnp.argmin(jnp.abs(angles - jnp.pi / 2))
    right_idx = jnp.argmin(jnp.abs(angles + jnp.pi / 2))
    gap = jnp.linalg.norm(jnp.diff(points,axis=0),axis=1)
    max_gap_idx = jnp.argmax(gap)

    right_gap = jnp.roll(gap, -right_idx)
    right_end_idx = jnp.argmax(right_gap > gap_threshold)
    right_end_idx = jnp.where(right_end_idx == 0, (max_gap_idx - right_idx)%len(gap), right_end_idx)

    right_boundary = jnp.roll(points, -right_idx, axis=0)
    cond = jnp.arange(len(points)) < right_end_idx
    right_boundary = jnp.where(cond[:,None], right_boundary, right_boundary[right_end_idx-1])

    left_gap = jnp.roll(gap[::-1], -len(gap)+left_idx)
    left_end_idx = jnp.argmax(left_gap > gap_threshold)
    left_end_idx = jnp.where(left_end_idx == 0, ( - max_gap_idx + left_idx)%len(gap),left_end_idx)

    left_boundary = jnp.roll(points[::-1], -len(gap)+left_idx, axis=0)
    cond = jnp.arange(len(points)) < left_end_idx
    left_boundary = jnp.where(cond[:,None], left_boundary, left_boundary[left_end_idx-1])

    def resample(points):
        ''' resample to equidistance points '''
        dr = jnp.diff(points, axis=0)
        s = jnp.hstack([jnp.array([0]), jnp.cumsum(dr[:,0]**2+dr[:,1]**2)])
        xx = jnp.interp(jnp.linspace(0,s[-1]), s, points[:,0])
        yy = jnp.interp(jnp.linspace(0,s[-1]), s, points[:,1])
        retval = jnp.hstack([xx[:,None],yy[:,None]])
        return retval

    left_boundary_points = resample(left_boundary)
    right_boundary_points = resample(right_boundary)

    def get_adjustment(point, dr):
        ss = jnp.linspace(0,1, 50)
        adjustment = 0 # R_ccw @ dr -> positive

        closest_idx = jnp.argmin( jnp.linalg.norm(point - left_boundary_points,axis=1) )
        left_r = left_boundary_points[closest_idx]
        ref_to_bdry = left_r - point
        cross = jnp.cross(dr, ref_to_bdry)
        adjustment = jnp.where(cross<adjust_margin, - (adjust_margin - cross), 0)

        closest_idx = jnp.argmin( jnp.linalg.norm(point - right_boundary_points,axis=1) )
        right_r = right_boundary_points[closest_idx]
        ref_to_bdry = right_r - point
        cross = jnp.cross(dr, ref_to_bdry)
        adjustment = jnp.where(cross>-adjust_margin, (adjust_margin + cross), adjustment)
        return adjustment

    d_path_point = jnp.diff(path_point, axis=0)
    d_path_point = d_path_point / jnp.linalg.norm(d_path_point, axis=1)[:,None]
    adjustment = jnp.array([get_adjustment(p,dr) for p,dr in zip(path_point[:-1],d_path_point)])
    # adjustment is like a sawtooth, we don't like that
    a_prev = adjustment[:-2]; a_mid = adjustment[1:-1]; a_next = adjustment[2:]
    is_min = jnp.logical_and(jnp.logical_and(a_mid > 0 , a_mid < a_prev) , a_mid < a_next)
    is_max = jnp.logical_and(jnp.logical_and(a_mid < 0 , a_mid > a_prev) , a_mid > a_next)
    cond = jnp.logical_or(is_min , is_max)
    updated_adjustment = jnp.where(cond, (a_prev + a_next)/2, 0)
    #return jnp.concatenate([jnp.zeros(1), updated_adjustment, jnp.zeros(1)]), left_boundary_points, right_boundary_points
    return adjustment, left_boundary_points, right_boundary_points


def main(args=None):
    rclpy.init(args=args)
    node = StanleyController()
    try:
        rclpy.spin(node)
    except (RuntimeError, KeyboardInterrupt):
        tu.summary()
        '''
        plt.plot(node.debug_dict['lateral'], label='lateral')
        plt.plot(node.debug_dict['dlateral'], label='dlateral')
        plt.plot(node.debug_dict['heading'], label='heading')
        plt.plot(node.debug_dict['dheading'], label='dheading')
        plt.legend()
        plt.show()
        '''

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

