import pdb
import numpy as np
import os
import sys
import pickle
global BASEDIR
BASEDIR = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
import matplotlib.pyplot as plt

os.environ["JAX_PLATFORM_NAME"] = "cpu"
import jax
import jax.numpy as jnp
from jax import jit,vmap
from functools import partial

import rclpy
from rclpy.node import Node
from time import time,sleep

from geometry_msgs.msg import PoseStamped, TransformStamped, PointStamped
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry, Path
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
from TimeUtil import TimeUtil

tu = TimeUtil(True)

DEBUG = False

class MppiController(Node):
    # parameters
    params = {'mu': 1.0489, 'C_Sf': 4.718, 'C_Sr': 5.4562, 'lf': 0.15875, 'lr': 0.17145, 'h': 0.074, 'm': 3.74, 'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189, 'sv_min': -3.2, 'sv_max': 3.2, 'v_switch': 7.319, 'a_max': 9.51, 'v_min':-5.0, 'v_max': 20.0, 'width': 0.31, 'length': 0.58}
    mass = params['m'] # mass of vehicle
    C_Sf = params['C_Sf']
    C_Sr = params['C_Sr']
    lf = params['lf']
    lr = params['lr']
    Iz = params['I']
    g = 9.81
    mu = params['mu']
    h = params['h']
    s_min = params['s_min']
    s_max = params['s_max']

    dt = 0.05
    horizon = T = 30
    m = 2
    n = 7
    def __init__(self):
        super().__init__('mppi_controller')
        # TODO put this in ros_param
        self.map_frame = 'sim_map'
        self.last_steering = 0
        self.control_freq = 20
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
        self.visual_pub = self.create_publisher(Path, '/visual_mppi', 1)
        self.timer = self.create_timer(1/self.control_freq, self.update_control)  # 50 Hz
        self.planner_traj = None
        self.odom = None
        #self.compile()
        if (DEBUG):
            self.fig, self.ax = plt.subplots()
            self.fig_lines = None
            self.ax.set_aspect('equal', adjustable='datalim')


    def traj_callback(self, msg: Trajectory):
        self.planner_traj = msg
    def odom_callback(self, msg: Odometry):
        self.odom = msg

    def get_current_states(self):
        # Lookup latest transform from global frame to base_link
        trans: TransformStamped = self.tf_buffer.lookup_transform(
            self.map_frame, 'ego_racecar/base_link', rclpy.time.Time())
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

    def get_ref_traj(self):
        # decode planner local path
        # find point closest to vehicle
        def decode(point):
            x = point.pose.position.x
            y = point.pose.position.y
            q = point.pose.orientation
            _, _, yaw = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
            speed = point.longitudinal_velocity_mps
            return (x,y,yaw,speed)

        # NOTE only use first 50 points, currently 25 is around the car location, publish before and after to ensure continuity
        traj = np.array([ decode(point) for point in self.planner_traj.points[:50] ])
        return traj

    def sample_control(self, steering, vx):
        '''
        steering, vx: current values
        sample control using motion primitives,the control satisfies three setpoints:
            u(0) = u0 (given), u(ts) = us, u(T) = uT
            sample ts = [1..T-1], us,ut = [-u_max, u_max]
            fit a 2nd order polynomial, and obtain the dudt,which are the controls (steering rate and ax)
        '''
        T = self.horizon; dt=self.dt
        # NOTE
        # since we are fitting a 2nd order polynomial, we are essentially sampling a fixed ddu_dtdt rate, with an initial setpoint
        # it may be more efficient to think in that space
        control_vec = vmap(make_one_sample, in_axes=(None,None,0))(steering, vx, int(time()*1e9) + np.arange(self.samples) ).block_until_ready()
        return control_vec

    def update_control(self):
        if (self.planner_traj is None or self.odom is None):
            self.get_logger().info('planner_traj and odom not ready')
            return

        self.u_noise_scale = np.array([radians(24), 5])
        self.samples = 1000
        self.temperature = 0.01 # smaller it is, harder the softmax

        # ------ Main Entry Point ----
        try:
            tu.s()
            t0 = time()
            tu.s('prep')
            x,y,heading,vx,vy,yaw_rate = self.get_current_states()

            # add in a slight lookahead distance
            lookahead = 0.5
            x_lookahead = x + cos(heading) * lookahead
            y_lookahead = y + sin(heading) * lookahead
            coord = np.array((x_lookahead, y_lookahead))

            traj = self.get_ref_traj()
            r = np.vstack([traj[:,0], traj[:,1]])
            uu = np.linspace(-24,25,50)

            # state x: [x, y, steering, vx, heading, yaw_rate, slip_angle]
            #           0  1    2        3   4          5             6
            # control u: [ steering_rate, ax]
            if (vx > 0.5):
                x0 = np.array([x,y, self.last_steering, vx, heading, yaw_rate, atan(vy/vx)],dtype=np.float32)
            else:
                x0 = np.array([x,y, self.last_steering, 0.5, heading, 0.0, 0.0],dtype=np.float32)
            tu.e('prep')


            # random sample
            tu.s('mppi_sample_control')
            key = jax.random.PRNGKey(int(time()*1e9))
            #u_ref = np.zeros((self.T,self.m))
            #u_vec = u_ref + jax.random.normal(key,shape=(self.samples,self.T,self.m)) * self.u_noise_scale
            #u_vec = u_ref + jax.random.uniform(key,shape=(self.samples,T,m), minval=-1.0, maxval=1.0) * self.u_noise_scale
            u_vec = self.sample_control(self.last_steering, vx)
            tu.e('mppi_sample_control')

            tu.s('mppi_rollout')
            cost_vec = np.array(vmap(lambda u:rollout_cost(traj,x0,u))(u_vec).block_until_ready())
            cost_vec[np.isnan(cost_vec)] = 1e9
            tu.e('mppi_rollout')

            tu.s('post')
            # synthesize control
            '''
            cost_min = np.min(cost_vec)
            cost_mean = np.mean(cost_vec-cost_min)
            weight_vec = np.exp(- (cost_vec - cost_min)/cost_mean / self.temperature)
            weight_vec /= np.sum(weight_vec)
            synthesized_control = np.sum(weight_vec.reshape(-1,1,1) * np.array(u_vec),axis=0)
            '''
            # FIXME
            idx = np.argmin(cost_vec)
            synthesized_control = np.array(u_vec)[idx]

            steering = self.last_steering + 1/self.control_freq*synthesized_control[0,0]
            steering = np.clip(steering, self.s_min, self.s_max)
            target_speed = vx + 1/self.control_freq*synthesized_control[0,1]
            if (isnan(steering) or isnan(target_speed)):
                steering = 0.0
                target_speed = 0.0

            freq = 1/(time()-t0)

            traj_point = np.array(get_interpolated_traj_point(traj, x0)[0])

            self.get_logger().info(f'steering: {degrees(steering):.2f}, speed: {target_speed:.2f} freq: {freq:.1f} , v_err {traj_point[3] - vx:.1f}')

            self.last_steering = steering

            tu.e('post')
            # publish command
            ackermann_msg = AckermannDriveStamped()
            ackermann_msg.header.frame_id = ''
            ackermann_msg.header.stamp = self.get_clock().now().to_msg()

            ackermann_msg.drive.steering_angle = steering
            ackermann_msg.drive.speed = target_speed
            self.drive_pub.publish(ackermann_msg)

            state_traj = rollout(x0, synthesized_control)
            # publish projected trajectory
            path_msg = Path()
            path_msg.header.frame_id = self.map_frame
            path_msg.header.stamp = self.get_clock().now().to_msg()
            for state in state_traj:
                pose = PoseStamped()
                pose.header = path_msg.header
                pose.pose.position.x = float(state[0])
                pose.pose.position.y = float(state[1])
                pose.pose.position.z = 0.0
                pose.pose.orientation.w = 1.0  # Default orientation
                path_msg.poses.append(pose)
            self.visual_pub.publish(path_msg)
            tu.e()

            # ---- DEBUG ----
            if (False):
                # check sampled control saturation
                traj_vec = np.array(vmap(lambda u: rollout(x0, u))(u_vec).block_until_ready())
                steering_vec = traj_vec[:,:,2]
                for i in range(len(steering_vec)):
                    plt.plot(steering_vec[i])
                plt.show()
                # how many are out of bound?
                i#np.abs(steering_vec) > 0.99*self.s_max

                vx_vec = traj_vec[:,:,3]
                for i in range(len(vx_vec)):
                    plt.plot(vx_vec[i])
                plt.show()
                pdb.set_trace()

            if (DEBUG):
                # visualize local path, the sampled trajectories, and the selected optimal
                traj_vec = np.array(vmap(lambda u: rollout(x0, u))(u_vec).block_until_ready())
                traj_point = np.array(get_interpolated_traj_point(traj, x0)[0])

                if (self.fig_lines is None):
                    self.fig_lines = []
                    for i in range(len(traj_vec)):
                        self.fig_lines.append(self.ax.plot(traj_vec[i,:,0], traj_vec[i,:,1],'r-')[0])
                    self.fig_ref = self.ax.plot(traj[:,0], traj[:,1],'bo')[0]
                    self.fig_optimal = self.ax.plot(state_traj[:,0], state_traj[:,1],'k*')[0]
                    self.fig_ref_tan = self.ax.plot([traj_point[0], traj_point[0] + cos(traj_point[2])*0.5], [traj_point[1], traj_point[1] + sin(traj_point[2])*0.5])[0]
                else:
                    for i in range(len(traj_vec)):
                        self.fig_lines[i].set_data(traj_vec[i,:,0], traj_vec[i,:,1])
                    self.fig_ref.set_data(traj[:,0], traj[:,1])
                    self.fig_optimal.set_data(state_traj[:,0], state_traj[:,1])
                    self.fig_ref_tan.set_data([traj_point[0], traj_point[0] + cos(traj_point[2])*0.5], [traj_point[1], traj_point[1] + sin(traj_point[2])*0.5])

                self.ax.relim()
                self.ax.autoscale_view()
                plt.draw()
                plt.pause(0.001)


        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f'Failed to get transform: {e}')



    def compile(self):
        # jax jit compile the first time we run the functions
        self.get_logger().info('compiling jax')
        t0 = time()
        traj = jnp.zeros((100,4))
        x0 = jnp.zeros(7)
        u_vec = jnp.zeros((10,T,m))
        cost_vec = vmap(lambda u:rollout_cost(traj,x0,u))(u_vec).block_until_ready()
        state_traj = rollout(x0, u_vec[0])
        get_interpolated_traj_point(traj, x0)
        rollout_cost(traj, x0, u_vec[0])
        eval_cost(traj,state_traj, u_vec[0])
        dynamics(x0,u_vec[0,0])
        self.get_logger().info(f'done, took {time()-t0:.2f} seconds')
        return

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

def wrap(x):
    return (x + np.pi) % (2*np.pi) - np.pi

# helper functions, jax.jit compiled
# static because jax doesn't handle class member functions well
@jit
def get_interpolated_traj_point(traj, x):
    min_dist = 1e9
    before = 0
    after = 0
    def find_dist(i):
        return (traj[i,0] - x[0])**2 + (traj[i,1] - x[1])**2 + (traj[i+1,0] - x[0])**2 + (traj[i+1,1] - x[1])**2
    idx_range = jnp.arange(0,traj.shape[0])
    dist_vec = vmap(find_dist)(idx_range)
    i = jnp.argmin(dist_vec)
    #interpolate
    # let before = A, after = B, x(car state) = P
    # t = AB dot AP / |AB|^2
    AB = (traj[i+1,0] - traj[i,0], traj[i+1,1] - traj[i,1])
    AP = (x[0] - traj[i,0], x[1] - traj[i,1])
    t = AB[0]*AP[0] + AB[1]*AP[1] / (AB[0]**2 + AB[1]**2)
    retval = traj[i]*t + traj[i+1]*(1-t)
    return retval ,i

@jit
def rollout_cost(traj, x0, u):
    state_traj = rollout(x0,u)
    return eval_cost(traj, state_traj, u)

@jit
def eval_cost(traj,state_traj, u):
    def step_cost(x,u_t):
        traj_point, guess = get_interpolated_traj_point( traj, x)
        # state x: [x, y, steering, vx, heading, yaw_rate, slip_angle]
        #           0  1    2        3   4          5             6
        # control u: [ steering_rate, ax]
        # lateral error, heading error, progress
        # dx**2 + dy**2 + dvx**2 + wrap(d_yaw)**2
        #jax.debug.print('x {},y {},steering {}, vx {}, heading {}', x[0], x[1], x[2]/jnp.pi*180, x[3], x[4]/jnp.pi*180, ordered=True)
        #this_cost = (traj_point[0] - x[0])**2 + (traj_point[1] - x[1])**2 + 10*(traj_point[3] - x[3])**2 + (traj_point[2] - x[4])**2 \
                #        + 0.001*(u_t[0]**2 + u_t[1]**2) +  ((jnp.abs(x[3]) + x[3])/2) * 10
        yaw_err = (x[4] - traj_point[2] + jnp.pi) % (2*jnp.pi) - jnp.pi
        dist_err = jnp.sqrt(jnp.abs(traj_point[0] - x[0])**2 +jnp.abs(traj_point[1] - x[1]))
        this_cost =  5*dist_err**2 + 8*jnp.abs(yaw_err)**2 + 0.6*(traj_point[3] - x[3])**2 + 1*(jnp.abs(x[3]-0.1)-(x[3]-0.1))
        return this_cost
    xx = state_traj[1:]
    uu = u
    cost_vec = vmap(step_cost)(xx,uu)

    return cost_vec.sum()

@jit
def rollout(x0, u):
    def scan_fun(_x, _u):
        newx = _x + dynamics(_x,_u)*MppiController.dt
        newx = newx.at[2].set(jnp.clip(newx[2],MppiController.s_min, MppiController.s_max))
        return newx, newx
    _, state_traj = jax.lax.scan(scan_fun, x0, u)
    return jnp.vstack([x0[None,:],state_traj])

@jit
def dynamics(x,u):
    T = MppiController.horizon
    mass = MppiController.mass
    C_Sf = MppiController.C_Sf
    C_Sr = MppiController.C_Sr
    lf = MppiController.lf
    lr = MppiController.lr
    Iz = MppiController.Iz
    g = 9.81
    mu = MppiController.mu
    h = MppiController.h*0
    # kinematic bicycle
    f_kinematic = jnp.array([x[3]*jnp.cos(x[4]),
         x[3]*jnp.sin(x[4]),
         u[0],
         u[1],
         x[3]/(lf+lr)*jnp.tan(x[2]),
         0,
         0])
    # dynamic bicycle
    f_dynamic = jnp.array([x[3]*jnp.cos(x[6] + x[4]),
        x[3]*jnp.sin(x[6] + x[4]),
        u[0],
        u[1],
        x[5],
        -mu*mass/(x[3]*Iz*(lr+lf))*(lf**2*C_Sf*(g*lr-u[1]*h) + lr**2*C_Sr*(g*lf + u[1]*h))*x[5] \
            +mu*mass/(Iz*(lr+lf))*(lr*C_Sr*(g*lf + u[1]*h) - lf*C_Sf*(g*lr - u[1]*h))*x[6] \
            +mu*mass/(Iz*(lr+lf))*lf*C_Sf*(g*lr - u[1]*h)*x[2],
        (mu/(x[3]**2*(lr+lf))*(C_Sr*(g*lf + u[1]*h)*lr - C_Sf*(g*lr - u[1]*h)*lf)-1)*x[5] \
            -mu/(x[3]*(lr+lf))*(C_Sr*(g*lf + u[1]*h) + C_Sf*(g*lr-u[1]*h))*x[6] \
            +mu/(x[3]*(lr+lf))*(C_Sf*(g*lr-u[1]*h))*x[2]])
    #FIXME
    #retval = jnp.where(x[3]<0.5, f_kinematic, f_dynamic)
    retval = jnp.where(x[3]<5, f_kinematic, f_dynamic)
    return retval

@jit
def make_one_sample(steering, vx, seed):
    T = MppiController.T
    dt = MppiController.dt
    idxs = jnp.arange(T)
    key = jax.random.PRNGKey(seed)
    # steering
    u0 = steering
    key, subkey = jax.random.split(key)
    ts = jax.random.randint(subkey,shape=1, minval=5, maxval=T-1)[0]
    key, subkey = jax.random.split(key)
    us = jax.random.uniform(subkey,shape=1, minval=MppiController.s_min, maxval=MppiController.s_max)[0]
    key, subkey = jax.random.split(key)
    uT = jax.random.uniform(subkey,shape=1, minval=MppiController.s_min, maxval=MppiController.s_max)[0]
    s1 = (us-u0)/ts/dt
    s2 = (uT-us)/(T-ts)/dt
    d_steering_dt = jnp.where( idxs < ts, s1, s2 )

    # acceleration
    u0 = vx
    vx_min = jnp.clip(vx - 1, 0.1, 6.0)
    vx_max = jnp.clip(vx + 1, 0.1, 6.0)
    key, subkey = jax.random.split(key)
    ts = jax.random.randint(subkey,shape=1, minval=5, maxval=T-1)[0]
    key, subkey = jax.random.split(key)
    us = jax.random.uniform(subkey,shape=1, minval=vx_min, maxval=vx_max)[0]
    key, subkey = jax.random.split(key)
    uT = jax.random.uniform(subkey,shape=1, minval=vx_min, maxval=vx_max)[0]
    s1 = (us-u0)/ts/dt
    s2 = (uT-us)/(T-ts)/dt
    d_speed_dt = jnp.where( idxs < ts, s1, s2 )

    return jnp.vstack([d_steering_dt, d_speed_dt]).T



def main(args=None):
    rclpy.init(args=args)
    node = MppiController()
    try:
        rclpy.spin(node)
        node.destroy_node()
        rclpy.shutdown()
    finally:
        tu.summary()


if __name__ == '__main__':
    main()

