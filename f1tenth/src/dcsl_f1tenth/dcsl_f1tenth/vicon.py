# Interface for VICON UDB Object Stream
# Parse Vicon UDP Object Stream, supports multiple objects on SINGLE port
# publish vicon pose with interpolated twist as an Odometry msg, frame: vicon -> vicon/obj_name_odom
# NOTE, the objects tracked by vicon can't change while this node is running.
import socket
from time import time,sleep
from struct import unpack
from math import degrees,radians,sin,cos,atan2,asin,acos
from scipy.spatial.transform import Rotation
from threading import Lock,Event
import pickle
import numpy as np
import threading
import re
import queue

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from transforms3d import euler

class Vicon(Node):
    def __init__(self):
        super().__init__('vicon')
        self.publisher_vec = {}
        self.obj_name_to_topic_name = {}
        self.vi = _Vicon(daemon=True)
        self.vi.getViconUpdate()
        sleep(0.1)
        self.setup()
        self.vi.updateCallback = self.publish
        self.last_update = time()

    def setup(self):
        ''' create the topic publishers '''
        for i in range(self.vi.obj_count):
            obj_name = self.vi.getItemName(i)
            if not obj_name in self.publisher_vec.keys():
                try:
                    topic_name = f'/vicon/{obj_name}'
                    self.obj_name_to_topic_name[obj_name] = topic_name
                    self.publisher_vec[obj_name] = self.create_publisher(Odometry, topic_name, 10)
                except rclpy.exceptions.InvalidTopicNameException as e:
                    self.get_logger().warning(str(e))
                    legal_pattern = re.compile(r'[^a-zA-Z0-9_~{}]')
                    sanitized_name = legal_pattern.sub('_', obj_name)
                    if sanitized_name[0] in '0123456789':
                        sanitized_name = 'n' + sanitized_name
                    topic_name = f'/vicon/{sanitized_name}'
                    self.obj_name_to_topic_name[obj_name] = topic_name
                    self.get_logger().warning('we are renaming the topic, topic name will NOT match vicon object name')
                    self.publisher_vec[obj_name] = self.create_publisher(Odometry, topic_name, 10)

                self.get_logger().info(f' now publishing {topic_name} ')


    def publish(self):
        #print( 1.0/ (time() - self.last_update))
        self.last_update = time()
        for i in range(self.vi.obj_count):
            # according to https://help.vicon.com/space/Tracker40/14188787/Access+Vicon+Tracker+data+from+Simulink
            # the euler angle is XYZ, intrinsic, or ZYX, extrinsic
            obj_name = self.vi.getItemName(i)
            (x,y,z,rx,ry,rz) = self.vi.getGlobalState(i)

            ts = self.get_clock().now().to_msg()
            odom = Odometry()
            odom.header.stamp = ts
            odom.header.frame_id = 'vicon'
            odom.child_frame_id = f'/vicon/{obj_name}'
            odom.pose.pose.position.x = x
            odom.pose.pose.position.y = y
            odom.pose.pose.position.z = z
            quat = euler.euler2quat(rz, ry, rx, axes='szyx')
            odom.pose.pose.orientation.x = quat[1]
            odom.pose.pose.orientation.y = quat[2]
            odom.pose.pose.orientation.z = quat[3]
            odom.pose.pose.orientation.w = quat[0]

            twist = self.vi.getGlobalTwist(i)
            odom.twist.twist.linear.x = twist[0]
            odom.twist.twist.linear.y = twist[1]
            odom.twist.twist.linear.z = twist[2]
            odom.twist.twist.angular.x = twist[3]
            odom.twist.twist.angular.y = twist[4]
            odom.twist.twist.angular.z = twist[5]
            self.publisher_vec[obj_name].publish(odom)
            #self.get_logger().info(f' {self.obj_name_to_topic_name[obj_name]}: {x,y,z}, euler: {degrees(rx),degrees(ry),degrees(rz)} ')


def quad_fit_functional(n, dt):
    t_history = np.arange(-n + 1, 1)
    A = ndarray(np.vstack([t_history ** 2, t_history, np.ones(n)]).T)
    A_op = np.linalg.inv(A.T @ A) @ A.T
    def quad_fit(x_history):
        x_history = ndarray(x_history).reshape((-1, n))
        return x_history @ A_op[1] / dt
    return quad_fit
def ndarray(x):
    return np.asarray(x, dtype=np.float64)

class _Vicon:
    # IP : IP address to listen on, if you only have a single network card, default should work
    # Port : Port number to listen on, default is vicon's default port
    # daemon : whether to spawn a daemon update thread
    #     If user prefer to get vicon update manually, this can be set to false
    #     However, user need to make sure to call getViconUpdate() frequently enough to prevent 
    #     the incoming network buffer from filling up, which would result in self.getViconUpdate
    #     gettting stacked up outdated vicon frames
    #
    #     In applications where getViconUpdate() cannot be called frequently enough,
    #     or when the user code can't afford to wait for getViconUpdate() to complete.
    #     User may choose to set daemon to True, in which case a new thread would be spawned 
    #     dedicated to receiving vicon frames and maintaining a local copy of the most recent states

    # NOTE If the the set of objects picked up by vicon changes as the program is running, each object's ID may change
    # It is recommended to work with a fixed number of objects that can be readily tracked by vicon
    # thoughout the use of the this class
    # One object per port would solve this issue since each object would have a unique ID, whether or not it is active/detected
    # However it is not currently supported. If you have a strong need for this feature please contact author
    def __init__(self,IP=None,PORT=None,daemon=True,enableKF=True):

        self.newState = Event()
        self.vicon_freq = 119.88
        self.dt = dt = 1.0/self.vicon_freq

        if IP is None:
            IP = "0.0.0.0"
        if PORT is None:
            PORT = 51001

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(0.05)
        self.sock.bind((IP, PORT))
        # number of currently published tracked objects
        self.obj_count = None
        # lock for accessing member variables since they are updated in a separate thread
        self.state_lock = Lock()
        self.vicon_loss = Event()

        self.recording_data = []
        self.recording = Event()

        # contains a list of names of tracked objects
        self.obj_names = []
        # a list of state tuples, state tuples take the form: (x,y,z,rx,ry,rz), in meters and radians, respectively
        # note that rx,ry,rz are euler angles in XYZ convention, this is different from the ZYX convention commonly used in aviation
        self.global_state_list = []
        self.local_state_list = []

        # flag used to stop update daemon thread
        self.quit_thread = False
        self.updateCallback = self.doNothing

        self.vel_est_n_step = 5
        self.quad_fit = quad_fit_functional(self.vel_est_n_step, dt)

        self.velocity_estimator_ready = Event()
        self.global_xyz_historys = []
        self.global_xyz_dots = []
        self.global_rxyz_dots = []
        self.local_xyz_historys = []
        self.local_xyz_dots = []

        # set the relation between Vicon world frame and user defined local frame
        # describe the rotation needed to rotate the world frame to track frame
        # in sequence Z, Y, X, each time using intermediate frame axis (intrinsic)
        # NOTE scipy.spatial.transform.Rotation uses active rotation
        # NOTE self.R is the passive rotation matrix
        #self.R = Rotation.from_euler("ZYX",[180,0,0],degrees=True).inv()
        # custome routine is faster
        #self.R = self.eulerZyxToR(radians(-90),0,0)
        self.R = self.eulerZyxToR(radians(-90),0,radians(180))

        # within world ref frame, where is local frame origin
        #self.local_frame_origin_world = np.array([1,3.5,-0])
        self.local_frame_origin_world = np.array([0,2.0,-0])

        if daemon:
            self.thread =  threading.Thread(name="vicon",target=self.viconUpateDaemon)
            self.thread.start()
        else:
            self.thread = None

        
    def __del__(self):
        if not (self.thread is None):
            self.quit_thread = True
            self.thread.join()
        self.sock.close()

    # set a callback function for new vicon updates
    def setCallback(self,fun):
        self.updateCallback = fun
        return

    # placeholder for an empty function
    def doNothing(self):
        pass

    # get name of an object given its ID. This can be useful for verifying ID <-> object relationship
    def getItemName(self,obj_id):
        self.state_lock.acquire()
        local_name = self.obj_names[obj_id]
        self.state_lock.release()
        return local_name

    # get item id from name
    def getItemID(self,obj_name):
        self.state_lock.acquire()
        local_names = self.obj_names
        self.state_lock.release()
        try:
            obj_id = local_names.index(obj_name)
        except ValueError:
            obj_id = None
            print("Error, item :"+str(obj_name)+" not found")
        finally:
            return obj_id

    # get state by id (local frame)
    def getState(self,inquiry_id):
        if self.obj_count is None:
            print("lost vicon")
            return None
        if inquiry_id>=self.obj_count:
            print("error: invalid id : "+str(inquiry_id))
            return None
        self.state_lock.acquire()
        retval = self.local_state_list[inquiry_id]
        self.state_lock.release()
        return retval

    # get state by id
    def getGlobalState(self,inquiry_id):
        if inquiry_id>=self.obj_count:
            print("error: invalid id : "+str(inquiry_id))
            return None
        self.state_lock.acquire()
        retval = self.global_state_list[inquiry_id]
        self.state_lock.release()
        return retval

    def getVelocity(self, inquiry_id):
        if inquiry_id>=self.obj_count:
            print("error: invalid id : "+str(inquiry_id))
            return None
        retval = np.array(self.local_xyz_dots[inquiry_id]).flatten()
        return retval

    def getGlobalTwist(self, inquiry_id):
        if inquiry_id>=self.obj_count:
            print("error: invalid id : "+str(inquiry_id))
            return None
        retval = np.concatenate([self.global_xyz_dots[inquiry_id],self.global_rxyz_dots[inquiry_id]]).flatten()
        return retval

    def viconUpateDaemon(self):
        while not self.quit_thread:
            self.getViconUpdate()

    def quit(self):
        self.stopUpdateDaemon()

    # stop the update thread
    def stopUpdateDaemon(self):
            if not (self.thread is None):
                self.quit_thread = True
                self.thread.join()
                self.thread = None

    def getViconUpdate(self,debugData=None):
        # the no of bytes here must agree with length of a vicon packet
        # typically 256,512 or 1024
        try:
            if debugData is None:
                data, addr = self.sock.recvfrom(256)
                # in python 2 data is of type str
                #data = data.encode('ascii')
            else:
                data = debugData
            obj_names = []
            global_state_list = []
            local_state_list = []

            self.obj_count = itemsInBlock = data[4]
            itemID = data[5] # always 0, not very useful
            itemDataSize = unpack('h',data[6:8])

            if (not self.velocity_estimator_ready.is_set()):
                self.local_xyz_historys = [[[0,0,0] for j in range(self.vel_est_n_step)] for i in range(self.obj_count)]
                self.global_xyz_historys = [[(0,0,0) for j in range(self.vel_est_n_step)] for i in range(self.obj_count)]
                self.global_rxyz_historys = [[(0,0,0) for j in range(self.vel_est_n_step)] for i in range(self.obj_count)]
                self.local_xyz_dots = [[0,0,0] for i in range(self.obj_count)]
                self.global_xyz_dots = [[0,0,0] for i in range(self.obj_count)]
                self.global_rxyz_dots = [[0,0,0] for i in range(self.obj_count)]
                self.velocity_estimator_ready.set()

            for i in range(itemsInBlock):
                offset = i*75
                itemName = data[offset+8:offset+32].rstrip(b'\0').decode()
                # raw data in mm, convert to m
                x = unpack('d',data[offset+32:offset+40])[0]/1000.0
                y = unpack('d',data[offset+40:offset+48])[0]/1000.0
                z = unpack('d',data[offset+48:offset+56])[0]/1000.0
                # euler angles,rad, rotation order: rx,ry,rz, using intermediate frame
                rx = unpack('d',data[offset+56:offset+64])[0]
                ry = unpack('d',data[offset+64:offset+72])[0]
                rz = unpack('d',data[offset+72:offset+80])[0]

                obj_names.append(itemName)
                global_state_list.append((x,y,z,rx,ry,rz))

                x_local, y_local, z_local = self.R @ (np.array((x,y,z)) - self.local_frame_origin_world)
                # FIXME
                #r = self.eulerZyxToR(rz, ry, rx)
                # XYZ: intrinsic rotated ref frame, xyz: extrinsic, static ref frame
                r = Rotation.from_euler("XYZ",[rx,ry,rz],degrees=False).inv()

                local_r = r.as_matrix() @ np.linalg.inv(self.R)
                rx_local, ry_local, rz_local = self.rotationToEulerZyx(local_r)
                local_state_list.append((x_local, y_local, z_local, rx_local, ry_local, rz_local))

                if (self.global_xyz_historys[-1] == (x,y,z)):
                    self.vicon_loss.set()
                else:
                    self.vicon_loss.clear()

                self.global_xyz_historys[i].append((x,y,z))
                self.global_xyz_historys[i].pop(0)
                self.global_rxyz_historys[i].append((rx,ry,rz))
                self.global_rxyz_historys[i].pop(0)
                # failsafe, check for repeating measurements


                self.local_xyz_historys[i].append([x_local,y_local,z_local])
                self.local_xyz_historys[i].pop(0)

                temp_xyz_history = np.array(self.local_xyz_historys[i])
                x_hist = temp_xyz_history[:,0]
                y_hist = temp_xyz_history[:,1]
                z_hist = temp_xyz_history[:,2]
                vx_local = self.quad_fit(x_hist)
                vy_local = self.quad_fit(y_hist)
                vz_local = self.quad_fit(z_hist)
                self.local_xyz_dots[i] = [vx_local,vy_local,vz_local]

                temp_xyz_history = np.array(self.global_xyz_historys[i])
                x_hist = temp_xyz_history[:,0]
                y_hist = temp_xyz_history[:,1]
                z_hist = temp_xyz_history[:,2]
                vx_global = self.quad_fit(x_hist)
                vy_global = self.quad_fit(y_hist)
                vz_global = self.quad_fit(z_hist)
                self.global_xyz_dots[i] = [vx_global,vy_global,vz_global]

                temp_rxyz_history = np.array(self.global_rxyz_historys[i])
                rx_hist = temp_rxyz_history[:,0]
                ry_hist = temp_rxyz_history[:,1]
                rz_hist = temp_rxyz_history[:,2]
                vrx_global = self.quad_fit(rx_hist)
                vry_global = self.quad_fit(ry_hist)
                vrz_global = self.quad_fit(rz_hist)
                self.global_rxyz_dots[i] = [vrx_global,vry_global,vrz_global]


            self.state_lock.acquire()
            self.obj_names = obj_names
            self.global_state_list = global_state_list
            self.local_state_list = local_state_list
            if (self.recording.is_set()):
                self.recording_data.append(local_state_list)
            self.state_lock.release()
            # call the callback function 
            self.updateCallback()

            self.newState.set()
            return local_state_list
        except socket.timeout:
            return None


    def testFreq(self,packets=100):
        # test actual frequency of vicon update, with PACKETS number of state updates
        tic = time()
        for i in range(packets):
            self.getViconUpdate()
        tac = time()
        return packets/(tac-tic)

    def startRecording(self, filename):
        self.recording_data = []
        self.recording.set()
        self.recording_filename = filename

    def stopRecording(self):
        f = open(self.recording_filename, "wb")
        pickle.dump(self.recording_data, f)
        f.close()
        #print_info("recording saved %s"%(self.recording_filename))
        return

    # r is passive rotation matrix
    def rotationToEulerZyx(self,r):
        #r = r.inv().as_matrix()
        rx = atan2(r[1,2],r[2,2])
        ry = -asin(r[0,2])
        rz = atan2(r[0,1],r[0,0])
        return (rx,ry,rz)

    # get the passive rotation matrix 
    # that is, x_body = R*x_world
    def eulerZyxToR(self,rz,ry,rx):
        R = [ [ cos(ry)*cos(rz), cos(ry)*sin(rz), -sin(ry)],
              [ sin(rx)*sin(ry)*cos(rz)-cos(rx)*sin(rz), sin(rx)*sin(ry)*sin(rz)+cos(rx)*cos(rz), cos(ry)*sin(rx)],
              [cos(rx)*sin(ry)*cos(rz)+sin(rx)*sin(rz), cos(rx)*sin(ry)*sin(rz)-sin(rx)*cos(rz), cos(ry)*cos(rx)]]
        
        return R

    # get passive rotation matrix
    def quatToR(self,q1,q2,q3,q0):
        # q1=qx
        # q2=qy
        # q3=qz
        # q0=q2
        R = [ [q0**2+q1**2-q2**2-q3**2, 2*q1*q2+2*q0*q3, 2*q1*q3-2*q0*q2],
              [2*q1*q2-2*q0*q3,  q0**2-q1**2+q2**2-q3**2, 2*q2*q3+2*q0*q1],
              [2*q1*q3+2*q0*q2,  2*q2*q3-2*q0*q1, q0**2-q1**2-q2**2+q3**2]]
        
        return R

    # for debug
    def saveFile(self,data,filename):
        newFile = open(filename, "wb")
        newFile.write(data)
        newFile.close()

    # for debug
    def loadFile(self,filename):
        newFile = open(filename, "rb")
        data = bytearray(newFile.read())
        newFile.close()
        return data

def main(args=None):
    rclpy.init(args=args)
    node = Vicon()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
