#!/usr/bin/env python3

'''
*****************************************************************************************
*
*               ===============================================
*                       Cosmo Logistic (CL) Theme (eYRC 2023-24)
*               ===============================================
*
*  This script should be used to implement Task 2A of Cosmo Logistic (CL) Theme (eYRC 2023-24).
*
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''

# Team ID:          [ 1243 ]
# Author List:      [ Kishore S, Pragadheesh RJ, Vimal VK, Jagannathan M]
# Filename:         2task2a.py
# Functions:
#                   Main Functions:
#                   [ main, calculate_intermediate_position, quaternion_from_euler, euler_from_quaternion]
#                   Functions of move_boxes class :
#                   [servo_publish, servo_publish_rotation, move_ur5, on_timer]
# Nodes:            
#                   Publishing Topics  - [ /servo_node/delta_twist_cmds ]
#                   Subscribing Topics - [ /tf, /obj_frames]
#                   Client services    - [ /GripperMagnetOn, /GripperMagnetOff, /servo_node/start_servo ]


################### IMPORT MODULES #######################


from os import path
from threading import Thread

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node

from pymoveit2 import MoveIt2
from pymoveit2.robots import ur5

import math
import numpy as np
import time
from std_msgs.msg import Int8
from std_srvs.srv import Trigger
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation as R
from linkattacher_msgs.srv import AttachLink, DetachLink
from geometry_msgs.msg import TwistStamped
from ros2_interfaces_cpp.msg import List



##################### FUNCTION DEFINITIONS #######################


def quaternion_from_euler(roll, pitch, yaw):
    """
    Description:
        Convert an Euler angle to a quaternion.
    
    Args:
        roll: The roll (rotation around x-axis) angle in radians.
        pitch: The pitch (rotation around y-axis) angle in radians.
        yaw: The yaw (rotation around z-axis) angle in radians.
    
    Returns:
        qx: The orientation in x axis in quaternion format
        qy: The orientation in y axis in quaternion format
        qz: The orientation in z axis in quaternion format
        qw: The angle of Rotation in the quaternion xyz vector

    """

    ############ Function Variables ############
    qx = 0
    qy = 0 
    qz = 0 
    qw = 0

    ############ Code ############

    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    
    return [qx, qy, qz, qw]



def euler_from_quaternion(quat):
    """
    Description:
        Convert an Quaternion values to Euler Angles format
    
    Args:
        quat[0] : The orientation in x axis in quaternion format
        quat[1] : The orientation in y axis in quaternion format
        quat[2] : The orientation in z axis in quaternion format
        quat[3] : The angle of Rotation in the quaternion xyz vector
    
    Returns:
        roll: The roll (rotation around x-axis) angle in radians.
        pitch: The pitch (rotation around y-axis) angle in radians.
        yaw: The yaw (rotation around z-axis) angle in radians.

    """

    ############ Function VARIABLES ############

    x = quat[0]
    y = quat[1]
    z = quat[2]
    w = quat[3]
    t0 = 0
    t1 = 0
    t2 = 0
    t3 = 0
    t4 = 0

    ############ Code ############
        
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1) #tan inverse function
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)  #sin inverse function
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)  #tan inverse function

    euler_angles = [roll_x, pitch_y, yaw_z]
    
    return euler_angles


def calculate_intermediate_position(position, quat_xyzw):
    """
    Description:
        Calculate intermediate position to attain the box
    
    Args:
        position : The distance in xyz axis for the box
        quat_xyzw : The orientation in xyz axis with angle w in quaternion format
    
    Output
        intermediate_position: The position value in xyz axis

    """

    ############ Code ############

    # Calculate the box's z-axis in the base_link frame
    box_z_axis = [0, 0, 1]  # Assuming the box's z-axis is the same for all boxes

    # Rotate the box's z-axis by its orientation
    r = R.from_quat(quat_xyzw)
    rotated_box_z_axis = r.apply(box_z_axis)

    # Calculate the intermediate position as 0.2 meters in the negative direction of the rotated box's z-axis
    intermediate_position = [
        position[0] - 0.15 * rotated_box_z_axis[0],
        position[1] - 0.15 * rotated_box_z_axis[1],
        position[2] - 0.15 * rotated_box_z_axis[2]
    ]

    return intermediate_position


class move_boxes(Node):
    '''
    ___CLASS___

    Description:    
        Class which servo feature to move the boxes
    '''

    def __init__(self):
        '''
        Description:    
            The __init__() function is called automatically every time the class is being used to create a new object.
        '''

        super().__init__('task2a2')

        ############ Topics for Publish/ Subscribe ############

        self.move_ur5_sub = self.create_subscription(List, '/obj_frames', self.move_ur5, 10)
        self.servo_status_sub = self.create_subscription(Int8, '/servo_node/status', self.servo_status_callback, 10)
        self.servo_publisher = self.create_publisher(TwistStamped, '/servo_node/delta_twist_cmds', 10)
        self.twist_msg = TwistStamped()

        #Configuring the Path of the Mesh file 
        DEFAULT_EXAMPLE_MESH = path.join(
            path.dirname(path.realpath(__file__)), "assets", "rack.stl"
        )

        ARM_BASE_MESH = path.join(
            path.dirname(path.realpath(__file__)), "assets", "arm_base.stl"
        )

        TABLE_MESH = path.join(
            path.dirname(path.realpath(__file__)), "assets", "table.stl"
        )


        # Create callback group that allows execution of callbacks in parallel without restrictions
        self.callback_group = ReentrantCallbackGroup()

        # Create MoveIt 2 interface
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=ur5.joint_names(),
            base_link_name=ur5.base_link_name(),
            end_effector_name=ur5.end_effector_name(),
            execute_via_moveit=True,
            group_name=ur5.MOVE_GROUP_ARM,
            callback_group=self.callback_group,
        )

        # Get parameters for Positions
        self.filepath = ""
        self.action = "add"
        self.position_rack_front = [0.57, 0.05, -0.56]
        self.quat_xyzw_rack_front = quaternion_from_euler(np.pi, np.pi, 0.0)
        self.position_rack_left = [0.29, 0.66, -0.56]
        self.quat_xyzw_rack_left = quaternion_from_euler(0.0, 0.0, np.pi/2)
        self.position_rack_right = [0.29, -0.66, -0.56]
        self.quat_xyzw_rack_right = quaternion_from_euler(0.0, 0.0, np.pi/2)
        self.position_arm_base = [-0.2, 0.00, -0.58]
        self.quat_xyzw_arm_base = quaternion_from_euler(0.0, 0.0, np.pi)
        self.position_table = [-0.8, 0.00, -0.60]
        self.quat_xyzw_table = quaternion_from_euler(0.0, 0.0, 0.0)

        #Create a buffer element to obtain the tf data
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.position_initial = [0.176, 0.108, 0.470]
        self.quat_xyzw_initial = quaternion_from_euler(np.pi/2, 0.0, np.pi/2)

        self.position_intermediate = [0.0, 0.10, 0.68]
        self.quat_xyzw_intermediate = quaternion_from_euler(np.pi/2, 0.0, np.pi/2)

        self.position_d = [-0.814, 0.108, 0.312]
        self.quat_xyzw_d = quaternion_from_euler(np.pi, np.pi/2, 0.0)
        self.joint_positions_center = [0.00, -2.39, 2.40, -3.15, -1.58, 3.15]
        self.joint_positions_d = [0.00, -2.39, -0.50, -3.15, -1.58, 3.15]
        self.joint_positions_d2 = [0.334, -2.39, -0.50, -3.15, -1.58, 3.15]
        self.joint_positions_d3 = [-0.334, -2.39, -0.50, -3.15, -1.58, 3.15]
        self.position_d2 = [0.15, 0.108, 0.6]
        self.quat_xyzw_d2 = quaternion_from_euler(np.pi/2, 0.0, np.pi/2)

        self.cartesian = False

        self.servo_status = 0

        # Use the default example mesh if invalid
        if not self.filepath:
            self.get_logger().info(f"Using the default example mesh file")
            self.filepath = DEFAULT_EXAMPLE_MESH
            self.filepath_arm_base = ARM_BASE_MESH
            self.filepath_table = TABLE_MESH

        # Make sure the mesh file exists
        if not path.exists(self.filepath):
            self.get_logger().error(f"File '{self.filepath}' does not exist")
            rclpy.shutdown()
            exit(1)

        # Determine ID of the collision mesh
        self.mesh_id_front = "_".join(path.basename(self.filepath).split(".")) + "_front"
        self.mesh_id_left = "_".join(path.basename(self.filepath).split(".")) + "_left"
        self.mesh_id_right = "_".join(path.basename(self.filepath).split(".")) + "_right"
        self.mesh_id_arm_base = "_".join(path.basename(self.filepath_arm_base).split("."))
        self.mesh_id_table = "_".join(path.basename(self.filepath_table).split("."))

        #To get the accuracy in Collision 
        for i in range(2):
            if "add" == self.action:
            
                # Add collision mesh rack_front
                self.get_logger().info(
                    f"Adding collision mesh '{self.filepath}' {{position: {list(self.position_rack_front)}, quat_xyzw: {list(self.quat_xyzw_rack_front)}}}"
                )

                self.moveit2.add_collision_mesh(
                    filepath=self.filepath, id=self.mesh_id_front, position=self.position_rack_front, quat_xyzw=self.quat_xyzw_rack_front, frame_id=ur5.base_link_name()
                )

                # Add collision mesh rack_left
                self.get_logger().info(
                    f"Adding collision mesh '{self.filepath}' {{position: {list(self.position_rack_left)}, quat_xyzw: {list(self.quat_xyzw_rack_left)}}}"
                )
                self.moveit2.add_collision_mesh(
                    filepath=self.filepath, id=self.mesh_id_left, position=self.position_rack_left, quat_xyzw=self.quat_xyzw_rack_left, frame_id=ur5.base_link_name()
                )
                
                # Add collision mesh rack_right
                self.get_logger().info(
                    f"Adding collision mesh '{self.filepath}' {{position: {list(self.position_rack_right)}, quat_xyzw: {list(self.quat_xyzw_rack_right)}}}"
                )
                self.moveit2.add_collision_mesh(
                    filepath=self.filepath, id=self.mesh_id_right, position=self.position_rack_right, quat_xyzw=self.quat_xyzw_rack_right, frame_id=ur5.base_link_name()
                )

                # Add collision mesh arm_base
                self.get_logger().info(
                    f"Adding collision mesh '{self.filepath_arm_base}' {{position: {list(self.position_arm_base)}, quat_xyzw: {list(self.quat_xyzw_arm_base)}}}"
                )
                self.moveit2.add_collision_mesh(
                    filepath=self.filepath_arm_base, id=self.mesh_id_arm_base, position=self.position_arm_base, quat_xyzw=self.quat_xyzw_arm_base, frame_id=ur5.base_link_name()
                )

                # Add collision mesh table
                self.get_logger().info(
                    f"Adding collision mesh '{self.filepath_table}' {{position: {list(self.position_table)}, quat_xyzw: {list(self.quat_xyzw_table)}}}"
                )
                self.moveit2.add_collision_mesh(
                    filepath=self.filepath_table, id=self.mesh_id_table, position=self.position_table, quat_xyzw=self.quat_xyzw_table, frame_id=ur5.base_link_name()
                )

            else:
                # Remove collision mesh
                self.get_logger().info(f"Removing collision mesh with ID '{self.mesh_id_front, self.mesh_id_left, self.mesh_id_right}'")
                self.moveit2.remove_collision_mesh(id=self.mesh_id_front)
                self.moveit2.remove_collision_mesh(id=self.mesh_id_left)
                self.moveit2.remove_collision_mesh(id=self.mesh_id_right)
                self.moveit2.remove_collision_mesh(id=self.mesh_id_arm_base)
                self.moveit2.remove_collision_mesh(id=self.mesh_id_table)

    def servo_status_callback(self, msg):
        self.servo_status = msg.data
        print(self.servo_status)

    def servo_publish_rotation(self, angles):
        '''
        Description:
            To publish the rotation task via the twist message

        Args:
            angles: The corresponding angles for xyz axis
        '''

        ############ Code ############

        self.twist_msg.header.stamp = self.get_clock().now().to_msg()
        self.twist_msg.twist.linear.x = 0.0
        self.twist_msg.twist.linear.y = 0.0
        self.twist_msg.twist.linear.z = 0.0
        self.twist_msg.twist.angular.x = angles[0]
        self.twist_msg.twist.angular.y = angles[1]
        self.twist_msg.twist.angular.z = angles[2]
        self.servo_publisher.publish(self.twist_msg)


    def servo_publish(self, distance):
        '''
        Description:
            To publish the translation task via the twist message

        Args:
            distance: The corresponding distance values for xyz axis
        '''

        ############ Code ############

        self.twist_msg.header.stamp = self.get_clock().now().to_msg()
        self.twist_msg.twist.linear.x = distance[0]
        self.twist_msg.twist.linear.y = distance[1]
        self.twist_msg.twist.linear.z = distance[2]
        self.twist_msg.twist.angular.x = 0.0
        self.twist_msg.twist.angular.y = 0.0
        self.twist_msg.twist.angular.z = 0.0
        self.servo_publisher.publish(self.twist_msg)


    def move_ur5(self, msg):
        '''
        Description:
            To move the robot to the desired place with servo service

        Args:
            msg: The Frames for boxes present which are detected using Aruco Detection
        '''

        ############ Code ############

        self.obj_frames = msg.data
        print("Received Frames: ", self.obj_frames)
        from_frame_rel = 'base_link'
        to_frame_rel = 'obj_1'

        # Spin the node in background thread(s)
        executor = rclpy.executors.MultiThreadedExecutor(2)
        executor.add_node(self)
        executor_thread = Thread(target=executor.spin, daemon=True, args=())
        executor_thread.start()

        print("Ready to move")

        #A client-server communication setup via a client creation for moveit servo task
        service_client = self.create_client(Trigger, '/servo_node/start_servo')

        while not service_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('MoveIt Servo start service not available, waiting...')
        
        #Asynchronus call for the server
        self.trigger_request = Trigger.Request()
        self.future = service_client.call_async(self.trigger_request)

        #Create a client for builtin Server of Gripper magnet on to pick the box
        gripper_control = self.create_client(AttachLink, '/GripperMagnetON')

        #Create a client for builtin Server of Gripper magnet off to place the box
        gripper_control_detach = self.create_client(DetachLink, '/GripperMagnetOFF')

        self.drop_change = 1

        # Lookup transform between base_link and object frames
        for child_frame in self.obj_frames:
            self.future = service_client.call_async(self.trigger_request)
            box_name = "box" + child_frame[4:]
            try:
                transform = self.tf_buffer.lookup_transform(
                    'base_link',
                    child_frame,
                    rclpy.time.Time())
                
                self.get_logger().info(
                    f'Transform from {to_frame_rel} to {from_frame_rel} and {transform.transform._translation}')
            except TransformException as ex:
                self.get_logger().info(
                    f'Could not transform {to_frame_rel} to {from_frame_rel}: {ex}')
                continue
            
            #Extract position and orientation from the transform
            self.position = [transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]
            self.quat_xyzw = [transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w]
            self.euler_angles = euler_from_quaternion(self.quat_xyzw)
            
            #Obtaining the intermediate position for box
            self.position_intermediate = calculate_intermediate_position(self.position, self.quat_xyzw)

            # Move to Position Intermediate to pick the box
            self.get_logger().info(
                f"Moving to {{position: {list(self.position_intermediate)}, quat_xyzw: {list(self.quat_xyzw)}}}"
            )
            self.moveit2.move_to_pose(position=self.position_intermediate, quat_xyzw=self.quat_xyzw, cartesian=self.cartesian)
            self.moveit2.wait_until_executed()

            
            # Lookup transform between base_link and ee frame
            try:
                transform = self.tf_buffer.lookup_transform('base_link', 'tool0', rclpy.time.Time())
            except (self.tf2_ros.LookupException, self.tf2_ros.ConnectivityException, self.tf2_ros.ExtrapolationException):
                self.get_logger().error(f"Failed to lookup transform from base_link to {child_frame}")
                continue

            # Publish the TF between object frame and base_link
            self.ee_position = [transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]
            self.ee_orientation = [transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w]
            self.ee_euler_angles = euler_from_quaternion(self.ee_orientation)
            print("End Effector Position", self.ee_position)

            #Calculating the final distance for box from current position
            self.distance = [
                self.position[0] - self.ee_position[0],
                self.position[1] - self.ee_position[1],
                self.position[2] - self.ee_position[2]
            ] 
            print("Distance: ",self.distance)

            #Request call for service
            self.future = service_client.call_async(self.trigger_request)


            ################   PICK AND PLACE ACTION  ################


            ##### Moving the End effector forward towards the box #####

            while abs(self.distance[0]) > 0.008 or abs(self.distance[1]) > 0.01 or abs(self.distance[2]) > 0.015 and self.servo_status == 0:
                while True:
                    #Gettting the tf data
                    try:
                        transform = self.tf_buffer.lookup_transform('base_link', 'tool0', rclpy.time.Time())
                        print("Got the transform!")
                        break
                    except (self.tf2_ros.LookupException, self.tf2_ros.ConnectivityException, self.tf2_ros.ExtrapolationException):
                        self.get_logger().error(f"Failed to lookup transform from base_link to {child_frame}")
                        print("Waiting for transform!")

                #End effector positioin from the tf data in xyz axis
                self.ee_position = [transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]
                print("End Effector Location ", self.ee_position)

                #Calculation of remaining distance for the box
                self.distance = [
                    self.position[0] - self.ee_position[0],
                    self.position[1] - self.ee_position[1],
                    self.position[2] - self.ee_position[2]
                ] 
                print("Distance: ",self.distance)

                #Translation motion for the distance using moveit servo
                self.servo_publish(self.distance)

            if not self.servo_status == 0:
                # Move to Position Intermediate to pick the box
                self.get_logger().info(
                    f"Moving to {{position: {list(self.position_intermediate)}, quat_xyzw: {list(self.quat_xyzw)}}}"
                )
                self.moveit2.move_to_pose(position=self.position_intermediate, quat_xyzw=self.quat_xyzw, cartesian=self.cartesian)
                self.moveit2.wait_until_executed()


            #####  Gripper Activation to pick the box  #####

            #Wait for service
            while not gripper_control.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('EEF service not available, waiting again...')

            #Taking the box
            req = AttachLink.Request()
            req.model1_name =  box_name    
            req.link1_name  = 'link'       
            req.model2_name =  'ur5'       
            req.link2_name  = 'wrist_3_link'  

            gripper_control.call_async(req)

            try:
                transform = self.tf_buffer.lookup_transform('base_link', 'tool0', rclpy.time.Time())
            except (self.tf2_ros.LookupException, self.tf2_ros.ConnectivityException, self.tf2_ros.ExtrapolationException):
                self.get_logger().error(f"Failed to lookup transform from base_link to {child_frame}")
                continue

            # Publish the TF between end effector and base_link
            self.ee_position = [transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]
            print("End Effector Location ", self.ee_position)

            self.distance = [
                self.position_intermediate[0] - self.ee_position[0],
                self.position_intermediate[1] - self.ee_position[1],
                self.position_intermediate[2] - self.ee_position[2]
            ]            
            print("Distance ",self.distance)

            #Planning for the task of placing the box
            self.future = service_client.call_async(self.trigger_request)

            #####  Moving the End effector backward away from the box  #####

            while abs(self.distance[0]) > 0.01 or abs(self.distance[1]) > 0.01 or abs(self.distance[2]) > 0.015 and self.servo_status == 0:
                while True:
                    #Gettting the tf data
                    try:
                        transform = self.tf_buffer.lookup_transform('base_link', 'tool0', rclpy.time.Time())
                        print("Got the transform!")
                        break
                    except (self.tf2_ros.LookupException, self.tf2_ros.ConnectivityException, self.tf2_ros.ExtrapolationException):
                        self.get_logger().error(f"Failed to lookup transform from base_link to {child_frame}")
                        print("Waiting for transform!")

                #End effector positioin from the tf data in xyz axis
                self.ee_position = [transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]
                print("End Effector Location ", self.ee_position)

                #Calculation of remaining distance from the box
                self.distance = [
                    self.position_intermediate[0] - self.ee_position[0],
                    self.position_intermediate[1] - self.ee_position[1],
                    self.position_intermediate[2] - self.ee_position[2]
                ]    

                print(self.distance)

                #Translation motion for the distance using moveit servo
                self.servo_publish(self.distance)

            if not self.servo_status == 0:
                # Move to Position Intermediate to pick the box
                self.get_logger().info(
                    f"Moving to {{position: {list(self.position_intermediate)}, quat_xyzw: {list(self.quat_xyzw)}}}"
                )
                self.moveit2.move_to_pose(position=self.position_intermediate, quat_xyzw=self.quat_xyzw, cartesian=self.cartesian)
                self.moveit2.wait_until_executed()


            # Move to Position Center
            self.get_logger().info(
                f"Moving to {{position: {list(self.position_d2)}, quat_xyzw: {list(self.quat_xyzw_d2)}}}"
            )
            self.moveit2.move_to_configuration(self.joint_positions_center)
            self.moveit2.wait_until_executed()


            ##### Place the box  #####

            if self.drop_change == 1:
                # Move to Position D
                self.get_logger().info(
                    f"Moving to {{position: {list(self.position_d)}, quat_xyzw: {list(self.quat_xyzw_d)}}}"
                )
                #self.moveit2.move_to_pose(position=self.position_d, quat_xyzw=self.quat_xyzw_d, cartesian=self.cartesian)
                self.moveit2.move_to_configuration(self.joint_positions_d2)
                self.moveit2.wait_until_executed()
                self.drop_change = 2

            elif self.drop_change == 2:
                # Move to Position D
                self.get_logger().info(
                    f"Moving to {{position: {list(self.position_d)}, quat_xyzw: {list(self.quat_xyzw_d)}}}"
                )
                #self.moveit2.move_to_pose(position=self.position_d, quat_xyzw=self.quat_xyzw_d, cartesian=self.cartesian)
                self.moveit2.move_to_configuration(self.joint_positions_d)
                self.moveit2.wait_until_executed()
                self.drop_change = 3

            else:
                # Move to Position D
                self.get_logger().info(
                    f"Moving to {{position: {list(self.position_d)}, quat_xyzw: {list(self.quat_xyzw_d)}}}"
                )
                #self.moveit2.move_to_pose(position=self.position_d, quat_xyzw=self.quat_xyzw_d, cartesian=self.cartesian)
                self.moveit2.move_to_configuration(self.joint_positions_d3)
                self.moveit2.wait_until_executed()
                self.drop_change = 1


            while not gripper_control.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('EEF service not available, waiting again...')

            #Droping the box 
            req_detach = DetachLink.Request()
            req_detach.model1_name =  box_name    
            req_detach.link1_name  = 'link'       
            req_detach.model2_name =  'ur5'       
            req_detach.link2_name  = 'wrist_3_link'  

            gripper_control_detach.call_async(req_detach)

            # Move to Position Center
            self.get_logger().info(
                f"Moving to {{position: {list(self.position_d2)}, quat_xyzw: {list(self.quat_xyzw_d2)}}}"
            )
            self.moveit2.move_to_configuration(self.joint_positions_center)
            self.moveit2.wait_until_executed()
        


    def on_timer(self):
        '''
        Description:
            To view the value and status of tf data obtained and print it in the terminal
        '''

        ############ Code ############

        # Parent frame
        from_frame_rel = 'base_link' 
        #Child frame
        to_frame_rel = 'obj_1' 

        #Finding the Lookup transform between the frames if not exit
        try:
            t = self.tf_buffer.lookup_transform(
                'base_link',
                'obj_1',
                rclpy.time.Time())
            
            self.get_logger().info(
                f'Transform from {to_frame_rel} to {from_frame_rel} and {t.transform._translation}')
        except TransformException as ex:
            self.get_logger().info(
                f'Could not transform {to_frame_rel} to {from_frame_rel}: {ex}')
            return


def main():
    '''
    Description:    Main function which creates a ROS node and spin around for the move_ur5 class to perform it's task
    '''

    # Initialisation
    rclpy.init()    

    # Creating ROS node
    node = rclpy.create_node('task_2a')  

    # Logging the Information
    move_boxes_class = move_boxes()

    #Make the node active until it is manually interrupted
    rclpy.spin(move_boxes_class)

    #Destroy the node after task completion done
    move_boxes_class.destroy_node()

    #Node shutdown after all the things done
    rclpy.shutdown()


if __name__ == '__main__':
    '''
    Description:    If the python interpreter is running that module (the source file) as the main program, 
                    it sets the special __name__ variable to have a value “__main__”. 
                    If this file is being imported from another module, __name__ will be set to the module’s name.
    '''
    main()                                     
