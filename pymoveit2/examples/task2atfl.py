#!/usr/bin/env python3

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
    Convert an Euler angle to a quaternion.
    
    Input
        :param roll: The roll (rotation around x-axis) angle in radians.
        :param pitch: The pitch (rotation around y-axis) angle in radians.
        :param yaw: The yaw (rotation around z-axis) angle in radians.
    
    Output
        :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
    """

    ############ Function VARIABLES ############
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

import math

def euler_from_quaternion(quat):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        x = quat[0]
        y = quat[1]
        z = quat[2]
        w = quat[3]
        
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        euler_angles = [roll_x, pitch_y, yaw_z]
        
        return euler_angles # in radians


def calculate_intermediate_position(position, quat_xyzw):
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

    def __init__(self):
        super().__init__('task2a')

        self.move_ur5_sub = self.create_subscription(List, '/obj_frames', self.move_ur5, 10)
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
        



        # Get parameters
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
        self.position_d2 = [0.15, 0.108, 0.6]
        self.quat_xyzw_d2 = quaternion_from_euler(np.pi/2, 0.0, np.pi/2)

        self.cartesian = False

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

        # Call on_timer function every second
        #self.timer = self.create_timer(1.0, self.move_ur5)
        
        #self.move_ur5()
        #Stopping the Motion
        #rclpy.shutdown()
        #exit(0)

    def servo_publish_rotation(self, angles):
        self.twist_msg.header.stamp = self.get_clock().now().to_msg()
        self.twist_msg.twist.linear.x = 0.0
        self.twist_msg.twist.linear.y = 0.0
        self.twist_msg.twist.linear.z = 0.0
        self.twist_msg.twist.angular.x = angles[0]
        self.twist_msg.twist.angular.y = angles[1]
        self.twist_msg.twist.angular.z = angles[2]
        self.servo_publisher.publish(self.twist_msg)

    def servo_publish(self, distance):
        self.twist_msg.header.stamp = self.get_clock().now().to_msg()
        self.twist_msg.twist.linear.x = distance[0]
        self.twist_msg.twist.linear.y = distance[1]
        self.twist_msg.twist.linear.z = distance[2]
        self.twist_msg.twist.angular.x = 0.0
        self.twist_msg.twist.angular.y = 0.0
        self.twist_msg.twist.angular.z = 0.0
        self.servo_publisher.publish(self.twist_msg)

    def move_ur5(self, msg):
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

        service_client = self.create_client(Trigger, '/servo_node/start_servo')

        while not service_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('MoveIt Servo start service not available, waiting...')
        
        self.trigger_request = Trigger.Request()
        self.future = service_client.call_async(self.trigger_request)

        gripper_control = self.create_client(AttachLink, '/GripperMagnetON')
        gripper_control_detach = self.create_client(DetachLink, '/GripperMagnetOFF')
        # Lookup transform between base_link and obj frame
        self.drop_change = False
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
            
            self.position = [transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]
            self.quat_xyzw = [transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w]
            self.euler_angles = euler_from_quaternion(self.quat_xyzw)
            
            self.position_intermediate = calculate_intermediate_position(self.position, self.quat_xyzw)





            '''# Lookup transform between base_link and ee frame
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

            self.distance = [
                self.position_intermediate[0] - self.ee_position[0],
                self.position_intermediate[1] - self.ee_position[1],
                self.position_intermediate[2] - self.ee_position[2]
            ] 
            print("Distance: ",self.distance)

            self.future = service_client.call_async(self.trigger_request)

            while abs(self.distance[0]) > 0.008 or abs(self.distance[1]) > 0.01 or abs(self.distance[2]) > 0.015:
                while True:
                    try:
                        transform = self.tf_buffer.lookup_transform('base_link', 'tool0', rclpy.time.Time())
                        print("Got the transform!")
                        break
                    except (self.tf2_ros.LookupException, self.tf2_ros.ConnectivityException, self.tf2_ros.ExtrapolationException):
                        self.get_logger().error(f"Failed to lookup transform from base_link to {child_frame}")
                        print("Waiting for transform!")

                self.ee_position = [transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]
                print("End Effector Location ", self.ee_position)

                self.distance = [
                    self.position_intermediate[0] - self.ee_position[0],
                    self.position_intermediate[1] - self.ee_position[1],
                    self.position_intermediate[2] - self.ee_position[2]
                ] 
                print("Distance: ",self.distance)
                self.servo_publish(self.distance)'''

            # Move to Position Intermediate
            self.get_logger().info(
                f"Moving to {{position: {list(self.position_intermediate)}, quat_xyzw: {list(self.quat_xyzw)}}}"
            )
            self.moveit2.move_to_pose(position=self.position_intermediate, quat_xyzw=self.quat_xyzw, cartesian=self.cartesian)
            self.moveit2.wait_until_executed()

            '''# Lookup transform between base_link and ee frame
            try:
                transform = self.tf_buffer.lookup_transform('base_link', 'tool0', rclpy.time.Time())
            except (self.tf2_ros.LookupException, self.tf2_ros.ConnectivityException, self.tf2_ros.ExtrapolationException):
                self.get_logger().error(f"Failed to lookup transform from base_link to {child_frame}")
                continue

            # Publish the TF between object frame and base_link
            self.ee_position = [transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]
            self.ee_orientation = [transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w]
            self.ee_euler_angles = euler_from_quaternion(self.ee_orientation)
            print("End Effector Orientation", self.ee_orientation)

            self.angles = [
                self.euler_angles[0] - self.ee_euler_angles[0],
                self.euler_angles[1] - self.ee_euler_angles[1],
                self.euler_angles[2] - self.ee_euler_angles[2]
            ]            
            print("Angles ",self.angles)

            self.future = service_client.call_async(self.trigger_request)

            while abs(self.angles[0]) > 0.008 or abs(self.angles[1]) > 0.008 or abs(self.angles[2]) > 0.008:
                while True:
                    try:
                        transform = self.tf_buffer.lookup_transform('base_link', 'tool0', rclpy.time.Time())
                        print("Got the transform!")
                        break
                    except (self.tf2_ros.LookupException, self.tf2_ros.ConnectivityException, self.tf2_ros.ExtrapolationException):
                        self.get_logger().error(f"Failed to lookup transform from base_link to {child_frame}")
                        print("Waiting for transform!")

                self.ee_position = [transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]
                self.ee_orientation = [transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w]
                self.ee_euler_angles = euler_from_quaternion(self.ee_orientation)
                print("End Effector Orientation", self.ee_orientation)

                self.angles = [
                    self.euler_angles[0] - self.ee_euler_angles[0],
                    self.euler_angles[1] - self.ee_euler_angles[1],
                    self.euler_angles[2] - self.ee_euler_angles[2]
                ]              
                print("Angles ",self.angles)

                self.servo_publish_rotation(self.angles)'''
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

            self.distance = [
                self.position[0] - self.ee_position[0],
                self.position[1] - self.ee_position[1],
                self.position[2] - self.ee_position[2]
            ] 
            print("Distance: ",self.distance)

            self.future = service_client.call_async(self.trigger_request)

            while abs(self.distance[0]) > 0.008 or abs(self.distance[1]) > 0.01 or abs(self.distance[2]) > 0.015:
                while True:
                    try:
                        transform = self.tf_buffer.lookup_transform('base_link', 'tool0', rclpy.time.Time())
                        print("Got the transform!")
                        break
                    except (self.tf2_ros.LookupException, self.tf2_ros.ConnectivityException, self.tf2_ros.ExtrapolationException):
                        self.get_logger().error(f"Failed to lookup transform from base_link to {child_frame}")
                        print("Waiting for transform!")

                self.ee_position = [transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]
                print("End Effector Location ", self.ee_position)

                self.distance = [
                    self.position[0] - self.ee_position[0],
                    self.position[1] - self.ee_position[1],
                    self.position[2] - self.ee_position[2]
                ] 
                print("Distance: ",self.distance)
                self.servo_publish(self.distance)

            # Move to Position of the Box
            '''self.get_logger().info(
                f"Moving to {{position: {list(self.position)}, quat_xyzw: {list(self.quat_xyzw)}}}"
            )
            self.moveit2.move_to_pose(position=self.position, quat_xyzw=self.quat_xyzw, cartesian=self.cartesian)
            print("Executing")
            self.moveit2.wait_until_executed()
            print("Executed")'''


            while not gripper_control.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('EEF service not available, waiting again...')

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

            # Publish the TF between object frame and base_link
            self.ee_position = [transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]
            print("End Effector Location ", self.ee_position)

            self.distance = [
                self.position_intermediate[0] - self.ee_position[0],
                self.position_intermediate[1] - self.ee_position[1],
                self.position_intermediate[2] - self.ee_position[2]
            ]            
            print("Distance ",self.distance)

            self.future = service_client.call_async(self.trigger_request)

            end_time = time.time() + 10
            while abs(self.distance[0]) > 0.01 or abs(self.distance[1]) > 0.01 or abs(self.distance[2]) > 0.015 and time.time()<end_time:
                while True:
                    try:
                        transform = self.tf_buffer.lookup_transform('base_link', 'tool0', rclpy.time.Time())
                        print("Got the transform!")
                        break
                    except (self.tf2_ros.LookupException, self.tf2_ros.ConnectivityException, self.tf2_ros.ExtrapolationException):
                        self.get_logger().error(f"Failed to lookup transform from base_link to {child_frame}")
                        print("Waiting for transform!")

                self.ee_position = [transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]
                print("End Effector Location ", self.ee_position)

                self.distance = [
                    self.position_intermediate[0] - self.ee_position[0],
                    self.position_intermediate[1] - self.ee_position[1],
                    self.position_intermediate[2] - self.ee_position[2]
                ]    

                print(self.distance)
                self.servo_publish(self.distance)

            # Move to Position D
            '''self.get_logger().info(
                f"Moving to {{position: {list(self.position_d2)}, quat_xyzw: {list(self.quat_xyzw_d2)}}}"
            )
            self.moveit2.move_to_pose(position=self.position_d2, quat_xyzw=self.quat_xyzw, cartesian=self.cartesian)
            self.moveit2.wait_until_executed()'''

            # Move to Position Center
            self.get_logger().info(
                f"Moving to {{position: {list(self.position_d2)}, quat_xyzw: {list(self.quat_xyzw_d2)}}}"
            )
            self.moveit2.move_to_configuration(self.joint_positions_center)
            self.moveit2.wait_until_executed()


            if not self.drop_change:
                # Move to Position D
                self.get_logger().info(
                    f"Moving to {{position: {list(self.position_d)}, quat_xyzw: {list(self.quat_xyzw_d)}}}"
                )
                #self.moveit2.move_to_pose(position=self.position_d, quat_xyzw=self.quat_xyzw_d, cartesian=self.cartesian)
                self.moveit2.move_to_configuration(self.joint_positions_d2)
                self.moveit2.wait_until_executed()

                self.drop_change = True
            else:
                # Move to Position D
                self.get_logger().info(
                    f"Moving to {{position: {list(self.position_d)}, quat_xyzw: {list(self.quat_xyzw_d)}}}"
                )
                #self.moveit2.move_to_pose(position=self.position_d, quat_xyzw=self.quat_xyzw_d, cartesian=self.cartesian)
                self.moveit2.move_to_configuration(self.joint_positions_d)
                self.moveit2.wait_until_executed()


            while not gripper_control.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('EEF service not available, waiting again...')

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

        # Move to Initial Position
        '''self.get_logger().info(
            f"Moving to {{position: {list(self.position_initial)}, quat_xyzw: {list(self.quat_xyzw_initial)}}}"
        )
        self.moveit2.move_to_pose(position=self.position_initial, quat_xyzw=self.quat_xyzw_initial, cartesian=self.cartesian)
        self.moveit2.wait_until_executed()
        return'''
        

    def on_timer(self):
        # Store frame names in variables that will be used to
        # compute transformations
        from_frame_rel = 'base_link'
        to_frame_rel = 'obj_1'

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
    rclpy.init()    

    # Creating ROS node
    node = rclpy.create_node('task_2a')  

    # Logging the Information
    #node.get_logger().info('Node created: task_2a') 
    move_boxes_class = move_boxes()


    rclpy.spin(move_boxes_class)

    move_boxes_class.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    '''
    Description:    If the python interpreter is running that module (the source file) as the main program, 
                    it sets the special __name__ variable to have a value “__main__”. 
                    If this file is being imported from another module, __name__ will be set to the module’s name.
    '''
    main()                                     
