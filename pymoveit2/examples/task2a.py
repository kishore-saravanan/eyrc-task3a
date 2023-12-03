#!/usr/bin/env python3
"""
Example of moving to a pose goal.
`ros2 run pymoveit2 ex_pose_goal.py --ros-args -p position:="[0.25, 0.0, 1.0]" -p quat_xyzw:="[0.0, 0.0, 0.0, 1.0]" -p cartesian:=False`
"""


'''
*****************************************************************************************
*
*        		===============================================
*           		    Cosmo Logistic (CL) Theme (eYRC 2023-24)
*        		===============================================
*
*  This script should be used to implement Task 1B of Cosmo Logistic (CL) Theme (eYRC 2023-24).
*
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''

# Team ID	:       [ eYRC#CL#1243 ]
# Author List	:	[ Kishore , Pragadheesh, Vimal, Jagannathan ]
# Filename	:	[ task2a.py ]
# Functions	:	[ main, quaternion_from_euler ]


################### IMPORT MODULES #######################


from os import path
from threading import Thread

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node

from pymoveit2 import MoveIt2
from pymoveit2.robots import ur5

import numpy as np
from time import sleep
import tf2_ros
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation as R

node = None
tf_buffer = None
moveit2 = None
cartesian = None
position_d = None
quat_xyzw_d = None
position_initial = None
quat_xyzw_initial = None


from linkattacher_msgs.srv import AttachLink, DetachLink


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


def on_timer():
    global node, tf_buffer, moveit2, cartesian, position_d, quat_xyzw_d, position_initial, quat_xyzw_initial
    obj_frames = ['obj_1','obj_3','obj_49']

    # Lookup transform between base_link and obj frame
    for child_frame in obj_frames:
        try: 
            transform = tf_buffer.lookup_transform('base_link', child_frame, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0))
            #transform = TransformStamped()

            #transform.header.stamp = node.get_clock().now().to_msg()
            #transform.header.frame_id = 'base_link'
            #transform.child_frame_id = child_frame
            node.get_logger().info(f"Created lookup the transform from base_link to {child_frame}")


        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            node.get_logger().error(f"Failed to lookup the transform from base_link to {child_frame}")
            continue
        
        position = [transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]
        orientation = [transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w]

        # Move to Position of the Box
        node.get_logger().info(
            f"Moving to {{position: {list(position)}, quat_xyzw: {list(orientation)}}}"
        )
        moveit2.move_to_pose(position=position, quat_xyzw=orientation, cartesian=cartesian)
        moveit2.wait_until_executed()

        '''while not gripper_control.wait_for_service(timeout_sec=1.0):
            node.get_logger().info('EEF service not available, waiting again...')

        req = AttachLink.Request()
        req.model1_name =  child_frame    
        req.link1_name  = 'link'       
        req.model2_name =  'ur5'       
        req.link2_name  = 'wrist_3_link'  

        gripper_control.call_async(req)'''
        
        
        # Move to Position D
        node.get_logger().info(
            f"Moving to {{position: {list(position_d)}, quat_xyzw: {list(quat_xyzw_d)}}}"
        )
        moveit2.move_to_pose(position=position_d, quat_xyzw=quat_xyzw_d, cartesian=cartesian)
        moveit2.wait_until_executed()

        '''while not gripper_control_off.wait_for_service(timeout_sec=1.0):
            node.get_logger().info('EEF service not available, waiting again...')

        req = DetachLink.Request()
        req.model1_name =  child_frame    
        req.link1_name  = 'link'       
        req.model2_name =  'ur5'       
        req.link2_name  = 'wrist_3_link'  

        gripper_control.call_async(req)'''
    
    # Move to Initial Position
    node.get_logger().info(
        f"Moving to {{position: {list(position_initial)}, quat_xyzw: {list(quat_xyzw_initial)}}}"
    )
    moveit2.move_to_pose(position=position_initial, quat_xyzw=quat_xyzw_initial, cartesian=cartesian)
    moveit2.wait_until_executed()

    

def main():
    global node, tf_buffer, moveit2, cartesian, position_d, quat_xyzw_d, position_initial, quat_xyzw_initial

    rclpy.init()

    # Create node for this example
    node = Node("task2a")

    # Declare parameter for joint positions
    node.declare_parameter(
        "filepath",
        "",
    )
    node.declare_parameter(
        "action",
        "add",
    )

    tf_buffer = tf2_ros.buffer.Buffer() 


    node.declare_parameter("position_rack_front", [0.57, 0.05, -0.56])
    node.declare_parameter("quat_xyzw_rack_front", quaternion_from_euler(np.pi, np.pi, 0.0))

    node.declare_parameter("position_rack_left", [0.29, 0.66, -0.56])
    node.declare_parameter("quat_xyzw_rack_left", quaternion_from_euler(0.0, 0.0, np.pi/2))

    node.declare_parameter("position_rack_right", [0.29, -0.66, -0.56])
    node.declare_parameter("quat_xyzw_rack_right", quaternion_from_euler(0.0, 0.0, np.pi/2))

    node.declare_parameter("position_arm_base", [-0.2, 0.00, -0.58])
    node.declare_parameter("quat_xyzw_arm_base", quaternion_from_euler(0.0, 0.0, np.pi))

    node.declare_parameter("position_table", [-0.8, 0.00, -0.60])
    node.declare_parameter("quat_xyzw_table", quaternion_from_euler(0.0, 0.0, 0.0))

    # Declare parameters for position and orientation for initial position
    node.declare_parameter("position_initial", [0.176, 0.108, 0.470])
    node.declare_parameter("quat_xyzw_initial", quaternion_from_euler(np.pi/2, 0.0, np.pi/2))

    # Declare parameters for position and orientation for position P1
    node.declare_parameter("position_p1", [0.35, 0.10, 0.68])
    node.declare_parameter("quat_xyzw_p1", quaternion_from_euler(np.pi/2, 0.0, np.pi/2))

    # Declare parameters for position and orientation for D
    node.declare_parameter("position_d", [-0.37, 0.12, 0.397])
    node.declare_parameter("quat_xyzw_d", quaternion_from_euler(np.pi, np.pi/2, 0.0))

    # Declare parameters for position and orientation for P2
    node.declare_parameter("position_p2", [0.194, -0.43, 0.701])
    node.declare_parameter("quat_xyzw_p2", quaternion_from_euler(np.pi/2, 0.0, 0.0))

    node.declare_parameter("cartesian", False)

    # Create callback group that allows execution of callbacks in parallel without restrictions
    callback_group = ReentrantCallbackGroup()

    # Create MoveIt 2 interface
    moveit2 = MoveIt2(
        node=node,
        joint_names=ur5.joint_names(),
        base_link_name=ur5.base_link_name(),
        end_effector_name=ur5.end_effector_name(),
        group_name=ur5.MOVE_GROUP_ARM,
        callback_group=callback_group,
    )

    # Spin the node in background thread(s)
    executor = rclpy.executors.MultiThreadedExecutor(2)
    executor.add_node(node)
    executor_thread = Thread(target=executor.spin, daemon=True, args=())
    executor_thread.start()

    # Get parameters
    filepath = ""
    action = "add"
    position_rack_front = [0.57, 0.05, -0.56]
    quat_xyzw_rack_front = quaternion_from_euler(np.pi, np.pi, 0.0)
    position_rack_left = [0.29, 0.66, -0.56]
    quat_xyzw_rack_left = quaternion_from_euler(0.0, 0.0, np.pi/2)
    position_rack_right = [0.29, -0.66, -0.56]
    quat_xyzw_rack_right = quaternion_from_euler(0.0, 0.0, np.pi/2)
    position_arm_base = [-0.2, 0.00, -0.58]
    quat_xyzw_arm_base = quaternion_from_euler(0.0, 0.0, np.pi)
    position_table = [-0.8, 0.00, -0.60]
    quat_xyzw_table = quaternion_from_euler(0.0, 0.0, 0.0)
    

    position_initial = [0.176, 0.108, 0.470]
    quat_xyzw_initial = quaternion_from_euler(np.pi/2, 0.0, np.pi/2)

    position_p1 = node.get_parameter("position_p1").get_parameter_value().double_array_value
    quat_xyzw_p1 = node.get_parameter("quat_xyzw_p1").get_parameter_value().double_array_value

    position_d = [-0.37, 0.12, 0.397]
    quat_xyzw_d = quaternion_from_euler(np.pi, np.pi/2, 0.0)

    position_p2 = node.get_parameter("position_p2").get_parameter_value().double_array_value
    quat_xyzw_p2 = node.get_parameter("quat_xyzw_p2").get_parameter_value().double_array_value

    cartesian = node.get_parameter("cartesian").get_parameter_value().bool_value

    # Use the default example mesh if invalid
    if not filepath:
        node.get_logger().info(f"Using the default example mesh file")
        filepath = DEFAULT_EXAMPLE_MESH
        filepath_arm_base = ARM_BASE_MESH
        filepath_table = TABLE_MESH

    # Make sure the mesh file exists
    if not path.exists(filepath):
        node.get_logger().error(f"File '{filepath}' does not exist")
        rclpy.shutdown()
        exit(1)

    # Determine ID of the collision mesh
    mesh_id_front = "_".join(path.basename(filepath).split(".")) + "_front"
    mesh_id_left = "_".join(path.basename(filepath).split(".")) + "_left"
    mesh_id_right = "_".join(path.basename(filepath).split(".")) + "_right"
    mesh_id_arm_base = "_".join(path.basename(filepath_arm_base).split("."))
    mesh_id_table = "_".join(path.basename(filepath_table).split("."))

    #To get the accuracy in Collision 
    for i in range(2):
        if "add" == action:
        
            # Add collision mesh rack_front
            node.get_logger().info(
                f"Adding collision mesh '{filepath}' {{position: {list(position_rack_front)}, quat_xyzw: {list(quat_xyzw_rack_front)}}}"
            )

            moveit2.add_collision_mesh(
                filepath=filepath, id=mesh_id_front, position=position_rack_front, quat_xyzw=quat_xyzw_rack_front, frame_id=ur5.base_link_name()
            )

            # Add collision mesh rack_left
            node.get_logger().info(
                f"Adding collision mesh '{filepath}' {{position: {list(position_rack_left)}, quat_xyzw: {list(quat_xyzw_rack_left)}}}"
            )
            moveit2.add_collision_mesh(
                filepath=filepath, id=mesh_id_left, position=position_rack_left, quat_xyzw=quat_xyzw_rack_left, frame_id=ur5.base_link_name()
            )
            
            # Add collision mesh rack_right
            node.get_logger().info(
                f"Adding collision mesh '{filepath}' {{position: {list(position_rack_right)}, quat_xyzw: {list(quat_xyzw_rack_right)}}}"
            )
            moveit2.add_collision_mesh(
                filepath=filepath, id=mesh_id_right, position=position_rack_right, quat_xyzw=quat_xyzw_rack_right, frame_id=ur5.base_link_name()
            )

            # Add collision mesh arm_base
            node.get_logger().info(
                f"Adding collision mesh '{filepath_arm_base}' {{position: {list(position_arm_base)}, quat_xyzw: {list(quat_xyzw_arm_base)}}}"
            )
            moveit2.add_collision_mesh(
                filepath=filepath_arm_base, id=mesh_id_arm_base, position=position_arm_base, quat_xyzw=quat_xyzw_arm_base, frame_id=ur5.base_link_name()
            )

            # Add collision mesh table
            node.get_logger().info(
                f"Adding collision mesh '{filepath_table}' {{position: {list(position_table)}, quat_xyzw: {list(quat_xyzw_table)}}}"
            )
            moveit2.add_collision_mesh(
                filepath=filepath_table, id=mesh_id_table, position=position_table, quat_xyzw=quat_xyzw_table, frame_id=ur5.base_link_name()
            )

        else:
            # Remove collision mesh
            node.get_logger().info(f"Removing collision mesh with ID '{mesh_id_front, mesh_id_left, mesh_id_right}'")
            moveit2.remove_collision_mesh(id=mesh_id_front)
            moveit2.remove_collision_mesh(id=mesh_id_left)
            moveit2.remove_collision_mesh(id=mesh_id_right)
            moveit2.remove_collision_mesh(id=mesh_id_arm_base)
            moveit2.remove_collision_mesh(id=mesh_id_table)

    #timer = node.create_timer(1.0, on_timer)

        # Move to Position P1
    node.get_logger().info(
        f"Moving to {{position: {list(position_p1)}, quat_xyzw: {list(quat_xyzw_p1)}}}"
    )
    moveit2.move_to_pose(position=position_p1, quat_xyzw=quat_xyzw_p1, cartesian=cartesian)
    moveit2.wait_until_executed()

    # Move to Position D
    node.get_logger().info(
        f"Moving to {{position: {list(position_d)}, quat_xyzw: {list(quat_xyzw_d)}}}"
    )
    moveit2.move_to_pose(position=position_d, quat_xyzw=quat_xyzw_d, cartesian=cartesian)
    moveit2.wait_until_executed()

    # Move to Position P2
    node.get_logger().info(
        f"Moving to {{position: {list(position_p2)}, quat_xyzw: {list(quat_xyzw_p2)}}}"
    )
    moveit2.move_to_pose(position=position_p2, quat_xyzw=quat_xyzw_p2, cartesian=cartesian)
    moveit2.wait_until_executed()

    # Move to Position D
    node.get_logger().info(
        f"Moving to {{position: {list(position_d)}, quat_xyzw: {list(quat_xyzw_d)}}}"
    )
    moveit2.move_to_pose(position=position_d, quat_xyzw=quat_xyzw_d, cartesian=cartesian)
    moveit2.wait_until_executed()


    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
        #Stopping the Motion
        rclpy.shutdown()
        exit(0)


if __name__ == "__main__":
    '''
    Description:    If the python interpreter is running that module (the source file) as the main program, 
                    it sets the special __name__ variable to have a value “__main__”. 
                    If this file is being imported from another module, __name__ will be set to the module’s name.
    '''
    main()