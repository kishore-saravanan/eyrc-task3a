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
# Filename	:	[ task1b.py ]
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


#Configuring the Path of the Mesh file 
DEFAULT_EXAMPLE_MESH = path.join(
    path.dirname(path.realpath(__file__)), "assets", "rack.stl"
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
    

def main():
    rclpy.init()

    # Create node for this example
    node = Node("task1b")


    # Declare parameter for joint positions
    node.declare_parameter(
        "filepath",
        "",
    )
    node.declare_parameter(
        "action",
        "add",
    )
    node.declare_parameter("position_rack_front", [0.57, 0.05, -0.56])
    node.declare_parameter("quat_xyzw_rack_front", quaternion_from_euler(np.pi, np.pi, 0.0))

    node.declare_parameter("position_rack_left", [0.29, 0.66, -0.56])
    node.declare_parameter("quat_xyzw_rack_left", quaternion_from_euler(0.0, 0.0, np.pi/2))

    node.declare_parameter("position_rack_right", [0.29, -0.66, -0.56])
    node.declare_parameter("quat_xyzw_rack_right", quaternion_from_euler(0.0, 0.0, np.pi/2))

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
    filepath = node.get_parameter("filepath").get_parameter_value().string_value
    action = node.get_parameter("action").get_parameter_value().string_value
    position_rack_front = node.get_parameter("position_rack_front").get_parameter_value().double_array_value
    quat_xyzw_rack_front = node.get_parameter("quat_xyzw_rack_front").get_parameter_value().double_array_value
    position_rack_left = node.get_parameter("position_rack_left").get_parameter_value().double_array_value
    quat_xyzw_rack_left = node.get_parameter("quat_xyzw_rack_left").get_parameter_value().double_array_value
    position_rack_right = node.get_parameter("position_rack_right").get_parameter_value().double_array_value
    quat_xyzw_rack_right = node.get_parameter("quat_xyzw_rack_right").get_parameter_value().double_array_value

    position_initial = node.get_parameter("position_initial").get_parameter_value().double_array_value
    quat_xyzw_initial = node.get_parameter("quat_xyzw_initial").get_parameter_value().double_array_value

    position_p1 = node.get_parameter("position_p1").get_parameter_value().double_array_value
    quat_xyzw_p1 = node.get_parameter("quat_xyzw_p1").get_parameter_value().double_array_value

    position_d = node.get_parameter("position_d").get_parameter_value().double_array_value
    quat_xyzw_d = node.get_parameter("quat_xyzw_d").get_parameter_value().double_array_value

    position_p2 = node.get_parameter("position_p2").get_parameter_value().double_array_value
    quat_xyzw_p2 = node.get_parameter("quat_xyzw_p2").get_parameter_value().double_array_value

    cartesian = node.get_parameter("cartesian").get_parameter_value().bool_value

    # Use the default example mesh if invalid
    if not filepath:
        node.get_logger().info(f"Using the default example mesh file")
        filepath = DEFAULT_EXAMPLE_MESH

    # Make sure the mesh file exists
    if not path.exists(filepath):
        node.get_logger().error(f"File '{filepath}' does not exist")
        rclpy.shutdown()
        exit(1)

    # Determine ID of the collision mesh
    mesh_id_front = "_".join(path.basename(filepath).split(".")) + "_front"
    mesh_id_left = "_".join(path.basename(filepath).split(".")) + "_left"
    mesh_id_right = "_".join(path.basename(filepath).split(".")) + "_right"

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

            sleep(2)

        else:
            # Remove collision mesh
            node.get_logger().info(f"Removing collision mesh with ID '{mesh_id_front, mesh_id_left, mesh_id_right}'")
            moveit2.remove_collision_mesh(id=mesh_id_front)
            moveit2.remove_collision_mesh(id=mesh_id_left)
            moveit2.remove_collision_mesh(id=mesh_id_right)


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
    
    # Move to Initial Position
    node.get_logger().info(
        f"Moving to {{position: {list(position_initial)}, quat_xyzw: {list(quat_xyzw_initial)}}}"
    )
    moveit2.move_to_pose(position=position_initial, quat_xyzw=quat_xyzw_initial, cartesian=cartesian)
    moveit2.wait_until_executed()

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