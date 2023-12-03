#!/usr/bin/env python3
"""
Example of adding and removing a collision object with a mesh geometry.
Note: Python module `trimesh` is required for this example (`pip install trimesh`).
`ros2 run pymoveit2 ex_collision_object.py --ros-args -p action:="add" -p position:="[0.5, 0.0, 0.5]" -p quat_xyzw:="[0.0, 0.0, -0.707, 0.707]"`
`ros2 run pymoveit2 ex_collision_object.py --ros-args -p action:="add" -p filepath:="./my_favourity_mesh.stl"`
`ros2 run pymoveit2 ex_collision_object.py --ros-args -p action:="remove"`
"""

from os import path
from threading import Thread

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node

from pymoveit2 import MoveIt2
from pymoveit2.robots import ur5

import numpy as np
from time import sleep

DEFAULT_EXAMPLE_MESH = path.join(
    path.dirname(path.realpath(__file__)), "assets", "rack.stl"
)

def signal_handler(sig, frame):
    # Handle the Ctrl+C signal here
    global exit_requested
    exit_requested = True

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
    node = Node("ex_collision_object")

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

    print(ur5.base_link_name())

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

    for i in range(2):
        if "add" == action:
            # Add collision mesh rack_front
            node.get_logger().info(
                f"Adding collision mesh '{filepath}' {{position: {list(position_rack_front)}, quat_xyzw: {list(quat_xyzw_rack_front)}}}"
            )

            # print(ur5.base_link_name())
            moveit2.add_collision_mesh(
                filepath=filepath, id=mesh_id_front, position=position_rack_front, quat_xyzw=quat_xyzw_rack_front, frame_id=ur5.base_link_name()
            )

            # Add collision mesh rack_left
            node.get_logger().info(
                f"Adding collision mesh '{filepath}' {{position: {list(position_rack_left)}, quat_xyzw: {list(quat_xyzw_rack_left)}}}"
            )
            # print(ur5.base_link_name())
            moveit2.add_collision_mesh(
                filepath=filepath, id=mesh_id_left, position=position_rack_left, quat_xyzw=quat_xyzw_rack_left, frame_id=ur5.base_link_name()
            )
            
            # Add collision mesh rack_right
            node.get_logger().info(
                f"Adding collision mesh '{filepath}' {{position: {list(position_rack_right)}, quat_xyzw: {list(quat_xyzw_rack_right)}}}"
            )
            # print(ur5.base_link_name())
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


    rclpy.shutdown()
    exit(0)


if __name__ == "__main__":
    main()
