#!/usr/bin/env python3

'''
*****************************************************************************************
*
*        		===============================================
*           		    Cosmo Logistic (CL) Theme (eYRC 2023-24)
*        		===============================================
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
# Author List:		[ Kishore S, Pragadheesh RJ, Vimal VK, Jagannathan M]
# Filename:		    1task2a.py
# Functions:
#                   Main Functions:
#			        [ main, calculate_rectangle_area, detect_aruco, quaternion_from_euler]
#                   Functions of aruco_tf class :
#                   [depthimagecb, colorimagecb, publish_tf, process_image]
# Nodes:		    
#                   Publishing Topics  - [ /tf, /obj_frames ]
#                   Subscribing Topics - [ /camera/aligned_depth_to_color/image_raw, /camer_info, /camera/color/image_raw]


################### IMPORT MODULES #######################

import rclpy
import sys
import cv2
import math
import tf2_ros
import numpy as np
from rclpy.node import Node
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CompressedImage, Image
from ros2_interfaces_cpp.msg import List

##################### FUNCTION DEFINITIONS #######################



def calculate_rectangle_area(coordinates):
    '''
    Description:    
        Function to calculate area or detected aruco

    Args:
        coordinates (list): coordinates of detected aruco (4 set of (x,y) coordinates that is of edges)

    Returns:
        area (float) : area of detected aruco
        width (float): width of detected aruco
    '''

    ############ Function VARIABLES ############

    area = None
    width = None

    ############ Code ############

    # Calculate the Euclidean distances between coordinates
    width = np.linalg.norm(coordinates[0] - coordinates[1])
    height = np.linalg.norm(coordinates[1] - coordinates[2])
    
    area = width*height

    return area, width


def detect_aruco(image):
    '''
    Description:    
        Function to perform aruco detection and return each detail of aruco detected values
    such as marker ID, distance, angle, width, center point location, etc.

    Args:
        image (Image):  Input image frame received from respective camera topic

    Returns:
        center_aruco_list       (list): Center points of all aruco markers detected
        distance_from_rgb_list  (list): Distance value of each aruco markers detected from RGB camera
        angle_aruco_list        (list): Angle of all pose estimated for aruco marker
        width_aruco_list        (list): Width of all detected aruco markers
        ids                     (list): List of all aruco marker IDs detected in a single frame 
    '''

    ############ Function VARIABLES ############
    
    #Threshold area for the image to be considered further
    aruco_area_threshold = 1500

    # Camera Matrix from /camer_info topic in gazebo.
    cam_mat = np.array([[931.1829833984375, 0.0, 640.0], [0.0, 931.1829833984375, 360.0], [0.0, 0.0, 1.0]])

    # The distortion matrix
    dist_mat = np.array([[0.0,0.0,0.0,0.0,0.0]])

    # Aruco Marker size
    size_of_aruco_m = 0.15

 
    ############ CODE ############

    # Conversion of BGR image to grayscale for aruco detection
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2
    gray = image

    # Initialization of the lists to store marker information
    center_aruco_list = []
    distance_from_rgb_list = []
    angle_aruco_list = []
    width_aruco_list = []
    ids_list = []

    # Aruco parameters
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    aruco_parameters = cv2.aruco.DetectorParameters_create()

    # Aruco markers in the images filtered by the constraints
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_parameters)


    #Analysis for all the corners with 4 values
    if len(corners) > 0:

        #Individual ids of aruco bot
        ids = ids.flatten()     
        
        #Creating Markers for identification
        total_markers = range(0, ids.size)

        for (markerCorner, markerID, i) in zip(corners, ids, total_markers):
          
            corners = markerCorner.reshape((4, 2))

            #Destructuring of corener data
            (topLeft, topRight, bottomRight, bottomLeft) = corners
            area, width = calculate_rectangle_area(corners)
            
            #Distance analysis by area calculation
            if area > aruco_area_threshold: 
                topRight = (int(topRight[0]), int(topRight[1]))
                bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                topLeft = (int(topLeft[0]), int(topLeft[1]))
                corners = corners.reshape(1, 4, 2)
                rVec, tVec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, size_of_aruco_m, cam_mat, dist_mat)
                distance_from_rgb = tVec[0][0][2]

                # Calculation of corrected aruco angle
                angle_aruco = (0.788 * rVec[0][0][2]) - ((rVec[0][0][2] ** 2) / 3160)

                # Box indication
                cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
                cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
                cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
                cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)

                #Draw the pose of the marker:
                point = cv2.drawFrameAxes(gray, cam_mat, dist_mat, rVec, tVec, 0.1, 4)
                
                cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)

                # Append values to respective lists
                center_aruco_list.append((cX, cY)) 
                distance_from_rgb_list.append(distance_from_rgb)
                angle_aruco_list.append(angle_aruco)
                width_aruco_list.append(width)
                ids_list.append(markerID)
                
                cv2.putText(image, str(markerID),(topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)
                print("[Inference] ArUco marker ID: {}".format(markerID))

    return center_aruco_list, distance_from_rgb_list, angle_aruco_list, width_aruco_list, ids_list


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




##################### CLASS DEFINITION #######################

class aruco_tf(Node):
    '''
    ___CLASS___

    Description:    
        Class which servers purpose to define process for detecting aruco marker and publishing tf on pose estimated.
    '''

    def __init__(self):
        '''
        Description:    
            The __init__() function is called automatically every time the class is being used to create a new object.
        '''

        super().__init__('task2a1')     # registering node

        ############ Topic SUBSCRIPTIONS ############

        self.color_cam_sub = self.create_subscription(Image, '/camera/color/image_raw', self.colorimagecb, 10)
        self.depth_cam_sub = self.create_subscription(Image, '/camera/aligned_depth_to_color/image_raw', self.depthimagecb, 10)
        self.obj_publisher = self.create_publisher(List, '/obj_frames', 10)
        self.frame_names = List()

        ############ Constructor VARIABLES / OBJECTS ############

        image_processing_rate = 0.2                                                     # rate of time to process image (seconds)
        self.bridge = CvBridge()                                                        # initialise CvBridge object for image conversion
        self.tf_buffer = tf2_ros.buffer.Buffer()                                        # buffer time used for listening transforms
        self.listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.br = tf2_ros.TransformBroadcaster(self)                                    # object as transform broadcaster to send transform wrt some frame_id
        self.timer = self.create_timer(image_processing_rate, self.process_image)       # creating a timer based function which gets called on every 0.2 seconds (as defined by 'image_processing_rate' variable)
        
        self.cv_image = None                                                            # colour raw image variable (from colorimagecb())
        self.depth_image = None       
        self.cv_bridge = CvBridge()                                                     # depth image variable (from depthimagecb())


    def depthimagecb(self, data):
        '''
        Description:    
            Callback function for aligned depth camera topic and convert the data to CV2 image

        Args:
            data (Image): Input depth image frame received from aligned depth camera topic

        Returns:
            Showing of the depth image
        '''

        ############ CODE ############

        self.depth_image = self.cv_bridge.imgmsg_to_cv2(data, '16UC1')

        #cv2.imshow("Depth Image window", self.depth_image)
        #cv2.waitKey(3)


    def colorimagecb(self, data):
        '''
        Description:    
            Callback function for colour camera raw topic for convert the data to CV2 image

        Args:
            data (Image): Input coloured raw image frame received from image_raw camera topic

        Returns:
            Showing of the color image
        '''

        ############ CODE ############

        self.rgb_image = self.cv_bridge.imgmsg_to_cv2(data, 'bgr8')

        #cv2.imshow("Image window", self.rgb_image)
        #cv2.waitKey(3)


    def publish_tf(self, parent_frame, child_frame, translation, rotation):
        '''
        Description:    
            Function to publish the data in /tf topic

        Args:
            parent_frame: Axis Frame of the camera_link
            child_frame : Axis Frame of the Boxes having Aruco Markers in the vicinity
            translation : x, y, z coordinates of the aruco frame 
            rotation    : x, y, z, w axis orientation values of the aruco frame

        Returns:
            Publish the /tf message in ROS server
        '''

        ############ CODE ############
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = parent_frame
        t.child_frame_id = child_frame
        t.transform.translation.x = translation[0]
        t.transform.translation.y = translation[1]
        t.transform.translation.z = translation[2]
        t.transform.rotation.x = rotation[0]
        t.transform.rotation.y = rotation[1]
        t.transform.rotation.z = rotation[2]
        t.transform.rotation.w = rotation[3]

        self.br.sendTransform(t)


    def process_image(self):
        '''
        Description:    
            Funtion to get the aruco frame values from the image

        Args:
            object (aruco_tf): The object variables are used to analyze the image and publish it in the /tf topic

        Returns:
            Showing of the image with detected markers and center points
        '''

        ############ Function VARIABLES ############

        # Camera info topic variables - image pixel size, focalX, focalY, etc.        
        sizeCamX = 1280
        sizeCamY = 720
        centerCamX = 640 
        centerCamY = 360
        focalX = 931.1829833984375
        focalY = 931.1829833984375
            

        ############ CODE ############


        # Get aruco center, distance from rgb, angle, width, and ids list from 'detect_aruco' function
        center_aruco_list, distance_from_rgb_list, angle_aruco_list, width_aruco_list, ids = detect_aruco(self.rgb_image)

        #List of Object Frames
        self.obj_frames = []

        # Loop over detected aruco markers to calculate position and orientation transform and publish TF
        for i in range(len(ids)):
            marker_id = ids[i]
            translation = center_aruco_list[i]
            rotation = [0.0, 0.0, 0.0, 1.0]  # Assume no rotation initially

            # Correct the input aruco angle
            angle_aruco = angle_aruco_list[i]
            #angle_aruco = (0.788 * angle_aruco) - ((angle_aruco ** 2) / 3160)

            # Calculate quaternions from roll, pitch, yaw (where roll and pitch are 0 while yaw is corrected aruco_angle)
            rotation = quaternion_from_euler(np.pi/2, 0, -angle_aruco+np.pi/2)

            # Get realsense depth from self.depth_image (convert mm to m)
            depth = self.depth_image[translation[1], translation[0]] / 1000.0

            # Rectify x, y, z based on focal length, center value, and size of the image
            cX, cY = translation
            y = depth * (sizeCamX - cX - centerCamX) / focalX
            z = depth * (sizeCamY - cY - centerCamY) / focalY
            x = depth

            
            # Publish the transform between aruco marker and camera_link
            parent_frame = 'camera_link'
            child_frame = f'cam_{marker_id}'
            self.publish_tf(parent_frame, child_frame, [x, y, z], rotation)

            # Lookup transform between base_link and obj frame
            try:
                transform = self.tf_buffer.lookup_transform('base_link', child_frame, rclpy.time.Time())
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                self.get_logger().error(f"Failed to lookup transform from base_link to {child_frame}")
                continue

            # Publish the TF between object frame and base_link
            parent_frame = 'base_link'
            child_frame = f'obj_{marker_id}'
            translation = [transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]
            rotation = quaternion_from_euler(np.pi/2 , 0 , -angle_aruco+np.pi/2)
            self.publish_tf(parent_frame, child_frame, translation, rotation)
            self.obj_frames.append(f'obj_{marker_id}')

        self.frame_names.data = self.obj_frames
        self.obj_publisher.publish(self.frame_names)


        print(self.obj_frames)

        # Show the image with detected markers and center points located
        cv2.imshow("Aruco Markers Detection", self.rgb_image)
        key = cv2.waitKey(1) & 0xFF

        


##################### FUNCTION DEFINITION #######################

def main():
    '''
    Description:    Main function which creates a ROS node and spin around for the aruco_tf class to perform it's task
    '''

    # Initialisation
    rclpy.init(args=sys.argv)    

    # Creating ROS node
    node = rclpy.create_node('aruco_tf_process')  

    # Logging the Information
    node.get_logger().info('Node created: Aruco tf process') 
           
    # Creating a new object for class 'aruco_tf'
    aruco_tf_class = aruco_tf()                      

    # Spin on the object to make it alive in ROS 2 DDS
    rclpy.spin(aruco_tf_class)          

    # Clear node after spin ends
    aruco_tf_class.destroy_node()                                   

    # Shutdown process
    rclpy.shutdown()                                                


if __name__ == '__main__':
    '''
    Description:    If the python interpreter is running that module (the source file) as the main program, 
                    it sets the special __name__ variable to have a value “__main__”. 
                    If this file is being imported from another module, __name__ will be set to the module’s name.
    '''
    main()