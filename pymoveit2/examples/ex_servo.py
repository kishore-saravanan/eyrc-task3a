#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
import time
from time import sleep

class MyPublisher(Node):

    def __init__(self):
        super().__init__('my_publisher')
        self.publisher_ = self.create_publisher(TwistStamped, '/servo_node/delta_twist_cmds', 10)
        timer_period = 0.008  # 125 Hz
        #self.timer = self.create_timer(timer_period, self.publish_twist)
        self.twist_msg = TwistStamped()
        self.publish_twist()

    def publish_twist(self):
        end_time  = time.time() + 5
        while rclpy.ok() and time.time() < end_time:
            self.twist_msg.header.stamp = self.get_clock().now().to_msg()
            self.twist_msg.twist.linear.x = 0.0
            self.twist_msg.twist.linear.y = -0.2
            self.twist_msg.twist.linear.z = -0.2
            self.twist_msg.twist.angular.x = 0.0
            self.twist_msg.twist.angular.y = 0.0
            self.twist_msg.twist.angular.z = 0.0
            self.publisher_.publish(self.twist_msg)
            print("moving")
        print("stopped")
        '''sleep(5)
        self.twist_msg.twist.linear.z = -0.2
        self.publisher_.publish(self.twist_msg)
        print("down")
        sleep(5)
        self.twist_msg.twist.linear.z = 0.0
        self.publisher_.publish(self.twist_msg)
        print("home")
        sleep(5)'''

def main(args=None):
    rclpy.init(args=args)
    node = MyPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
