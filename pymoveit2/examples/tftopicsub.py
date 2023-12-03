#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
import time

class TfSubscriber(Node):
    def __init__(self):
        super().__init__('tf_subscriber')
        self.tf_data = None
        self.subscription = self.create_subscription(
            TFMessage,
            '/tf',  # Topic name for tf data
            self.listener_callback,  # Callback function to handle received messages
            10  # QoS profile, 10 is the queue size
        )

    def listener_callback(self, msg):
        for transform in msg.transforms:
            parent_frame = transform.header.frame_id
            child_frame = transform.child_frame_id
            translation = transform.transform.translation
            rotation = transform.transform.rotation
            self.tf_data = {
                    "parent_frame": parent_frame,
                    "child_frame": child_frame,
                    "translation": [translation.x, translation.y, translation.z],
                    "rotation": [rotation.x, rotation.y, rotation.z, rotation.w]
                }
            self.get_logger().info(f"Received transform: Parent Frame: {parent_frame}, Child Frame: {child_frame}, Translation: {self.tf_data['translation']}, Rotation: {self.tf_data['rotation']}")
            time.sleep(3)

def main(args=None):
    rclpy.init(args=args)
    node = TfSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

