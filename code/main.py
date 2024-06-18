import rclpy
from rclpy.node import Node
import cv2

#from std_msgs.msg import String
from sensor_msgs.msg import Image
import numpy as np
from cv_bridge import CvBridge

class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            Image,
            #String,
            '/camera/rgb/image_raw',
            #'topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        #выводит массив пикселей
        #self.get_logger().info('I heard: "%s"' % msg.data)
        
        #получаем из массива пикселей картинку
        bridge = CvBridge()
        image_np = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        cv2.imshow("img", image_np)
        cv2.waitKey(1)

    

def main(args=None):
    
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
