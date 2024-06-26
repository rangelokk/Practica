Полезная ссылка для связи топиков; для дистанции; для координат:  
https://github.com/IntelRealSense/realsense-ros/issues/1342  
Камера ее название:
https://github.com/mgonzs13/ros2_asus_xtion

from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy, QoSReliabilityPolicy
from sensor_msgs.msg import Image, PointCloud2
from message_filters import Subscriber
from message_filters import TimeSynchronizer

class Multiply_Subscriber(Node):
    def __init__(self):
        qos_profile_i = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_RELIABLE,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
            depth=5
        )
        self.image = Subscriber(
            '/camera/rgb/image_raw',
            Image,
            qos_profile=qos_profile_i,
            queue_size=10)
        
        qos_profile_d = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
            depth=5
        )
        self.depth = Subscriber(
            '/camera/depth_registered/points',
            PointCloud2,
            qos_profile=qos_profile_d,
            queue_size=10)
        
        ts = TimeSynchronizer([self.image, self.depth], 10)
        ts.registerCallback(self.Callback)

    def Callback(self, img_msg, dep_msg):
        self.get_logger().info('Multi msg')
