Полезная ссылка для связи топиков; для дистанции; для координат:  
https://github.com/IntelRealSense/realsense-ros/issues/1342  
Камера ее название:
https://github.com/mgonzs13/ros2_asus_xtion



class message_filters.Subscriber(*args, **kwargs)

    ROS subscription filter. Identical arguments as rospy.Subscriber.

    This class acts as a highest-level filter, simply passing messages from a ROS subscription through to the filters which have connected to it.

    registerCallback(cb, *args)

        Register a callback function cb to be called when this filter has output. The filter calls the function cb with a filter-dependent list of arguments, followed by the call-supplied arguments args.



```
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy, QoSReliabilityPolicy
from sensor_msgs.msg import Image, PointCloud2
from message_filters import Subscriber, ExactTimeSynchronizer

class Multiply_Subscriber(Node):
    def __init__(self):
        super().__init__('multiply_subscriber')

        qos_profile_i = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_RELIABLE,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
            depth=5
        )
        self.image_sub = Subscriber(self, '/camera/rgb/image_raw', Image, qos_profile=qos_profile_i)

        qos_profile_d = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
            depth=5
        )
        self.depth_sub = Subscriber(self, '/camera/depth_registered/points', PointCloud2, qos_profile=qos_profile_d)

        self.ts = ExactTimeSynchronizer([self.image_sub, self.depth_sub], 10)
        self.ts.registerCallback(self.Callback)

    def Callback(self, img_msg, dep_msg):
        self.get_logger().info('Multi msg')

    def destroy_node(self):
        self.image_sub.destroy()
        self.depth_sub.destroy()
        super().destroy_node()
```
