class Multiply_Subscriber(Node):
    def __init__(self):
        Qos_profile_i = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_RELIABLE,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
            depth=5
        )
        self.image = message_filters.Subscriber(
            '/camera/rgb/image_raw',
            Image,
            self.Callback,
            qos_profile=Qos_profile_i)
        self.image
        Qos_profile_d = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
            depth=5
        )
        self.depth = message_filters.Subscriber(
            PointCloud2,
            '/camera/depth_registered/points',
            self.Callback,
            qos_profile=Qos_profile_d)
        self.depth
    def Callback(self, img, dep):
        self.get_logger().info('Multi msg')
