Полезная ссылка для связи топиков; для дистанции; для координат:  
https://github.com/IntelRealSense/realsense-ros/issues/1342  
Камера ее название:
https://github.com/mgonzs13/ros2_asus_xtion

https://stackoverflow.com/questions/65774814/adding-new-points-to-point-cloud-in-real-time-open3d  
Старый DepthNode:
```
#Для облака точек
class Depth_Subscriber(Node):
    def __init__(self):
        super().__init__('depth_subscriber')
        Qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
            depth=5
        )
        self.subscription = self.create_subscription(
            PointCloud2,
            '/camera/depth_registered/points',
            self.depth_callback,
            qos_profile=Qos_profile)
        self.subscription  # prevent unused variable warning
    def depth_callback(self, msg):
        self.get_logger().info('Depth Image')
        ##xyz, rgb = pointcloud2_to_array(ros_cloud)
        pcd = pointcloud2_to_open3d(msg)
        '''
        Pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        '''
        o3d.visualization.draw_geometries([pcd])
```

