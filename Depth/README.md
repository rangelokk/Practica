Полезная ссылка для связи топиков; для дистанции; для координат:  
https://github.com/IntelRealSense/realsense-ros/issues/1342  
Камера ее название:
https://github.com/mgonzs13/ros2_asus_xtion

https://stackoverflow.com/questions/65774814/adding-new-points-to-point-cloud-in-real-time-open3d
```
self.vis = o3d.visualization.Visualizer()
self.vis.create_window(height=480, width=640)
self.pcd = o3d.geometry.PointCloud()
self.vis.add_geometry(self.pcd)

def multi_callback(self, img_msg, dep_msg):
        self.get_logger().info('Multi msg')
  
        bridge = CvBridge()
        image_np = bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        
        recognize(image_np)

        self.pcd = pointcloud2_to_open3d(dep_msg)
        #self.pcd.points.extend(pointcloud2_to_open3d(dep_msg))
        self.vis.update_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()
```
