Полезная ссылка для связи топиков; для дистанции; для координат:  
https://github.com/IntelRealSense/realsense-ros/issues/1342  
Камера ее название:
https://github.com/mgonzs13/ros2_asus_xtion


1) Семантическая сегментация, также гайд по установке PyTorch, если совсем никак без него (Гитхаб)
https://github.com/isl-org/Open3D-ML
2) Семантическая сегментация с PyTorch (с обучением модели)
https://www.open3d.org/docs/latest/python_api/open3d.ml.torch.pipelines.SemanticSegmentation.html
3) Семантическая сегментация (с обучением модели)
https://open-model-zoo.readthedocs.io/en/latest/demos/3d_segmentation_demo/python/README.html
4) Сегментация плоскости облака точек Open3D (Без PyTorch)
https://programmersought.com/article/85928403768/
5) Визуализация облака точек
https://learngeodata.eu/visualise-massive-point-cloud-in-python/
6) Сегментация у гика
https://www.geeksforgeeks.org/image-segmentation-with-watershed-algorithm-opencv-python/
7) 4 метода сегментации в Opencv 
https://machinelearningknowledge.ai/image-segmentation-in-python-opencv/
8) Облако точек
https://sigmoidal.ai/en/point-cloud-processing-with-open3d-and-python/
9) Сегментация (Гитхаб)
https://github.com/isl-org/Open3D-PointNet2-Semantic3D
10) Какие-то репозитории для сегментации 
https://encord.com/blog/github-repositories-image-segmentation/
11) Сегментация
https://www.topbots.com/automate-3d-point-cloud-segmentation/
12) Гитхаб
https://github.com/felixchenfy/open3d_ros_pointcloud_conversion
13) Какие-то хорошие уроки
https://stackforgeeks.com/blog/running-a-python-node-in-ros2-humble
14) Что-то про ros2
https://roboticsbackend.com/write-minimal-ros2-python-node/
15) Гитхаб и PyTorch Crop Geometry
https://github.com/yu-frank/PerspectiveCropLayers
16) Гитхаб Crop Geometry
https://github.com/yu4u/imgcrop
17) Гитхаб Crop Geometry
https://github.com/isl-org/Open3D/issues/3186
18) Гитхаб Crop Geometry
https://github.com/isl-org/Open3D/issues/4858
19) https://github.com/isl-org/Open3D/issues/1138
20) Что-то про Open3D и points
https://yiyangd.github.io/o3d_notes01/
21) туториалы по Open3d
https://www.programmerall.com/article/26062068847/
22) Примеры кода с геометрией
https://www.programcreek.com/python/example/110516/open3d.draw_geometries
23) Что-то про Open3d
https://analyticsindiamag.com/guide-to-open3d-an-open-source-modern-library-for-3d-data-processing/
24) о, хабр
https://habr.com/ru/companies/skillfactory/articles/693338/


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

