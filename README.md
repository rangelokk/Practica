# Practica
Ведем дневник практики  
Переговорная:
https://github.com/MarSpit/LiDAR_camera_calibration_tool  

ССЫЛКИ:  
- https://www.open3d.org/html/tutorial/Basic/pointcloud.html  
(может, binary_image - это тот самый файл .json для crop из этой документации?)  
- https://github.com/noshluk2/Point-Cloud-Segmentaion-PCL-and-Open3D/tree/main/Python_Open3D  
Код для сегментации, в целом совпадает с тем, что я пыталась написать сама.  
- https://github.com/rangelokk/Practica/blob/main/Segmentation_Pointcloud2/code.py
Попробовала написать bounding box, хотя бы для вдохновения к первой ссылке документации
- https://github.com/isl-org/Open3D/blob/main/examples/python/geometry/point_cloud_crop.py  
Crop Geometry  

1) Новая попытка кода
```
import cv2

def segment_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 

    return binary_image

# Принимаем облако точек и проводим сегментацию изображения
def segment_pointcloud(data):
    point_cloud = pointcloud2_to_open3d(data)
    # Преобразование point_cloud в RGB изображение
    image = np.asarray(point_cloud.colors).reshape(-1, 1, 3)
    image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR)

    # Вызываем функцию для сегментации изображения
    segmented_image = segment_image(image) # надо устанавливать: pip install segment-image

    return segmented_image

data = # облако точек
segmented_image = segment_pointcloud(data)
```

2) Буду делать больше попыток!!!
```
import rclpy
import open3d as o3d
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs_py import point_cloud2
import cv2

# Функция для сегментации изображения и вывода результата в окне Open3D
def segment_and_visualize_pointcloud(data):
    point_cloud = pointcloud2_to_open3d(data)
    # Преобразование point_cloud в RGB изображение
    image = np.asarray(point_cloud.colors).reshape(-1, 1, 3)
    image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR)

    # Сегментация изображения с использованием OpenCV
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Создание окна Open3D для отображения результата
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(point_cloud)
    vis.run()
    # Отображение сегментированного изображения в окне Open3D
    texture = o3d.geometry.Image(binary_image)
    o3d.visualization.draw_geometries_with_image([point_cloud], [texture])

# Пример использования функции для сегментации изображения из облака точек и отображения результата в окне Open3D
data = # ваше облако точек
segment_and_visualize_pointcloud(data)

```

3) Сегментация облака точек с использованием библиотеки Open3D
```
import rclpy
import open3d as o3d
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs_py import point_cloud2
import cv2

def segment_point_cloud(point_cloud):
    points = np.array(point_cloud.data)                     # Преобразование облака точек в формат numpy
    pcd = o3d.geometry.PointCloud()                         # Создание объекта для представления облака точек в Open3D
    pcd.points = o3d.utility.Vector3dVector(points)         # Сегментация облака точек с использованием метода кластеризации
    labels = np.array(pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=True)) # Присвоение случайных цветов каждому кластеру для визуализации
    max_label = labels.max()
    colors = plt.cm.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))     # Применение цветов к точкам
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])                              # Отображение результата сегментации во всплывающем окне Open3D
    o3d.visualization.draw_geometries([pcd])

segment_point_cloud(point_cloud)
```

4) Сегментация
```
import open3d as o3d
import cv2
import numpy as np

# Загрузка облака точек point_cloud2
point_cloud2 = o3d.io.read_point_cloud("point_cloud2.ply")

# Преобразование изображения RGBD в изображение RGB с использованием Open3D
rgbd_image = o3d.geometry.RGBDImage.create_from_point_cloud_pcl(point_cloud2, camera_intrinsic)
rgb_image = rgbd_image.color

# Преобразование изображения RGB в формат OpenCV для дальнейшей обработки
rgb_image_cv2 = np.asarray(rgb_image)
rgb_image_cv2 = cv2.cvtColor(rgb_image_cv2, cv2.COLOR_RGB2BGR)

cv2.imshow("Segmented Image", segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
