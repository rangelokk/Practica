# Practica
Ведем дневник практики  
Переговорная:

1) Попробовать еще раз код на вращение в Open3D:  
```
import numpy as np
import open3d as o3d

# Ваша функция pointcloud2_to_open3d(data) здесь

# Создаем окно для отображения облака точек
vis = o3d.visualization.Visualizer()
vis.create_window()

# Добавляем облако точек в окно
point_cloud = pointcloud2_to_open3d(data)
vis.add_geometry(point_cloud)

# Получаем указатель на окно
render_option = vis.get_render_option()
view_control = vis.get_view_control()

# Включаем режим вращения мышью
view_control.rotate(10.0, 0.0)

# Запускаем визуализацию и ожидаем закрытия окна
while True:
    vis.update_geometry(point_cloud)
    vis.poll_events()
    vis.update_renderer()

vis.destroy_window()
```
2) Попробовать типа анимировать модель:  
```
vis = o3d.visualization.Visualizer()
vis.create_window()

# геометрия — это облако точек, используемое в вашей анимации
geometry = o3d.geometry.PointCloud()
vis.add_geometry(geometry)

for i in range(icp_iteration):
    # теперь измените точки вашей геометрии
    # вы можете использовать любой метод, который вам больше подходит, это всего лишь пример
    geometry.points = pcd_list[i].points
    vis.update_geometry(geometry)
    vis.poll_events()
    vis.update_renderer()
```
3) Попытка в сегментацию, первый блин комом :(  
```
import os
import cv2
import open3d as o3d
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs_py import point_cloud2

def segment_and_visualize_images_from_pointcloud(pointcloud_data, segmented_images_folder):
    field_names = [field.name for field in pointcloud_data.fields]
    cloud_data = list(point_cloud2.read_points(pointcloud_data, skip_nans=True, field_names=field_names))
    xyz = [(x, y, z) for x, y, z, rgb in cloud_data]

    # Сегментация изображений из папки
    segmented_images = []
    for filename in os.listdir(segmented_images_folder):
        if filename.endswith(".png"):
            image_path = os.path.join(segmented_images_folder, filename)
            segmented_image = cv2.imread(image_path)
            segmented_images.append(segmented_image)

    # Процесс сегментации изображений и облака точек
    # Ваш код для сегментации изображений и облака точек

    # Визуализация сегментированных изображений и облака точек
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # Добавление облака точек в окно
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(np.array(xyz))
    vis.add_geometry(point_cloud)

    # Добавление сегментированных изображений в окно
    for segmented_image in segmented_images:
        # Ваш код для добавления сегментированных изображений в окно визуализации

    # Отображение окна визуализации
    vis.run()
    vis.destroy_window()
```
