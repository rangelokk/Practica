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
4) Еще одна проба сегментации
```
import os
import cv2

def segment_and_save_images(input_folder, output_folder):
    # Создание выходной папки, если она не существует
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            image_path = os.path.join(input_folder, filename)
            segmented_image = cv2.imread(image_path)

            # Ваш код для сегментации изображения
            # Например, используйте алгоритмы компьютерного зрения или нейронные сети для сегментации

            # Сохранение сегментированного изображения в выходную папку
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, segmented_image)

    print("Сегментация и сохранение завершены.")

# Путь к папке с исходными изображениями
input_folder = "/home/mlserver/dataset/Segmented"
# Путь к папке для сохранения сегментированных изображений
output_folder = "/path/to/output/folder"

# Вызов функции для сегментации и сохранения изображений
segment_and_save_images(input_folder, output_folder)

#Прежде чем использовать этот код, убедитесь, что у вас есть необходимые библиотеки (например, OpenCV)
#и что вы дополните функцию segment_and_save_images соответствующим кодом для сегментации изображений.

```
5) Новая попытка кода
```
import cv2

def segment_image(image):
    # Пример сегментации изображения с использованием OpenCV
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#нижнее подчеркивание - это такое название переменной, всё норм!

    return binary_image

# Принимаем облако точек и проводим сегментацию изображения
def segment_pointcloud(data):
    point_cloud = pointcloud2_to_open3d(data)
    
    # Преобразование point_cloud в изображение (предположим, что требуется RGB изображение)
    image = np.asarray(point_cloud.colors).reshape(-1, 1, 3)
    image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR)

    # Вызываем функцию для сегментации изображения
    segmented_image = segment_image(image)

    return segmented_image

# Пример использования функции для сегментации изображения из облака точек
data = # ваше облако точек
segmented_image = segment_pointcloud(data)
```
6) Буду делать больше попыток!!!
```
import rclpy
import open3d as o3d
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs_py import point_cloud2
import cv2

# ... (ваш предоставленный код)

# Функция для сегментации изображения и вывода результата в окне Open3D
def segment_and_visualize_pointcloud(data):
    point_cloud = pointcloud2_to_open3d(data)
    
    # Преобразование point_cloud в изображение (предположим, что требуется RGB изображение)
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
#Пожалуйста, убедитесь, что вы передаете правильные данные в функцию segment_and_visualize_pointcloud, и что данные соответствуют ожидаемому формату для корректной работы функции.
```
