# Practica
Ведем дневник практики  
Переговорная:
```
segmented_pc = o3d.geometry.PointCloud()
        segmented_images = recognize(image_np)
        if(len(segmented_images)!=0):
            for im in segmented_images:
                mini_pc = pcd.crop(im)
                segmented_pc += mini_pc
```
```
TypeError: crop(): incompatible function arguments. The following argument types are supported:
    1. (self: open3d.cuda.pybind.geometry.PointCloud, bounding_box: open3d::geometry::AxisAlignedBoundingBox, invert: bool = False) -> open3d.cuda.pybind.geometry.PointCloud
    2. (self: open3d.cuda.pybind.geometry.PointCloud, bounding_box: open3d::geometry::OrientedBoundingBox, invert: bool = False) -> open3d.cuda.pybind.geometry.PointCloud

Invoked with: PointCloud with 137624 points., array([[[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        ...,
```

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
7) Аанг не сдался - и я не сдамся!
Для выполнения задачи сегментации изображения, которое появляется на экране в результате отработки функции, вам нужно будет загрузить и обработать это изображение, а затем сопоставить его с другими сегментированными изображениями.
```
import os
import cv2
import open3d as o3d

def segment_and_match(segmented_image_path):
    # Загрузка изображения для сегментации
    image = cv2.imread(segmented_image_path)

    # Сегментация изображения (пример)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Загрузка других сегментированных изображений для сопоставления (пример)
    segmented_images_folder = '/home/mlserver/dataset/Segmented'
    segmented_images = []
    for filename in os.listdir(segmented_images_folder):
        if filename.endswith('.png'):
            segmented_image = cv2.imread(os.path.join(segmented_images_folder, filename))
            segmented_images.append(segmented_image)

    # Сопоставление сегментированного изображения с другими изображениями (пример)
    for seg_img in segmented_images:
        # Действия по сопоставлению изображений
        pass

    # Отображение результатов в окне Open3D
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # Добавление вашего изображения и других сегментированных изображений в окно Open3D
    vis.add_geometry(o3d.geometry.Image(image))
    for seg_img in segmented_images:
        vis.add_geometry(o3d.geometry.Image(seg_img))

    vis.run()

# Пример использования функции для сегментации и сопоставления изображения
segmented_image_path = '/path/to/your/image.png'
segment_and_match(segmented_image_path)
```
8) Зуко не сдался - и я не сдамся!
Для сегментации изображений, полученных с RGBD-камеры с помощью библиотек open3d, point_cloud2 и rclpy, вы можете использовать следующий подход:

1. Преобразуйте изображения RGBD в RGB изображение.
2. Примените алгоритм сегментации, например, на основе цвета или глубины, для каждого RGB изображения.
3. Сопоставьте сегментированные изображения с уже имеющимися сегментированными изображениями.

```
import cv2
import open3d as o3d

def segment_image(rgb_image):
    # Преобразование RGB изображения в формат OpenCV
    rgb_image_cv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    # Применение алгоритма сегментации (например, на основе цвета)
    gray_image = cv2.cvtColor(rgb_image_cv, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary_image

# Пример использования функции для сегментации изображения
# Предположим, у вас есть RGB изображение image_rgb
# image_rgb = полученное изображение с RGBD-камеры

# Сегментация изображения
segmented_image = segment_image(image_rgb)

# Сопоставление сегментированного изображения с другими изображениями
# Добавьте необходимую логику для сопоставления

# Отображение сегментированного изображения
cv2.imshow('Segmented Image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
#Пожалуйста, убедитесь, что данные, полученные из RGBD-камеры, правильно обрабатываются и преобразуются в RGB формат перед применением алгоритма сегментации. Кроме того, добавьте логику для сопоставления с уже #имеющимися сегментированными изображениями.
```
Для сопоставления сегментированных изображений с уже имеющимися сегментированными изображениями можно использовать различные методы, в зависимости от ваших конкретных требований. Один из подходов — использование алгоритма сопоставления особых точек (например, SIFT, SURF, ORB) для нахождения соответствий между изображениями. 

```
import cv2

def match_segmented_images(segmented_image1, segmented_image2):
    # Инициализация детектора особых точек (например, ORB)
    detector = cv2.ORB_create()

    # Обнаружение особых точек и вычисление их дескрипторов для каждого изображения
    keypoints1, descriptors1 = detector.detectAndCompute(segmented_image1, None)
    keypoints2, descriptors2 = detector.detectAndCompute(segmented_image2, None)

    # Инициализация объекта для поиска соответствий между дескрипторами
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Поиск соответствий между дескрипторами
    matches = matcher.match(descriptors1, descriptors2)

    # Отображение сопоставленных особых точек на изображениях
    matched_image = cv2.drawMatches(segmented_image1, keypoints1, segmented_image2, keypoints2, matches, None)

    return matched_image

# Пример использования функции для сопоставления двух сегментированных изображений
# Предположим, у вас есть два сегментированных изображения segmented_image1 и segmented_image2

# Сопоставление сегментированных изображений
matched_result = match_segmented_images(segmented_image1, segmented_image2)

# Отображение результата сопоставления
cv2.imshow('Matched Result', matched_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
Этот код реализует логику сопоставления и позволяет найти соответствия между особыми точками на двух изображениях и отобразить результат сопоставления. Убедитесь, что ваши сегментированные изображения содержат достаточное количество текстурных особых точек для успешного сопоставления.
