# Practica
Ведем дневник практики  
Переговорная:
```
segmented_pc = o3d.geometry.PointCloud()
        segmented_images = recognize(image_np)
        if(len(segmented_images)!=0):
            for im in segmented_images:
                #mini_pc = pcd.crop(im)
                #segmented_pc += mini_pc
                indices_to_select = get_indice_from_segmented_image(im)
                if(len(indices_to_select)!=0):
                    selected_points = pcd.select_by_index(indices_to_select)
                    segmented_pc += selected_points
                    print("a")

        self.vis.clear_geometries()
        
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        if(len(segmented_images)==0 or len(segmented_pc.points) == 0):
            self.vis.add_geometry(pcd)
        else:
            self.vis.add_geometry(segmented_pc)

        #print("Field of view (after changing) %.2f" % self.vis.get_view_control().get_rotate())
        self.vis.reset_view_point()
        self.vis.poll_events()
        self.vis.update_renderer()
```
```
corrupted double-linked list (not small)
Aborted (core dumped)
```

1) Новая попытка кода
```
import cv2

def segment_image(image):
    # Пример сегментации изображения с использованием OpenCV
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) #нижнее подчеркивание - это такое название переменной, всё норм!

    return binary_image

# Принимаем облако точек и проводим сегментацию изображения
def segment_pointcloud(data):
    point_cloud = pointcloud2_to_open3d(data)
    # **Преобразование point_cloud в изображение (предположим, что требуется RGB изображение)**
    image = np.asarray(point_cloud.colors).reshape(-1, 1, 3)
    image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR)

    #Вызываем функцию для сегментации изображения
    segmented_image = segment_image(image)

    return segmented_image

# Пример использования функции для сегментации изображения из облака точек
data = # ваше облако точек
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
    # **Преобразование point_cloud в изображение (предположим, что требуется RGB изображение)**
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

3) Для сегментации изображений, полученных с RGBD-камеры с помощью библиотек open3d, point_cloud2 и rclpy, вы можете использовать следующий подход:

1. Преобразуйте изображения RGBD в RGB изображение.
2. Примените алгоритм сегментации, например, на основе цвета или глубины, для каждого RGB изображения.
3. Сопоставьте сегментированные изображения с уже имеющимися сегментированными изображениями.

```
import cv2
import open3d as o3d

def segment_image(rgb_image):
    # **Преобразование RGB изображения в формат OpenCV**
    rgb_image_cv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    # Применение алгоритма сегментации (например, на основе цвета)
    gray_image = cv2.cvtColor(rgb_image_cv, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_image

segmented_image = segment_image(image_rgb)

cv2.imshow('Segmented Image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
Для сопоставления сегментированных изображений с уже имеющимися сегментированными изображениями можно использовать различные методы, в зависимости от ваших конкретных требований. Один из подходов — использование алгоритма сопоставления особых точек (например, SIFT, SURF, ORB) для нахождения соответствий между изображениями. 

```
import cv2

def match_segmented_images(segmented_image1, segmented_image2):
    detector = cv2.ORB_create()
    keypoints1, descriptors1 = detector.detectAndCompute(segmented_image1, None)
    keypoints2, descriptors2 = detector.detectAndCompute(segmented_image2, None)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2)
    matched_image = cv2.drawMatches(segmented_image1, keypoints1, segmented_image2, keypoints2, matches, None)
    return matched_image
matched_result = match_segmented_images(segmented_image1, segmented_image2)
cv2.imshow('Matched Result', matched_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

4) Сегментация облака точек с использованием библиотеки Open3D
```
import rclpy
import open3d as o3d
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs_py import point_cloud2
import cv2

def segment_point_cloud(point_cloud):
    points = np.array(point_cloud.data) # Преобразование облака точек в формат numpy
    pcd = o3d.geometry.PointCloud() # Создание объекта для представления облака точек в Open3D
    pcd.points = o3d.utility.Vector3dVector(points) # Сегментация облака точек с использованием метода кластеризации
    labels = np.array(pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=True)) # Присвоение случайных цветов каждому кластеру для визуализации
    max_label = labels.max()
    colors = plt.cm.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1)) # Применение цветов к точкам
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3]) # Отображение результата сегментации во всплывающем окне Open3D
    o3d.visualization.draw_geometries([pcd])

# Пример использования функции для сегментации облака точек
# Предположим, у вас есть переменная point_cloud, содержащая данные облака точек
# Сегментация облака точек
segment_point_cloud(point_cloud)
```



