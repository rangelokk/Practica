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
2) Попробовать типа анимировать модель
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
