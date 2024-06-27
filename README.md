# Practica
Ведем дневник практики  
Переговорная:
```
import open3d as o3d
import numpy as np
from sensor_msgs import point_cloud2

def pointcloud2_to_open3d(data, colors):
    points = []
    for p, color in zip(point_cloud2.read_points(data, field_names=("x", "y", "z", "rgb"), skip_nans=True), colors):
        points.append([p[0], p[1], p[2]])
    
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    
    # Преобразование цветов в формат RGB
    colors_rgb = [(int(color) & 0xff, (int(color) >> 8) & 0xff, (int(color) >> 16) & 0xff) for color in colors]
    point_cloud.colors = o3d.utility.Vector3dVector(np.asarray(colors_rgb) / 255.0)
    
    return point_cloud
```
