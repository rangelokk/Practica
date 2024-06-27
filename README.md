# Practica
Ведем дневник практики  
Переговорная:
```
def pointcloud2_to_open3d(data):
        points = []
        for p in point_cloud2.read_points(data, field_names=("x", "y", "z"), skip_nans=True):
            points.append([p[0], p[1], p[2]])
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        return point_cloud
```
