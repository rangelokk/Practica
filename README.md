# Practica
Ведем дневник практики  
Переговорная:
```
def pointcloud2_to_open3d(data):
    field_names=[field.name for field in data.fields]
    cloud_data = list(point_cloud2.read_points(data, skip_nans=True, field_names = field_names))
    xyz = [(x,y,z) for x,y,z,rgb in cloud_data ]
    
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(np.array(xyz))
    if "rgb" in field_names:
        rgb = [(int(rgb) & 0xff, (int(rgb) >> 8) & 0xff, (int(rgb) >> 16) & 0xff) for x,y,z,rgb in cloud_data ]
    point_cloud.colors = o3d.utility.Vector3dVector(np.asarray(rgb)/255.0)
    
    return point_cloud
```
