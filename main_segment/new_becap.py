
import rclpy
from rclpy.node import Node
import cv2

import message_filters
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
import sensor_msgs.msg as sensor_msgs
import std_msgs.msg as std_msgs
from message_filters import Subscriber
from message_filters import TimeSynchronizer, ApproximateTimeSynchronizer
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
import numpy as np
from cv_bridge import CvBridge
import open3d as o3d
from ctypes import *

count=0




def recognize(image_np):
        global count
        width = 640
        height = 480
        net = cv2.dnn.readNet("/home/mlserver/darknet/backup/yolo-obj_best.weights", "/home/mlserver/darknet/yolo-obj.cfg")
        classes = []
        with open("/home/mlserver/darknet/build/darknet/x64/data/obj.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        #преобразование в формат с которым может работать yolo v3
        blob = cv2.dnn.blobFromImage(image_np, 0.00392, (416,416),(0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []
        segmented_images = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                #Если вероятность того, что это именно этот объект выше чем 50% то выводим
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x,y,w,h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0, 255, size=(len(classes), 3))
        for i in range(len(boxes)):
            if i in indexes:
                x,y,w,h = boxes[i]

                print("Прямоугольник детектирования: " + str(x)+ '-' + str(x+w) + ' ' + str(y) + '-' + str(y+h))

                image = image_np.copy()

                mask = np.zeros(image_np.shape[:2], np.uint8)
                bgdModel = np.zeros((1, 65), np.float64)
                fgbModel = np.zeros((1, 65), np.float64)
                rect = boxes[i]
                cv2.grabCut(image, mask, rect, bgdModel, fgbModel, 5, cv2.GC_INIT_WITH_RECT)
                mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
                image = image * mask2[:, :, np.newaxis]
                segmented_images.append(image)
                '''
                cv2.imwrite('/home/mlserver/dataset/Segmented/s'+str(count)+'.png', image) #для сохранения сегментированных изобр в реал.времени
                count += 1
                '''
                label = str(classes[class_ids[i]]) #+ " x: " + str(x+w/2) + "y: " + str(y+h/2)
                color = (255, 255, 0)#colors[i]
                cv2.rectangle(image_np, (x,y), (x + w, y + h), color, 2)
                cv2.putText(image_np, label, (x, y + 30), font, 3, color, 3)
        cv2.imshow("img", image_np)
        cv2.waitKey(1)
        return segmented_images


convert_rgbUint32_to_tuple = lambda rgb_uint32: (
    (rgb_uint32 & 0x00ff0000)>>16, (rgb_uint32 & 0x0000ff00)>>8, (rgb_uint32 & 0x000000ff)
)
convert_rgbFloat_to_tuple = lambda rgb_float: convert_rgbUint32_to_tuple(
    int(cast(pointer(c_float(rgb_float)), POINTER(c_uint32)).contents.value)
)
def pointcloud2_to_open3d(data):
    field_names=[field.name for field in data.fields]
    cloud_data = list(point_cloud2.read_points(data, skip_nans=True, field_names = field_names))
    xyz = [(x,y,z) for x,y,z,rgb in cloud_data ]
    
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(np.array(xyz))
    if "rgb" in field_names:
        rgb = [convert_rgbFloat_to_tuple(rgb) for x,y,z,rgb in cloud_data ]
    point_cloud.colors = o3d.utility.Vector3dVector(np.asarray(rgb)/255.0)
    
    return point_cloud


def Cropinng(im, pc):
    inc_point = []
    non_zero_indices = np.argwhere(im != 0)
    print("ДЕТЕКТИРОВАЛ И НАРЕЗАЛ " + str(len(non_zero_indices)))
    filtered_points = []

    f, y = 585, 60

    min_i, max_i, min_j, max_j = 600, 0, 600, 0
    for i in range(480):
        for j in range(640):
            if np.any(im[i,j]) != 0:
                min_i, max_i, min_j, max_j = min(i, min_i), max(i, max_i), min(j, min_j), max(j, max_j)
    print("Диапазон сегментации i: " + str(min_i) + "-" + str(max_i)+ " j: " + str(min_j) + "-" + str(max_j))
    min_i, max_i, min_j, max_j = 600, -600, 600, -600
    induces_to_select = []
    #for point in np.asarray(pc.points):
    for i in range(0, len(pc.points)):
        u = round((f * np.asarray(pc.points)[i][0] + y* np.asarray(pc.points)[i][1])/ np.asarray(pc.points)[i][2]) + 320
        v = round((f * np.asarray(pc.points)[i][1])/ np.asarray(pc.points)[i][2]) + 240

        min_i, max_i, min_j, max_j = min(u, min_i), max(u, max_i), min(v, min_j), max(v, max_j)
        #print(point[0], point[1], point[2])
        if (u>0 and u<480) and (v>0 and v<640):
            if np.any(im[u,v]) != 0:
                induces_to_select.append(i)
                
    print("Диапазон облака точек u: " + str(min_i) + "-" + str(max_i)+ " v: " + str(min_j) + "-" + str(max_j))
    print("Колличество индексов: " + str(len(induces_to_select)))
    # Создаем новое облако точек Open3D
    filtered_pc = o3d.geometry.PointCloud()
    filtered_pc = pc.select_by_index(induces_to_select)
    #filtered_pc.points = o3d.utility.Vector3dVector(filtered_points)
    

    return filtered_pc

def point_cloud(points, parent_frame):
    # In a PointCloud2 message, the point cloud is stored as an byte 
    # array. In order to unpack it, we also include some parameters 
    # which desribes the size of each individual point.
    ros_dtype = sensor_msgs.PointField.FLOAT32
    dtype = np.float32
    itemsize = np.dtype(dtype).itemsize # A 32-bit float takes 4 bytes.

    data = points.astype(dtype).tobytes() 

    # The fields specify what the bytes represents. The first 4 bytes 
    # represents the x-coordinate, the next 4 the y-coordinate, etc.
    fields = [sensor_msgs.PointField(
        name=n, offset=i*itemsize, datatype=ros_dtype, count=1)
        for i, n in enumerate('xyz')]

    # The PointCloud2 message also has a header which specifies which 
    # coordinate frame it is represented in. 
    header = std_msgs.Header(frame_id=parent_frame)

    return sensor_msgs.PointCloud2(
        header=header,
        height=1, 
        width=points.shape[0],
        is_dense=False,
        is_bigendian=False,
        fields=fields,
        point_step=(itemsize * 3), # Every point consists of three float32s.
        row_step=(itemsize * 3 * points.shape[0]),
        data=data
    )

#Синхронизация двух топиков
class Multiply_Subscriber(Node):
    
    def __init__(self):
        super().__init__('multiply_subscriber')
        self.bridge = CvBridge()
        qos_profile_i = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_RELIABLE,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
            depth=5
        )
        self.image_sub = Subscriber(self, Image, '/camera/rgb/image_raw', qos_profile=qos_profile_i)

        qos_profile_d = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
            depth=5
        )
        self.depth_sub = Subscriber(self, PointCloud2, '/camera/depth_registered/points',  qos_profile=qos_profile_d)
        #self.cam_sub = Subscriber(self, CameraInfo, '/camera/camera_info')

        self.ts = ApproximateTimeSynchronizer([self.image_sub, self.depth_sub], 10, 0.1)
        self.ts.registerCallback(self.multi_callback)
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(height=480, width=640)
        self.vis.add_geometry(o3d.geometry.PointCloud())

        self.pcd_publisher = self.create_publisher(sensor_msgs.PointCloud2, 'pcd', 10)
    


    def multi_callback(self, img_msg, dep_msg):
        self.get_logger().info('Multi msg')
  

        bridge = CvBridge()
        image_np = bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        pcd = pointcloud2_to_open3d(dep_msg)


        #pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        segmented_pc = o3d.geometry.PointCloud()
        segmented_images = recognize(image_np)
        if(len(segmented_images)!=0):
            for im in segmented_images:
                #mini_pc = pcd.crop(im)
                #segmented_pc += mini_pc
                indices_to_select = Cropinng(im, pcd)
                
                if(len(indices_to_select.points)!=0):
                    print("Всего точек: " + str(len(pcd.points)))
                    #selected_points = pcd.select_by_index(indices_to_select)
                    segmented_pc += indices_to_select
        print("Колличество сегментированных точек: " + str(len(segmented_pc.points)))
        
        self.pcd = point_cloud(np.asarray(segmented_pc.points), 'camera_link')
        self.pcd_publisher.publish(self.pcd)

        self.vis.clear_geometries()
        
        ##segmented_pc.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.vis.add_geometry(segmented_pc)

        self.vis.reset_view_point()
        self.vis.poll_events()
        self.vis.update_renderer()
        

    def destroy_node(self):
        self.image_sub.destroy()
        self.depth_sub.destroy()
        self.vis.destroy_window()
        super().destroy_node()



def main(args=None):
    rclpy.init(args=args)
    multi_sub = Multiply_Subscriber()

    rclpy.spin(multi_sub)
   
    multi_sub.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
