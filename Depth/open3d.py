#27 июня

import rclpy
from rclpy.node import Node
import cv2

import message_filters
from sensor_msgs.msg import Image
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

                image = image_np.copy()

                mask = np.zeros(image_np.shape[:2], np.uint8)
                bgdModel = np.zeros((1, 65), np.float64)
                fgbModel = np.zeros((1, 65), np.float64)
                rect = boxes[i]
                cv2.grabCut(image, mask, rect, bgdModel, fgbModel, 5, cv2.GC_INIT_WITH_RECT)
                mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
                image = image * mask2[:, :, np.newaxis]
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

# Преобразование цветов в формат RGB
    '''
    colors_rgb = [(int(color) & 0xff, (int(color) >> 8) & 0xff, (int(color) >> 16) & 0xff) for color in colors]
    point_cloud.colors = o3d.utility.Vector3dVector(np.asarray(colors_rgb) / 255.0)
    '''

def pointcloud2_to_array(pointcloud2: PointCloud2) -> tuple:
    """
    Convert a ROS PointCloud2 message to a numpy array.

    Args:
        pointcloud2 (PointCloud2): the PointCloud2 message

    Returns:
        (tuple): tuple containing (xyz, rgb)
    """
    """
    pc_array = ros_numpy.point_cloud2.pointcloud2_to_array(pointcloud2)
    split = ros_numpy.point_cloud2.split_rgb_field(pc_array)
    rgb = np.stack([split["b"], split["g"], split["r"]], axis=2)
    xyz = ros_numpy.point_cloud2.get_xyz_points(pc_array, remove_nans=False)
    xyz = np.array(xyz).reshape((pointcloud2.height, pointcloud2.width, 3))
    nan_rows = np.isnan(xyz).all(axis=2)
    xyz[nan_rows] = [0, 0, 0]
    rgb[nan_rows] = [0, 0, 0]
    return xyz, rgb
    """
    pass

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

        self.ts = ApproximateTimeSynchronizer([self.image_sub, self.depth_sub], 10, 0.1)
        self.ts.registerCallback(self.multi_callback)
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(height=480, width=640)
        self.vis.add_geometry(o3d.geometry.PointCloud())

    def multi_callback(self, img_msg, dep_msg):
        self.get_logger().info('Multi msg')
  
        bridge = CvBridge()
        image_np = bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        
        recognize(image_np)

        self.vis.clear_geometries()
        self.vis.add_geometry(pointcloud2_to_open3d(dep_msg))
        self.vis.get_view_control().change_field_of_view(self.vis.get_view_control().get_field_of_view() +30)
        print("Field of view (after changing) %.2f" % self.vis.get_view_control().get_field_of_view())
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

if __name__ == '__main__':
    main()
