
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
#import ros_numpy
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
                # extract the bounding box coordinates
                rect = boxes[i]
                # apply GrabCut
                cv2.grabCut(image, mask, rect, bgdModel, fgbModel, 5, cv2.GC_INIT_WITH_RECT)
                mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
                image = image * mask2[:, :, np.newaxis]
                cv2.imwrite('/home/mlserver/dataset/Segmented/s'+str(count)+'.png', image) #для сохранения сегментированных изобр в реал.времени
                count += 1

                label = str(classes[class_ids[i]]) #+ " x: " + str(x+w/2) + "y: " + str(y+h/2)
                color = (255, 255, 0)#colors[i]
                cv2.rectangle(image_np, (x,y), (x + w, y + h), color, 2)
                cv2.putText(image_np, label, (x, y + 30), font, 3, color, 3)
        cv2.imshow("img", image_np)
        cv2.waitKey(1)

class ImageSubscriber(Node):
    i =0
    framerate = 7
    def __init__(self):
        super().__init__('image_subscriber')
        Qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_RELIABLE,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
            depth=5
        )
        self.subscription = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.image_callback,
            qos_profile=Qos_profile)
        self.subscription  # prevent unused variable warning
    def image_callback(self, msg):
        self.get_logger().info('Image msg')
  
        bridge = CvBridge()
        image_np = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        
        recognize(image_np)

        cv2.imshow("img", image_np)
        '''
        Для съемок
        if(self.i%self.framerate==0):
            cv2.imwrite('/home/mlserver/dataset/m2_Sponge/ms2'+str(self.i/self.framerate)+'.png', image_np)
        self.i += 1
        '''
        cv2.waitKey(1)


def pointcloud2_to_open3d(data):
        points = []
        for p in point_cloud2.read_points(data, field_names=("x", "y", "z"), skip_nans=True):
            points.append([p[0], p[1], p[2]])
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        return point_cloud
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

#Для облака точек
class Depth_Subscriber(Node):
    def __init__(self):
        super().__init__('depth_subscriber')
        Qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
            depth=5
        )
        self.subscription = self.create_subscription(
            PointCloud2,
            '/camera/depth_registered/points',
            self.depth_callback,
            qos_profile=Qos_profile)
        self.subscription  # prevent unused variable warning
    def depth_callback(self, msg):
        self.get_logger().info('Depth Image')
        ##xyz, rgb = pointcloud2_to_array(ros_cloud)
        pcd = pointcloud2_to_open3d(msg)
        '''
        Pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        '''
        o3d.visualization.draw_geometries([pcd])

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
        self.pcd = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd)


    def multi_callback(self, img_msg, dep_msg):
        self.get_logger().info('Multi msg')
  
        bridge = CvBridge()
        image_np = bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        
        recognize(image_np)


        self.pcd = pointcloud2_to_open3d(dep_msg)
        #self.pcd.points.extend(pointcloud2_to_open3d(dep_msg))
        self.vis.update_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()

        #o3d.visualization.draw_geometries([pcd])





    def destroy_node(self):
        self.image_sub.destroy()
        self.depth_sub.destroy()
        self.vis.destroy_window()
        super().destroy_node()



def main(args=None):
    rclpy.init(args=args)

    #image_subscriber = ImageSubscriber()
    #depth_subscriber = Depth_Subscriber() 
    multi_sub = Multiply_Subscriber()

    #rclpy.spin(image_subscriber)
    #rclpy.spin(depth_subscriber)


    rclpy.spin(multi_sub)
    
    #image_subscriber.destroy_node()
    #depth_subscriber.destroy_node()

    multi_sub.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
