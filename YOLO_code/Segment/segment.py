import rclpy
from rclpy.node import Node
import cv2

from sensor_msgs.msg import Image
import numpy as np
from cv_bridge import CvBridge
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
                cv2.imwrite('/home/mlserver/dataset/Segmented/s'+str(count)+'.png', image)
                count += 1

                label = str(classes[class_ids[i]]) #+ " x: " + str(x+w/2) + "y: " + str(y+h/2)
                color = (255, 255, 0)#colors[i]
                cv2.rectangle(image_np, (x,y), (x + w, y + h), color, 2)
                cv2.putText(image_np, label, (x, y + 30), font, 3, color, 3)
        '''
        if(len(boxes)!=0):
            cv2.imwrite('/home/mlserver/dataset/Segmented/s'+str(count)+'.png', image_np)
            count += 1
        '''
        cv2.imshow("img", image_np)
        cv2.waitKey(1)

class MinimalSubscriber(Node):
    i =0
    framerate = 7
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
    def listener_callback(self, msg):
        #self.get_logger().info('I heard: "%s"' % msg.data)
  
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


    

def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
