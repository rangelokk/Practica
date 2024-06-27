1) Как обнаруживать объекты, используя YOLO, OpenCV и PyTorch в Python
https://waksoft.susu.ru/2021/05/19/kak-vypolnit-obnaruzhenie-obektov-yolo-s-pomoshhyu-opencv-i-pytorch-v-python/
(код+объяснения) (Совпадает с тем, что мы уже написали, есть coco.names, веса, blob)
2) (код для 1 пункта) Код с YOLO, darknet, файлы cfg  
https://github.com/x4nth055/pythoncode-tutorials/tree/master/machine-learning/object-detection  
3) Обнаружение объектов в режиме реального времени с помощью предварительно обученных YOLOv3 и OpenCV  
https://gurpreet-ai.github.io/real-time-object-detection-with-pre-trained-YOLOv3-and-opencv/  
4) Обнаружение объектов с помощью YOLO и OpenCV  
https://www.geeksforgeeks.org/object-detection-with-yolo-and-opencv/?ref=ml_lbp  
5) В этом проекте реализован классификатор обнаружения объектов на изображениях и видео в режиме реального времени с использованием предварительно обученных моделей yolov3.  
https://github.com/iArunava/YOLOv3-Object-Detection-with-OpenCV
6) *Про Yolo и ее установку и обучение*  
https://github.com/AlexeyAB/darknet?tab=readme-ov-file#how-to-train-to-detect-your-custom-objects
7) Разметка изображений  
https://github.com/AlexeyAB/Yolo_mark  
8) Как быстро создать обучающий датасет для задач обнаружения объектов YOLO с помощью Label Studio
https://habr.com/ru/articles/670532/
9) Как подготовить данные для обучения нейросети
https://www.freecodecamp.org/news/how-to-detect-objects-in-images-using-yolov8/
10) Сегментация в Yolo
https://docs.ultralytics.com/ru/tasks/segment/#export
11) Сегментация в YoloV3
https://cuda-chen.github.io/programming/2019/12/07/image-segmentation-with-YOLOv3-and-GrabCut.html
12) Как на своих данных обучить нейросеть(с кодом)
https://develop-nil.com/kak-sozdat-svoj-sobstvennyj-object-detector/
13) Тренировка нейросети
https://docs.ultralytics.com/ru/yolov5/tutorials/train_custom_data/#supported-environments
14) Инфа про YoloV3 на Гикс
https://www.geeksforgeeks.org/yolov3-from-scratch-using-pytorch/

Старый ImageNode:
```
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
```
