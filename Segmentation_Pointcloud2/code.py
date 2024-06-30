'''
Идея подхода решения задачи:
1. Обработка RGBD-изображения:
   - Загрузите RGBD-изображение из облака точек point_cloud2.
   - Преобразуйте его в формат, который можно использовать для детекции объектов с помощью нейросети YOLO.

2. Детекция объекта с помощью YOLO:
   - Используйте нейросеть YOLO для детекции губки на RGBD-изображении.
   - Получите координаты ограничивающего прямоугольника (bounding box) объекта (губки).

3. Сегментация губки:
   - Используйте полученные координаты bounding box для выделения области, содержащей губку, из облака точек point_cloud2.

4. Вывод результатов в Open3D:
   - Выведите область сегментированной губки во всплывающем окне Open3D.
'''

import open3d as o3d
count=0

# Вспомогательная функция для построения bounding box в Open3D
def draw_bounding_box(point_cloud, x, y, w, h):
    box = o3d.geometry.OrientedBoundingBox()
    box.center = (x + w/2, y + h/2, 0)  # Центр bounding box
    box.R = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # Ориентация (в данном случае оставляем по умолчанию)
    box.half_extents = (w / 2, h / 2, 0)  # Половина размера по каждой оси
    o3d.visualization.draw_geometries([point_cloud, box])  # Отображаем облако точек и bounding box

# Ваша функция recognize с дополнениями для построения bounding box
def recognize(image_np, point_cloud):
    # ... (ваш текущий код)
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

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]

            # Вывод bounding box на RGBD-изображении
            cv2.rectangle(image_np, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image_np, label, (x, y + 30), font, 3, color, 3)

            # Построение bounding box в Open3D
            draw_bounding_box(point_cloud, x, y, w, h)

    cv2.imshow("img", image_np)
    cv2.waitKey(1)
