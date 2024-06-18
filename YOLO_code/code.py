"""
Если не видит, то вот так поправить:
# Путь к файлу с весами YOLO
weights_path = '/путь/к/папке/с/файлом/yolov3.weights'

# Путь к файлу конфигурации YOLO
config_path = '/путь/к/папке/с/файлом/yolov3.cfg'

# Загрузка весов и конфигурации YOLO
net = cv2.dnn.readNet(weights_path, config_path)
"""

import cv2

def recognize(image_np):
    # Загрузка предварительно обученной модели YOLO v3
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # Преобразование изображения в формат, с которым может работать YOLO
    blob = cv2.dnn.blobFromImage(image_np, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Детекция объектов
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Параметры ограничивающего прямоугольника
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Отображение детекции на изображении
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(image_np, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image_np, label, (x, y + 30), font, 3, color, 3)

    # Отображение изображения с детекцией
    cv2.imshow("Image", image_np)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
