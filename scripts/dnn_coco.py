from pathlib import Path
import cv2
import numpy as np

model = cv2.dnn.readNet(
    model="../models/frozen_inference_graph.pb",
    config="../models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt",
    framework="TensorFlow"
)

classes = Path("../models/object_detection_classes_coco.txt").read_text().splitlines()
classes = [c.strip() for c in classes]
class_colors = np.random.uniform(0, 255, size=(len(classes), 3))

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    blob = cv2.dnn.blobFromImage(image=frame, size=(300, 300), mean=(104, 117, 123), swapRB=True, crop=False)
    model.setInput(blob)
    outputs = model.forward()

    for detection in outputs[0, 0]:
        confidence = detection[2]
        if confidence > 0.5:
            class_id = int(detection[1]) - 1
            class_name = classes[class_id]
            class_color = class_colors[class_id]

            x1 = int(detection[3] * frame.shape[1])
            y1 = int(detection[4] * frame.shape[0])
            x2 = int(detection[5] * frame.shape[1])
            y2 = int(detection[6] * frame.shape[0])

            label = f"Class {class_name}: {confidence:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), class_color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, class_color, 2)

    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break