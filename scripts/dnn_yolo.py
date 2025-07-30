from pathlib import Path
import cv2
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("../models/YOLOv5s.onnx")
labels = Path("../models/coco_labels.txt").read_text().splitlines()

INPUT_WIDTH = 640
INPUT_HEIGHT = 640
CONF_THRESHOLD = 0.97
NMS_THRESHOLD = 0.45

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

def preprocess(_frame):
    image = cv2.resize(_frame, (INPUT_WIDTH, INPUT_HEIGHT))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image

def postprocess(_outputs, _frame):
    h, w = _frame.shape[:2]
    boxes, confidences, class_ids = [], [], []

    output = _outputs[0][0]
    for det in output:
        scores = det[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > CONF_THRESHOLD:
            cx, cy, bw, bh = det[0:4]
            x = int((cx - bw / 2) * w / INPUT_WIDTH)
            y = int((cy - bh / 2) * h / INPUT_HEIGHT)
            width = int(bw * w / INPUT_WIDTH)
            height = int(bh * h / INPUT_HEIGHT)

            boxes.append([x, y, width, height])
            confidences.append(float(confidence))
            class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)
    for i in indices:
        i = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
        box = boxes[i]
        x, y, w_box, h_box = box
        label = f"{labels[class_ids[i]]} {confidences[i]:.2f}"
        color = (0, 255, 0)

        cv2.rectangle(_frame, (x, y), (x + w_box, y + h_box), color, 2)
        cv2.putText(_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    input_image = preprocess(frame)
    outputs = session.run(None, {session.get_inputs()[0].name: input_image})
    postprocess(outputs, frame)

    cv2.imshow('YOLOv5 Object Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
