from pathlib import Path
import cv2
import numpy as np

model = cv2.dnn.readNet(
    model="../models/DenseNet_121.caffemodel",
    config="../models/DenseNet_121.prototxt",
    framework="Caffe"
)
classes = Path("../models/classification_classes_ILSVRC2012.txt").read_text().splitlines()
classes = [c.split(",")[0].strip() for c in classes]


imcap = cv2.VideoCapture(0)
imcap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
imcap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while imcap.isOpened():
    ret, frame = imcap.read()

    if not ret:
        break

    blob = cv2.dnn.blobFromImage(image=frame, scalefactor=0.01, size=(224, 224), mean=(104, 117, 123))
    model.setInput(blob)
    outputs = model.forward()[0].reshape(1000, 1)
    label_id = np.argmax(outputs)

    probs = np.exp(outputs) / np.sum(np.exp(outputs))
    final_prob = np.max(probs) * 100.

    prob_text = f"{classes[label_id]}: {final_prob:.2f}%"
    cv2.putText(frame, prob_text, (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Webcam', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


imcap.release()
cv2.destroyAllWindows()