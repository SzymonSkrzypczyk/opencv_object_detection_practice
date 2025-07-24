import cv2

model = cv2.dnn.readNetFromONNX("../models/YOLOv5s.onnx")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    blob = cv2.dnn.blobFromImage(frame, 1./255, (640, 640), (0, 0, 0), True, crop=False)
    model.setInput(blob)

    outputs = model.forward()
    print(outputs)

    cv2.imshow('frame', frame)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
