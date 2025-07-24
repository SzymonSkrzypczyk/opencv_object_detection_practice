import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

BG_LOWER_HSV_COLOR = np.array([0, 0, 40], dtype=np.uint8)
BG_UPPER_HSV_COLOR = np.array([179, 50, 220], dtype=np.uint8)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    output_frame = frame.copy()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    bg_mask = cv2.inRange(hsv, BG_LOWER_HSV_COLOR, BG_UPPER_HSV_COLOR)
    fg_mask = cv2.bitwise_not(bg_mask)

    kernel = np.ones((3, 3), np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 300:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Detected Objects', output_frame)
    cv2.imshow('Mask', fg_mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
