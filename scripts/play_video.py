import cv2

imcap = cv2.VideoCapture(0)
imcap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
imcap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while imcap.isOpened():
    ret, frame = imcap.read()

    if not ret:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('Webcam', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


imcap.release()
cv2.destroyAllWindows()
