import cv2
imcap = cv2.VideoCapture(0)
imcap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
imcap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    success, frame = imcap.read()

    if not success:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

imcap.release()
cv2.destroyAllWindows()
