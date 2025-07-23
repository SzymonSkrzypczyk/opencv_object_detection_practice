import cv2

imcap = cv2.VideoCapture(0)
# vid_capture = cv2.VideoCapture('Resources/Image_sequence/Cars%04d.jpg')

if not imcap.isOpened():
    exit()

imcap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
imcap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# print(imcap.get(cv2.CAP_PROP_FPS))
print(imcap.get(5)) # 5 is the index for CAP_PROP_FPS

while True:
    success, frame = imcap.read()

    if not success:
        break

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('e'):
        break

imcap.release()
cv2.destroyAllWindows()