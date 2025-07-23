import cv2
imcap = cv2.VideoCapture(0)
imcap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
imcap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

reference_image = cv2.imread('data/airpods.jpg', cv2.IMREAD_GRAYSCALE)
# reference_image = cv2.resize(reference_image, (640, 480))
keypoints_ref, descriptors_ref = orb.detectAndCompute(reference_image, None)

while True:
    success, frame = imcap.read()

    if not success:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = orb.detectAndCompute(frame, None)

    matches = bf.match(descriptors_ref, descriptors)
    matches = sorted(matches, key=lambda x: x.distance)

    frame = cv2.drawMatches(reference_image, keypoints_ref, frame, keypoints, matches[:10], None, flags=1)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

imcap.release()
cv2.destroyAllWindows()
