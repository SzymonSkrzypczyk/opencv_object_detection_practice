import cv2
import numpy as np

image = cv2.imread("../data/blobs.jpg", 1)
image_2 = image.copy()

## Simple Blob Detection
blob_detector = cv2.SimpleBlobDetector_create()
# Detect blobs
keypoints = blob_detector.detect(image)

# Draw detected blobs as red circles
image_blobs = cv2.drawKeypoints(image, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow("Keypoints", image_blobs)

## By params
params = cv2.SimpleBlobDetector_Params()

params.filterByArea = True
# .filterByCircularity = True
# .filterByConvexity = True
# .filterByInertia = True
# .filterByColor = True
params.minArea = 3000

detector_params = cv2.SimpleBlobDetector_create(params)
keypoints_2 = detector_params.detect(image)

image_blobs_2 = cv2.drawKeypoints(image, keypoints_2, np.array([]), (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow("Keypoints params", image_blobs_2)

cv2.waitKey(0)
cv2.destroyAllWindows()
