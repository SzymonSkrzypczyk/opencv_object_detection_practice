import cv2
import numpy as np

image = cv2.imread("../data/blobs.jpg", 0)

image = cv2.GaussianBlur(image, (5, 5), 0)

## Sobel
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
sobel_xy = cv2.Sobel(image, cv2.CV_64F, 1, 1, ksize=5)

cv2.imshow("Sobel X", sobel_x)
cv2.imshow("Sobel Y", sobel_y)
cv2.imshow("Sobel XY", sobel_xy)

## Canny
edges = cv2.Canny(image, 100, 200)
cv2.imshow("Edges", edges)

cv2.waitKey(0)
cv2.destroyAllWindows()
