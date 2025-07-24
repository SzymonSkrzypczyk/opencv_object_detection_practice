import cv2
import numpy as np

image = cv2.imread("../data/airpods.jpg", 1)

identity_kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)  # identical image
laplace_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)  # laplacian kernel for sharpening
blur_kernel = np.ones((5, 5), dtype=np.float32) / 25  # average blur kernel

identity_image = cv2.filter2D(image, -1, identity_kernel)  # ddepth=-1 means the output image will have the same depth as the input image
blur_image = cv2.filter2D(image, -1, blur_kernel)
predefined_blur_image = cv2.blur(image, (5, 5))  # another way to apply a blur filter
gaussian_blur_image = cv2.GaussianBlur(image, (5, 5), 0)
median_image = cv2.medianBlur(image, 5)  # median filter, good for salt and pepper noise
laplacian_image = cv2.filter2D(image, -1, laplace_kernel)
bilatery_image = cv2.bilateralFilter(image, 9, 75, 75)  # bilateral filter, good for edge-preserving smoothing, but expensive in terms of computation

cv2.imshow("Original image", image)
cv2.imshow("Identity image", identity_image)
cv2.imshow("Blur image", blur_image)
cv2.imshow("Predefined blur image", predefined_blur_image)
cv2.imshow("Gaussian blur image", gaussian_blur_image)
cv2.imshow("Median image", median_image)
cv2.imshow("Laplacian image", laplacian_image)
cv2.imshow("Bilatery image", bilatery_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
