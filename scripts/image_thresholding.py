import cv2

image = cv2.imread("../data/airpods.jpg", 0)
numbers = cv2.imread("../data/numbers.jpg", 1)
numbers = cv2.cvtColor(numbers, cv2.COLOR_BGR2GRAY)

th, dst = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)
th2, dst2 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)  # set to either 1 or 0
th3, dst3 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)  # set to either 1 or 0
th4, dst4 = cv2.threshold(image, 127, 255, cv2.THRESH_TRUNC)  # set to threshold value
th5, dst5 = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO)  # The destination pixel value is set to the pixel value of the corresponding source , if the source pixel value is greater than the threshold
th6, dst6 = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO_INV)

th7, dst7 = cv2.threshold(numbers, 0, 255, cv2.THRESH_BINARY)

cv2.imshow("Original", image)
cv2.imshow("Binary Global Thresholding", dst)
cv2.imshow("Binary Global Thresholding (127)", dst2)
cv2.imshow("Binary Inverted Global Thresholding (127)", dst3)
cv2.imshow("Truncated Global Thresholding (127)", dst4)
cv2.imshow("To Zero Global Thresholding (127)", dst5)
cv2.imshow("To Zero Inverted Global Thresholding (127)", dst6)
cv2.imshow("Binary Global Thresholding (Numbers)", dst7)

cv2.waitKey(0)
cv2.destroyAllWindows()
