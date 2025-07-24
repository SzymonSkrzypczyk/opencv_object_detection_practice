import cv2

image = cv2.imread("../data/airpods.jpg", 1)

image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
image_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

cv2.imshow("BGR", image)
cv2.imshow("Lab", image_lab)  # good for changing light conditions, not for color detection
cv2.imshow("YCrCb", image_ycrcb)  # similar conclusions
cv2.imshow("HSV", image_hsv)

cv2.waitKey(0)
cv2.destroyAllWindows()
