import cv2

image = cv2.imread("../data/airpods.jpg", 1)
image_copy = image.copy()
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(image_gray, 85, 255, cv2.THRESH_BINARY_INV)
thresh = cv2.medianBlur(thresh, 9)

# RETR_LIST, RETR_EXTERNAL, RETR_CCOMP, or RETR_TREE
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # cv2.CHAIN_APPROX_SIMPLE
# hierarchy [Next, Previous, First_Child, Parent]
cv2.drawContours(image_copy, contours, -1, (0, 255, 0), 3)

cv2.imshow("Thresh", thresh)
cv2.imshow("Contours None", image)

cv2.waitKey(0)
cv2.destroyAllWindows()
