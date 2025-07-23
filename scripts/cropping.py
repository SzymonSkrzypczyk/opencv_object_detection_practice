import cv2

image = cv2.imread("../data/airpods.jpg", -1)
h, w, c = image.shape

cv2.imshow('original_image', image)

cropped = image[w // 2 - 200 : w // 2 + 200, h // 2 - 200 : h // 2 + 200]  # start_row:end_row, start_col:end_col

cv2.imshow('cropped_image', cropped)

cv2.waitKey(0)
cv2.destroyAllWindows()
