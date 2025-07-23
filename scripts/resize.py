import cv2

image = cv2.imread("../data/airpods.jpg", 1)
h, w, c = image.shape  # height, width, channels
bigger_w = int(w * 1.5)
bigger_h = int(h * 1.5)
image_bigger = cv2.resize(image, (bigger_w, bigger_h), interpolation=cv2.INTER_LINEAR)
cv2.imshow('image_bigger', image_bigger)
smaller_w = int(w * 0.5)
smaller_h = int(h * 0.5)
image_smaller = cv2.resize(image, (smaller_w, smaller_h), interpolation=cv2.INTER_LINEAR)
cv2.imshow('image_smaller', image_smaller)

scale_bigger = 1.2
scale_smaller = 0.8

image_scaled_bigger = cv2.resize(image, None, fx=scale_bigger, fy=scale_bigger, interpolation=cv2.INTER_LINEAR)
image_scaled_smaller = cv2.resize(image, None, fx=scale_smaller, fy=scale_smaller, interpolation=cv2.INTER_LINEAR)

cv2.imshow('image_scaled_bigger', image_scaled_bigger)
cv2.imshow('image_scaled_smaller', image_scaled_smaller)

cv2.waitKey(0)
cv2.destroyAllWindows()
