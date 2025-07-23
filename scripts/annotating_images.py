import cv2

image = cv2.imread('../data/airpods.jpg', 1)
h, w, _ = image.shape
image_line = image.copy()
image_circle = image.copy()
image_rectangle = image.copy()
image_ellipse = image.copy()
image_text = image.copy()

image_line = cv2.line(image_line, (0, 0), (2000, 1000), (255, 0, 0), 15)
image_circle = cv2.circle(image_circle, (w // 2, h // 2 + 300), 800, (255, 0, 0), 15)
image_rectangle = cv2.rectangle(image_rectangle, (0, 0), (w // 2, h // 2), (255, 0, 0), 15)
image_ellipse = cv2.ellipse(image_ellipse, (w // 2, h // 2), (w // 4, h // 8), 0, 0, 180, (255, 0, 0), 15)
image_text = cv2.putText(image_text, "Hello There!", (w // 2 - 100, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5, cv2.LINE_AA)
# thickness = -1 to fill the circle

cv2.imshow('image_line', image_line)
cv2.imshow('image_circle', image_circle)
cv2.imshow('image_rectangle', image_rectangle)
cv2.imshow('image_ellipse', image_ellipse)
cv2.imshow('image_text', image_text)

cv2.waitKey(0)
cv2.destroyAllWindows()
