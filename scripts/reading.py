from pathlib import Path
import cv2

image = cv2.imread("../data/airpods.jpg", 1)

image_rot = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
# image_rot = cv2.cvtColor(image_rot, cv2.COLOR_BGR2RGB)

cv2.imshow('image', image_rot)
cv2.waitKey(0)
# cv2.destroyAllWindows()
cv2.destroyWindow("image")

cv2.imwrite("test.jpg", image_rot)
Path("test.jpg").unlink()
