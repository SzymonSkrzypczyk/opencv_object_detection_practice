import cv2
import numpy as np

image = cv2.imread('../data/airpods.jpg', cv2.IMREAD_COLOR)

height, width = image.shape[:2]
center = (width / 2, height / 2)

rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=45, scale=1)
rotated_image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(width, height))

cv2.imshow('Original image', image)
cv2.imshow('Rotated image', rotated_image)

tx, ty = width / 4, height / 4
translation_matrix = np.array([
    [1, 0, tx],
    [0, 1, ty]
], dtype=np.float32)  # move the image by tx pixels right and ty pixels down

translated_image = cv2.warpAffine(image, translation_matrix, (width, height))

cv2.imshow('Translated image', translated_image)

cv2.waitKey(0)
cv2.destroyAllWindows()