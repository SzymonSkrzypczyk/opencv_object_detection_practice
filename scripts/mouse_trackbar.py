import cv2

top_left_corner = []
bottor_right_corner = []
image = cv2.imread("../data/airpods.jpg", 1)


def draw_rectangle(event, x, y, flags, *param):
    global top_left_corner, bottor_right_corner

    if event == cv2.EVENT_LBUTTONDOWN:
        top_left_corner = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        bottor_right_corner = (x, y)
        cv2.rectangle(image, top_left_corner, bottor_right_corner, (0, 255, 0), 2)
        cv2.imshow("Image", image)


temp = image.copy()
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", draw_rectangle)

k = 0

while k != 27:  # Press ESC to exit
    cv2.imshow("Image", image)
    k = cv2.waitKey(0)
    if k == 99:
        image = temp.copy()
        cv2.imshow("Image", image)

cv2.destroyAllWindows()