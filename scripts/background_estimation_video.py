import cv2
import numpy as np

cap = cv2.VideoCapture('../data/cctv.avi')
# extracting median frame from video
# it is based on the assumption that the background is static
# so its median will be the background
# and the rest of the frames are the foreground objects and will be calculated using difference

frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=25)

frames = []
for fid in frameIds:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = cap.read()
    frames.append(frame)

medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)
grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diffFrame = cv2.absdiff(grayMedianFrame, grayFrame)
    _, threshFrame = cv2.threshold(diffFrame, 30, 255, cv2.THRESH_BINARY)

    cv2.imshow('frame', frame)
    cv2.imshow('diff', diffFrame)
    cv2.imshow('thresh', threshFrame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
