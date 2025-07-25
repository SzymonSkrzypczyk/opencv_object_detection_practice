import cv2
import mediapipe as mp
from fontTools.ttLib.scaleUpem import visit
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# setup the gesture recognizer
base_options = BaseOptions(
            model_asset_path='../models/gesture_recognizer.task'
        )
options = vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE
)

gesture_recognizer = vision.GestureRecognizer.create_from_options(options)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    gestures = gesture_recognizer.recognize(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame))
    result = "No Gesture Detected"
    if gestures.gestures:
        result = f"GESTURE: {gestures.gestures[0][0].category_name.upper()} - Confidence: {gestures.gestures[0][0].score:.2f}"
    print(result)
    """
    results = hands.process(rgb_frame)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    """
    cv2.putText(frame, result, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
