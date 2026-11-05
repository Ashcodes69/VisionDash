import math
import os
import urllib.request

import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Automatically download the required model file if it doesn't exist
os.makedirs("models", exist_ok=True)
MODEL_PATH = "models/face_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("Downloading MediaPipe Face Landmarker model (this only happens once)...")
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Download complete!")

# 2. Initialize the MediaPipe Tasks Face Landmarker
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1,
)
detector = vision.FaceLandmarker.create_from_options(options)

# Eye landmark indices for MediaPipe Face Mesh
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

EAR_THRESHOLD = 0.22


def calculate_distance(p1, p2):
    return math.dist(p1, p2)


def calculate_ear(eye_landmarks):
    v1 = calculate_distance(eye_landmarks[1], eye_landmarks[5])
    v2 = calculate_distance(eye_landmarks[2], eye_landmarks[4])
    h = calculate_distance(eye_landmarks[0], eye_landmarks[3])
    if h == 0:
        return 0.0
    return (v1 + v2) / (2.0 * h)


def get_landmark_coords(face_landmarks, indices, img_w, img_h):
    coords = []
    for idx in indices:
        # Note: The new Tasks API returns a list of landmarks directly
        landmark = face_landmarks[idx]
        x = int(landmark.x * img_w)
        y = int(landmark.y * img_h)
        coords.append((x, y))
    return coords


# Start Webcam
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv.flip(frame, 1)
    img_h, img_w, _ = frame.shape

    # 3. Convert frame to the new mp.Image format required by Tasks API
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # 4. Process the image
    detection_result = detector.detect(mp_image)

    state = "Searching for face..."
    color = (0, 0, 255)

    if detection_result.face_landmarks:
        for face_landmarks in detection_result.face_landmarks:
            right_eye = get_landmark_coords(
                face_landmarks, RIGHT_EYE_INDICES, img_w, img_h
            )
            left_eye = get_landmark_coords(
                face_landmarks, LEFT_EYE_INDICES, img_w, img_h
            )

            for pt in right_eye + left_eye:
                cv.circle(frame, pt, 2, (0, 255, 0), -1)

            right_ear = calculate_ear(right_eye)
            left_ear = calculate_ear(left_eye)
            avg_ear = (right_ear + left_ear) / 2.0

            if avg_ear < EAR_THRESHOLD:
                state = f"CLOSED (Blink) (EAR: {avg_ear:.2f}"
                color = (0, 0, 255)
            else:
                state = f"OPEN (EAR: {avg_ear:.2f})"
                color = (0, 255, 0)

    cv.putText(frame, state, (30, 50), cv.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv.LINE_AA)
    cv.imshow("VisionDash - Eye Tracker", frame)

    if cv.waitKey(1) == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
detector.close()
