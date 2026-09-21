import cv2
import urllib.request
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Download Model Asset if not present locally
model_path = 'pose_landmarker_heavy.task'
if not os.path.exists(model_path):
    print("Downloading MediaPipe Pose Model Asset (One-time download)...")
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
    urllib.request.urlretrieve(url, model_path)
    print("Download complete!")

# 2. Setup MediaPipe Landmarker Engine (Tasks API for MediaPipe 1.0+)
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_poses=1,
    min_pose_detection_confidence=0.5
)
detector = vision.PoseLandmarker.create_from_options(options)

# 33-Point Stick Skeleton Connection Mapping
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), # Arms
    (11, 23), (12, 24), (23, 24),                     # Torso
    (23, 25), (25, 27), (24, 26), (26, 28)            # Legs
]

cap = cv2.VideoCapture(0)

print("\n--- MODULE 2: 33-POINT SKELETON TRACKING (MediaPipe 1.0+) ---")
print("Press 'Q' to quit video feed.\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    
    # Convert BGR OpenCV Frame to MediaPipe Image Format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Detect Pose
    detection_result = detector.detect(mp_image)

    status_text = "SEARCHING FOR HUMAN POSE..."
    color = (0, 165, 255)

    if detection_result.pose_landmarks:
        landmarks = detection_result.pose_landmarks[0]
        
        # Draw Keypoint Dots
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

        # Draw Skeleton Sticks Lines
        for start_idx, end_idx in POSE_CONNECTIONS:
            pt1 = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
            pt2 = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
            cv2.line(frame, pt1, pt2, (255, 255, 0), 2)

        status_text = f"SKELETON ACTIVE: {len(landmarks)}/33 POINTS DETECTED"
        color = (0, 255, 0)

    # UI Panel
    cv2.rectangle(frame, (10, 10), (480, 50), (0, 0, 0), -1)
    cv2.putText(frame, status_text, (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    cv2.imshow("SIH 26187 - Module 2: 33-Point Stick Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()