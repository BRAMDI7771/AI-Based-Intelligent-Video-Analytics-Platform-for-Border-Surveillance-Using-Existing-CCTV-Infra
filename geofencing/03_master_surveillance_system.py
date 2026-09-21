import cv2
import numpy as np
import urllib.request
import os
import math
import time
from datetime import datetime
from collections import deque
from shapely.geometry import Point, Polygon
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import winsound

# --- 0. DIRECTORIES & LOGGING SETUP ---
snapshot_dir = "breach_snapshots"
os.makedirs(snapshot_dir, exist_ok=True)
log_file = "surveillance_logs.txt"

def play_alarm():
    try:
        winsound.Beep(2500, 250)
    except Exception:
        pass

def log_breach(person_id, intent_text):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a") as f:
        f.write(f"[{timestamp}] ALERT: Red Zone Breach! Target #{person_id} | Intent: {intent_text}\n")

# --- 1. SETUP MODEL ASSET ---
model_path = 'pose_landmarker_heavy.task'
if not os.path.exists(model_path):
    print("Downloading Model Asset...")
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
    urllib.request.urlretrieve(url, model_path)

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_poses=5, 
    min_pose_detection_confidence=0.3,
    min_pose_presence_confidence=0.3
)
detector = vision.PoseLandmarker.create_from_options(options)

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28)
]

polygon_points = []
geofence_locked = False
digital_fence = None
person_histories = {}
last_snapshot_time = 0

def mouse_draw_geofence(event, x, y, flags, param):
    global polygon_points, geofence_locked
    if event == cv2.EVENT_LBUTTONDOWN and not geofence_locked:
        polygon_points.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        polygon_points = []
        geofence_locked = False

# --- 2. WEBCAM & WINDOW SETUP ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

win_name = "SIH 26187 - Master Border Surveillance System (Multi-Target)"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win_name, 1280, 720)
cv2.setMouseCallback(win_name, mouse_draw_geofence)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    
    # Draw Geofence Points
    if len(polygon_points) > 0:
        pts = np.array(polygon_points, np.int32).reshape((-1, 1, 2))
        if geofence_locked:
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=3)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], color=(0, 0, 200))
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        else:
            cv2.polylines(frame, [pts], isClosed=False, color=(0, 255, 255), thickness=2)
            for pt in polygon_points:
                cv2.circle(frame, pt, 4, (0, 255, 255), -1)

    # Detection Logic
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)

    zone_status = "ZONE SECURE"
    zone_color = (0, 255, 0)
    total_detected = 0
    any_breach = False

    if detection_result.pose_landmarks:
        total_detected = len(detection_result.pose_landmarks)

        for i, landmarks in enumerate(detection_result.pose_landmarks):
            person_id = i + 1
            
            for lm in landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

            for p1, p2 in POSE_CONNECTIONS:
                pt1 = (int(landmarks[p1].x * w), int(landmarks[p1].y * h))
                pt2 = (int(landmarks[p2].x * w), int(landmarks[p2].y * h))
                cv2.line(frame, pt1, pt2, (255, 255, 0), 2)

            nose_x, nose_y = int(landmarks[0].x * w), int(landmarks[0].y * h)
            l_sh_x, l_sh_y = int(landmarks[11].x * w), int(landmarks[11].y * h)
            r_sh_x, r_sh_y = int(landmarks[12].x * w), int(landmarks[12].y * h)
            
            center_x = int((l_sh_x + r_sh_x) / 2)
            center_y = int((l_sh_y + r_sh_y) / 2)

            person_height = abs(center_y - nose_y) * 2
            person_width = abs(l_sh_x - r_sh_x) + 1
            aspect_ratio = person_height / float(person_width)

            if person_id not in person_histories:
                person_histories[person_id] = deque(maxlen=10)
            person_histories[person_id].append((center_x, center_y))

            history = person_histories[person_id]
            speed = 0
            if len(history) > 1:
                dx = history[-1][0] - history[0][0]
                dy = history[-1][1] - history[0][1]
                speed = math.sqrt(dx**2 + dy**2) / len(history)

            if aspect_ratio < 0.8:
                intent_text = "CRAWLING (THREAT)"
                label_color = (0, 165, 255)
            elif speed > 10:
                intent_text = "RUNNING (HIGH THREAT)"
                label_color = (0, 0, 255)
            elif speed > 5:
                intent_text = "FAST APPROACH"
                label_color = (0, 255, 255)
            else:
                intent_text = "PATROLLING"
                label_color = (0, 255, 0)

            cv2.putText(frame, f"ID #{person_id}: {intent_text}", (center_x - 40, center_y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 2)

            if geofence_locked and digital_fence:
                head_in_zone = digital_fence.contains(Point(nose_x, nose_y))
                torso_in_zone = digital_fence.contains(Point(center_x, center_y))
                
                if head_in_zone or torso_in_zone:
                    any_breach = True
                    curr_time = time.time()
                    if curr_time - last_snapshot_time > 3:
                        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                        snap_path = os.path.join(snapshot_dir, f"breach_ID{person_id}_{time_str}.jpg")
                        cv2.imwrite(snap_path, frame)
                        log_breach(person_id, intent_text)
                        last_snapshot_time = curr_time

    if any_breach:
        zone_status = "CRITICAL: RED ZONE BREACH DETECTED!"
        zone_color = (0, 0, 255)
        play_alarm()

    # Dashboard HUD
    cv2.rectangle(frame, (10, 10), (680, 110), (0, 0, 0), -1)
    cv2.putText(frame, f"STATUS: {zone_status}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, zone_color, 2)
    cv2.putText(frame, f"TARGETS DETECTED: {total_detected}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, "GEOFENCE: Click points & press 'S' to lock | 'R' to reset", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    cv2.imshow(win_name, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') or key == ord('S'):
        if len(polygon_points) >= 3:
            # Force valid polygon convex hull to avoid crash
            poly_np = np.array(polygon_points, dtype=np.int32)
            hull = cv2.convexHull(poly_np)
            hull_pts = [tuple(pt[0]) for pt in hull]
            digital_fence = Polygon(hull_pts)
            geofence_locked = True
    elif key == ord('r') or key == ord('R'):
        polygon_points = []
        geofence_locked = False
        digital_fence = None
        person_histories.clear()
    elif key == ord('q') or key == ord('Q'):
        break

cap.release()
cv2.destroyAllWindows()