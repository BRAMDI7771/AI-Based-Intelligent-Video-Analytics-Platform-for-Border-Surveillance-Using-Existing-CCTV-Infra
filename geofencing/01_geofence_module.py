import cv2
import numpy as np
from shapely.geometry import Point, Polygon

# Global variables for drawing polygon
polygon_points = []
geofence_locked = False
digital_fence = None

def mouse_draw_geofence(event, x, y, flags, param):
    global polygon_points, geofence_locked, digital_fence
    
    # Left click to add boundary points
    if event == cv2.EVENT_LBUTTONDOWN and not geofence_locked:
        polygon_points.append((x, y))
        print(f"Point added: ({x}, {y})")

    # Right click to reset/clear boundary
    elif event == cv2.EVENT_RBUTTONDOWN:
        polygon_points = []
        geofence_locked = False
        digital_fence = None
        print("Geofence Reset!")

# Setup Camera Feed (0 for Webcam or pass video file path 'border_feed.mp4')
cap = cv2.VideoCapture(0)

cv2.namedWindow("SIH 26187 - Module 1: Geofence Setup")
cv2.setMouseCallback("SIH 26187 - Module 1: Geofence Setup", mouse_draw_geofence)

print("\n--- INSTRUCTIONS ---")
print("1. Click LEFT MOUSE BUTTON to draw boundary points (Digital Polygon Fence).")
print("2. Press 'S' key to LOCK the geofence.")
print("3. Press 'R' or RIGHT MOUSE CLICK to Reset Boundary.")
print("4. Press 'Q' to Quit.\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # Draw Polygon in real-time
    if len(polygon_points) > 0:
        pts = np.array(polygon_points, np.int32).reshape((-1, 1, 2))
        
        if geofence_locked:
            # Locked Fence (Red Zone)
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=3)
            
            # Fill semi-transparent red overlay inside fence
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], color=(0, 0, 200))
            cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
        else:
            # Drawing phase (Yellow lines)
            cv2.polylines(frame, [pts], isClosed=False, color=(0, 255, 255), thickness=2)
            for pt in polygon_points:
                cv2.circle(frame, pt, 5, (0, 255, 0), -1)

    # UI Instructions Overlay
    cv2.rectangle(frame, (10, 10), (520, 80), (0, 0, 0), -1)
    status_text = "FENCE LOCKED" if geofence_locked else "DRAWING FENCE (Click points & press 'S')"
    status_color = (0, 0, 255) if geofence_locked else (0, 255, 255)
    
    cv2.putText(frame, f"STATUS: {status_text}", (20, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    cv2.putText(frame, f"TOTAL BOUNDARY POINTS: {len(polygon_points)}", (20, 65), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("SIH 26187 - Module 1: Geofence Setup", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') or key == ord('S'):
        if len(polygon_points) >= 3:
            geofence_locked = True
            digital_fence = Polygon(polygon_points)
            print("Geofence Boundary Successfully Saved!")
        else:
            print("Minimum 3 points needed to create a valid Polygon zone!")
    elif key == ord('r') or key == ord('R'):
        polygon_points = []
        geofence_locked = False
        digital_fence = None
    elif key == ord('q') or key == ord('Q'):
        break

cap.release()
cv2.destroyAllWindows()
