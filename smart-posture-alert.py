import cv2
import mediapipe as mp
import numpy as np
import time
import sys


SLUCH_ANGLE_THRESHOLD = 30      
SLUCH_DURATION_THRESHOLD = 30   
CAMERA_INDEX = 0               

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,          
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils


def alert():
    if sys.platform == "win32":
        import winsound
        winsound.Beep(1000, 500)   
    else:
        print('\a', end='', flush=True)  

def angle_between(v1, v2):
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0:
        return 0.0
  
    cos = max(-1.0, min(1.0, dot / norm))
    return np.degrees(np.arccos(cos))

def is_slouching(ear, shoulder):
    vec = np.array([ear[0] - shoulder[0], ear[1] - shoulder[1]])
   
    upward = np.array([0, -1])
    ang = angle_between(vec, upward)
    return ang > SLUCH_ANGLE_THRESHOLD, ang


cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("Cannot open webcam")
    sys.exit(1)


slouch_start = None  
alert_active = False 

print("Posture Alerter started. Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = pose.process(rgb)
    rgb.flags.writeable = True

    posture_text = "No person detected"
    slouch_duration = 0

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

   
        h, w, _ = frame.shape
        lm = results.pose_landmarks.landmark

      
        left_ear = (lm[7].x * w, lm[7].y * h)
        right_ear = (lm[8].x * w, lm[8].y * h)
        left_shoulder = (lm[11].x * w, lm[11].y * h)
        right_shoulder = (lm[12].x * w, lm[12].y * h)

    
        ear_visible = False
        shoulder_visible = False
        ear = None
        shoulder = None

        if lm[7].visibility > 0.5 and lm[8].visibility > 0.5:
            ear = np.mean([left_ear, right_ear], axis=0)
            ear_visible = True
        elif lm[7].visibility > 0.5:
            ear = left_ear
            ear_visible = True
        elif lm[8].visibility > 0.5:
            ear = right_ear
            ear_visible = True

        if lm[11].visibility > 0.5 and lm[12].visibility > 0.5:
            shoulder = np.mean([left_shoulder, right_shoulder], axis=0)
            shoulder_visible = True
        elif lm[11].visibility > 0.5:
            shoulder = left_shoulder
            shoulder_visible = True
        elif lm[12].visibility > 0.5:
            shoulder = right_shoulder
            shoulder_visible = True

        if ear_visible and shoulder_visible:
            slouching, angle = is_slouching(ear, shoulder)
            cv2.putText(frame, f"Angle: {angle:.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if slouching:
                posture_text = "SLOUCHING"
                if slouch_start is None:
                    slouch_start = time.time()
                else:
                    slouch_duration = time.time() - slouch_start
                    if slouch_duration >= SLUCH_DURATION_THRESHOLD and not alert_active:
                        alert()
                        alert_active = True
            else:
                posture_text = "Upright"
                slouch_start = None
                alert_active = False

  
            if slouch_start is not None:
                timer_str = f"Slouching: {slouch_duration:.1f}s / {SLUCH_DURATION_THRESHOLD}s"
                cv2.putText(frame, timer_str, (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    color = (0, 255, 0) if "Upright" in posture_text else (0, 0, 255)
    cv2.putText(frame, posture_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Posture Alerter", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pose.close()
