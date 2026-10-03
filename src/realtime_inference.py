import os
import math
import joblib
import numpy as np
import mediapipe as mp
import threading
from pathlib import Path

# Safe platform check for Windows audio vs Cloud Linux
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

# ------------------------------------------------------------
# PROJECT PATHS & MODEL INITIALIZATION
# ------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
MODEL_PATH = PROJECT_ROOT / "models" / "fall_detection_model.pkl"

# Load Alert Agent if present
try:
    from .alert_agent import AlertResponseAgent
    alert_agent = AlertResponseAgent()
except Exception:
    alert_agent = None

# Load Trained Model
model = None
if MODEL_PATH.exists():
    try:
        model = joblib.load(MODEL_PATH)
        print(f"Machine learning model loaded successfully from: {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"Warning: Model file not found at {MODEL_PATH}")

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3,
)

# ------------------------------------------------------------
# PIPELINE STATE VARIABLES
# ------------------------------------------------------------
SEQUENCE_LENGTH = 30
frame_buffer = []
fall_latched = False
fall_trigger_counter = 0
head_y_history = []
alert_sent_for_current_fall = False
alarm_playing = False

# ------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------
def calculate_angle(p1, p2):
    """Calculates angle of vector p1 -> p2 relative to vertical axis."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6))

def play_alarm():
    global alarm_playing
    alarm_playing = True
    try:
        if HAS_WINSOUND:
            winsound.Beep(1200, 500)
    finally:
        alarm_playing = False

def trigger_alarm():
    global alarm_playing
    if not alarm_playing and HAS_WINSOUND:
        threading.Thread(target=play_alarm, daemon=True).start()

def reset_pipeline_state():
    """Resets tracking buffers for fresh video runs."""
    global frame_buffer, fall_latched, fall_trigger_counter, head_y_history, alert_sent_for_current_fall
    frame_buffer.clear()
    head_y_history.clear()
    fall_latched = False
    fall_trigger_counter = 0
    alert_sent_for_current_fall = False

# ------------------------------------------------------------
# CORE INFERENCE FUNCTION FOR APP.PY / WEBRTC
# ------------------------------------------------------------
def process_frame(frame):
    """
    Processes a single OpenCV BGR frame.
    Returns: (processed_frame, is_fall_boolean, status_text_string)
    """
    global fall_latched, fall_trigger_counter, head_y_history, frame_buffer, alert_sent_for_current_fall

    if frame is None:
        return frame, False, "EMPTY FRAME"

    if model is None:
        return frame, False, "MODEL NOT LOADED"

    import cv2

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    ml_fall_prob = 0.0
    status_text = "Normal Activity"

    if results.pose_landmarks:
        # Draw MediaPipe skeleton
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        lms = results.pose_landmarks.landmark

        # Joint coordinates
        left_s = [lms[mp_pose.PoseLandmark.LEFT_SHOULDER].x, lms[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
        right_s = [lms[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, lms[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
        left_h = [lms[mp_pose.PoseLandmark.LEFT_HIP].x, lms[mp_pose.PoseLandmark.LEFT_HIP].y]
        right_h = [lms[mp_pose.PoseLandmark.RIGHT_HIP].x, lms[mp_pose.PoseLandmark.RIGHT_HIP].y]

        s_mid = [(left_s[0] + right_s[0]) / 2.0, (left_s[1] + right_s[1]) / 2.0]
        h_mid = [(left_h[0] + right_h[0]) / 2.0, (left_h[1] + right_h[1]) / 2.0]

        torso_angle = calculate_angle(s_mid, h_mid)
        nose_y = lms[mp_pose.PoseLandmark.NOSE].y

        # Head velocity tracking
        head_y_history.append(nose_y)
        if len(head_y_history) > 10:
            head_y_history.pop(0)

        downward_velocity = (head_y_history[-1] - head_y_history[0]) if len(head_y_history) > 1 else 0.0

        # Construct ML feature vector (33 landmarks * 3 coords = 99 features)
        landmarks = []
        for lm in lms:
            landmarks.extend([lm.x, lm.y, lm.z])

        frame_buffer.append(landmarks)

        # Evaluate model prediction on frame buffer sequence
        if len(frame_buffer) == SEQUENCE_LENGTH:
            input_data = np.array(frame_buffer).flatten().reshape(1, -1)
            try:
                ml_fall_prob = model.predict_proba(input_data)[0][1]
            except Exception:
                ml_fall_prob = float(model.predict(input_data)[0])

            frame_buffer.pop(0)

        # Fall Detection Logic
        is_falling_motion = (
            downward_velocity > 0.12 or
            torso_angle > 50.0 or
            ml_fall_prob > 0.50
        )

        if is_falling_motion:
            fall_trigger_counter += 1
            if fall_trigger_counter >= 2:
                fall_latched = True

        # Recovery Logic
        is_standing_upright = (torso_angle < 30.0 and nose_y < 0.50)
        if is_standing_upright:
            fall_latched = False
            fall_trigger_counter = 0
            alert_sent_for_current_fall = False

    # Handle Alert State & Audio
    if fall_latched:
        status_text = f"FALL DETECTED! (Prob: {ml_fall_prob:.2f})"
        trigger_alarm()

        if alert_agent and not alert_sent_for_current_fall:
            try:
                alert_agent.create_alert(camera_name="Live Camera", confidence=ml_fall_prob)
            except Exception as e:
                print(f"Alert Agent Error: {e}")
            alert_sent_for_current_fall = True
    else:
        status_text = f"Normal Activity (Prob: {ml_fall_prob:.2f})"

    return frame, fall_latched, status_text
