from datetime import datetime
import cv2
import numpy as np

# MediaPipe explicit import fallback
try:
    from mediapipe.python.solutions import pose as mp_pose
    from mediapipe.python.solutions import drawing_utils as mp_drawing
except ImportError:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

# Initialize Pose model
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Global thread-safe log storage & state tracking
FALL_LOGS = []
IS_CURRENTLY_FALLEN = False


def reset_pipeline_state():
    """Resets tracking buffers, log history, and fall state."""
    global FALL_LOGS, IS_CURRENTLY_FALLEN
    FALL_LOGS.clear()
    IS_CURRENTLY_FALLEN = False


def get_fall_history():
    """Returns the list of recorded fall events."""
    return FALL_LOGS


def process_frame(frame: np.ndarray, source_mode: str = "Live Feed"):
    """
    Processes a single BGR frame for fall detection and updates event logs.
    
    Returns:
        processed_frame (np.ndarray): Frame with skeleton overlay and text banner
        is_fall (bool): True if a fall is detected in current frame
        status_text (str): Readable status message
    """
    global IS_CURRENTLY_FALLEN, FALL_LOGS

    if frame is None:
        return frame, False, "No Frame Received"

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    is_fall = False
    status_text = "NORMAL"

    if results.pose_landmarks:
        # Draw pose keypoints
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        landmarks = results.pose_landmarks.landmark

        # Extract Keypoints: Nose (0), Left Hip (23), Right Hip (24)
        nose_y = landmarks[0].y
        left_hip_y = landmarks[23].y
        right_hip_y = landmarks[24].y
        hip_y = (left_hip_y + right_hip_y) / 2.0

        # Heuristic Fall Condition
        if nose_y > hip_y or abs(nose_y - hip_y) < 0.12:
            is_fall = True
            status_text = "FALL DETECTED!"
            
            # Log new fall event (with state debouncing)
            if not IS_CURRENTLY_FALLEN:
                IS_CURRENTLY_FALLEN = True
                log_entry = {
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Event": "FALL DETECTED",
                    "Source": source_mode,
                    "Confidence": "HIGH",
                    "Status": "Alert Triggered"
                }
                FALL_LOGS.append(log_entry)
        else:
            status_text = "PERSON STANDING"
            IS_CURRENTLY_FALLEN = False

    return frame, is_fall, status_text
