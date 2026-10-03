import cv2
import numpy as np

# Robust, multi-path MediaPipe import strategy
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

def reset_pipeline_state():
    """Resets tracking buffers or counters if used."""
    pass

def process_frame(frame: np.ndarray):
    """
    Processes a single BGR frame for fall detection.
    Returns:
        processed_frame (np.ndarray): Frame with visual overlays
        is_fall (bool): True if fall detected
        status_text (str): Readable status message
    """
    if frame is None:
        return frame, False, "No Frame Received"

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    is_fall = False
    status_text = "NORMAL"

    if results.pose_landmarks:
        # Draw skeleton overlay
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        landmarks = results.pose_landmarks.landmark

        # Extract Keypoints: Nose (0), Left Hip (23), Right Hip (24), Left Ankle (27), Right Ankle (28)
        nose_y = landmarks[0].y
        left_hip_y = landmarks[23].y
        right_hip_y = landmarks[24].y
        hip_y = (left_hip_y + right_hip_y) / 2.0

        # Heuristic / Geometry Fall Condition: Head level drops near or below hip level
        if nose_y > hip_y or abs(nose_y - hip_y) < 0.12:
            is_fall = True
            status_text = "FALL DETECTED!"
        else:
            status_text = "PERSON STANDING"

    return frame, is_fall, status_text
