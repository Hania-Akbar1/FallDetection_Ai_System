import os
import sys
import time
import json
import tempfile
from pathlib import Path
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# ---------------------------------------------------------
# IMPORT YOUR FALL DETECTION INFERENCE MODULE HERE
# ---------------------------------------------------------
try:
    from src.realtime_inference import process_frame
except ImportError:
    # Default fallback if import path differs
    def process_frame(frame):
        return frame, False, "SYSTEM READY"


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
ALERT_HISTORY_PATH = OUTPUTS_DIR / "alert_history.json"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="FallDetection.AI Dashboard",
    page_icon="🚨",
    layout="wide"
)

st.title("🚨 FallDetection.AI System")
st.caption("Real-Time Computer Vision & Machine Learning Fall Detection")

def load_alerts():
    if not ALERT_HISTORY_PATH.exists():
        return []
    try:
        with open(ALERT_HISTORY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []

alerts = load_alerts()

st.sidebar.header("System Controls")
input_mode = st.sidebar.radio(
    "Select Input Mode",
    ["📹 Upload Video File", "📷 Live Browser Camera", "📋 Alert Logs"]
)

st.sidebar.markdown("---")
st.sidebar.metric(label="Total Alerts Logged", value=len(alerts))

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class FallDetectionVideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        processed_img, is_fall, status = process_frame(img)
        return frame.from_ndarray(processed_img, format="bgr24")


# =========================================================
# OPTION 1: UPLOAD VIDEO FILE (SMOOTH PLAYBACK FIX)
# =========================================================
if input_mode == "📹 Upload Video File":
    st.subheader("📹 Video File Evaluation")
    st.write("Upload a video to run real-time pose estimation and fall detection.")

    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        # Save uploaded bytes to temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.avi')
        tfile.write(uploaded_file.read())
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 60:
            fps = 30.0  # Fallback standard frame rate
        
        frame_delay = 1.0 / fps

        if not cap.isOpened():
            st.error("Error opening video file stream.")
        else:
            st.success(f"Video loaded successfully ({int(cap.get(cv2.CAP_PROP_FRAME_COUNT))} frames @ {int(fps)} FPS).")

            # Playback Control Button
            start_btn = st.button("▶️ Start Fall Detection Analysis")

            if start_btn:
                st_frame = st.empty()
                status_box = st.empty()
                progress_bar = st.progress(0)

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                current_frame = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    current_frame += 1

                    # 1. Execute Inference Model Pipeline
                    processed_frame, fall_detected, status_text = process_frame(frame)

                    # 2. Convert BGR (OpenCV) to RGB (Streamlit display)
                    frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

                    # 3. Render frame in UI container
                    st_frame.image(frame_rgb, use_container_width=True)

                    # 4. Display live status & progress
                    if fall_detected:
                        status_box.error(f"🚨 ALERT: {status_text}")
                    else:
                        status_box.info(f"🟢 Status: {status_text}")

                    if total_frames > 0:
                        progress_bar.progress(min(current_frame / total_frames, 1.0))

                    # Pace frame rate for real-time visualization
                    time.sleep(frame_delay)

                cap.release()
                st.success("✅ Fall detection video evaluation completed.")

# =========================================================
# OPTION 2: LIVE WEBRTC CAMERA STREAM
# =========================================================
elif input_mode == "📷 Live Browser Camera":
    st.subheader("📷 Live WebRTC Camera Stream")
    webrtc_streamer(
        key="fall-detection-cam",
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=FallDetectionVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )

# =========================================================
# OPTION 3: ALERT LOGS
# =========================================================
elif input_mode == "📋 Alert Logs":
    st.subheader("📋 Recorded Alert Logs")
    if not alerts:
        st.info("No recorded fall events logged yet.")
    else:
        for alert in reversed(alerts):
            st.write(f"**Timestamp:** {alert.get('timestamp', 'N/A')} | **Status:** {alert.get('status', 'FALL DETECTED')}")
