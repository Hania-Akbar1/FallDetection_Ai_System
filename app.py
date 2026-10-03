import os
import sys
import time
import json
import tempfile
from pathlib import Path
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# Setup Paths
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
ALERT_HISTORY_PATH = OUTPUTS_DIR / "alert_history.json"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Page Configuration
st.set_page_config(
    page_title="FallDetection.AI Dashboard",
    page_icon="🚨",
    layout="wide"
)

st.title("🚨 FallDetection.AI System")
st.caption("Real-Time Computer Vision & Machine Learning Fall Detection")

# Load Alerts Helper
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

# Sidebar Navigation
st.sidebar.header("System Controls")
input_mode = st.sidebar.radio(
    "Select Input Mode",
    ["📹 Upload Video File", "📷 Live Browser Camera", "📋 Alert Logs"]
)

st.sidebar.markdown("---")
st.sidebar.metric(label="Total Alerts Logged", value=len(alerts))

# STUN server configuration for cloud WebRTC connectivity
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Custom Video Processor Class for WebRTC Frame Manipulation
class FallDetectionVideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # -----------------------------------------------------------
        # NOTE: Pass `img` into your MediaPipe / ML model pipeline here
        # Example: annotated_img = run_fall_detection_model(img)
        # -----------------------------------------------------------
        
        return frame.from_ndarray(img, format="bgr24")

# Option 1: Upload Video File
if input_mode == "📹 Upload Video File":
    st.subheader("📹 Video File Evaluation")
    st.write("Upload a pre-recorded video to run fall detection inference.")
    
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        st_frame = st.empty()
        
        st.info("Processing video stream...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st_frame.image(frame_rgb, use_container_width=True)
            time.sleep(0.02)
            
        cap.release()
        st.success("Video processing complete.")

# Option 2: Live Browser Camera Stream via WebRTC
elif input_mode == "📷 Live Browser Camera":
    st.subheader("📷 Live WebRTC Camera Stream")
    st.write("Allow browser permissions to test real-time webcam detection directly on the cloud.")
    
    webrtc_streamer(
        key="fall-detection-cam",
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=FallDetectionVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )

# Option 3: Alert Logs
elif input_mode == "📋 Alert Logs":
    st.subheader("📋 Recorded Alert Logs")
    if not alerts:
        st.info("No recorded fall events logged yet.")
    else:
        for alert in reversed(alerts):
            st.write(f"**Timestamp:** {alert.get('timestamp', 'N/A')} | **Status:** {alert.get('status', 'FALL DETECTED')}")
