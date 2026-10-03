import os
import sys
import time
import json
import tempfile
from pathlib import Path
import cv2
import streamlit as st

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

# App Header
st.title("🚨 FallDetection.AI System")
st.caption("Real-Time Computer Vision & Machine Learning Fall Detection")

# Helper to load persistent alerts
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

# Sidebar Navigation Controls
st.sidebar.header("System Controls")
input_mode = st.sidebar.radio(
    "Select Input Mode",
    ["📹 Upload Video File", "📷 Live Camera Test", "📋 Alert Logs"]
)

st.sidebar.markdown("---")
st.sidebar.metric(label="Total Alerts Logged", value=len(alerts))

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
            
            # Convert BGR (OpenCV) to RGB (Streamlit)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st_frame.image(frame_rgb, use_container_width=True)
            time.sleep(0.02)
            
        cap.release()
        st.success("Video processing complete.")

# Option 2: Live Camera Testing
elif input_mode == "📷 Live Camera Test":
    st.subheader("📷 Live Camera Testing")
    st.write("Test fall detection using your device camera in real time.")
    
    run_live = st.checkbox("Activate Camera Stream")
    camera_window = st.empty()

    if run_live:
        camera = cv2.VideoCapture(0)
        
        if not camera.isOpened():
            st.error("Unable to access local webcam on cloud container.")
            st.info("Note: Browser webcams on hosted cloud services require WebRTC (`streamlit-webrtc`) for browser-to-server streaming.")
        else:
            while run_live:
                ret, frame = camera.read()
                if not ret:
                    st.warning("Failed to grab camera frame.")
                    break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                camera_window.image(frame_rgb, use_container_width=True)
                time.sleep(0.03)
            
            camera.release()

# Option 3: Alert Logs
elif input_mode == "📋 Alert Logs":
    st.subheader("📋 Recorded Alert Logs")
    if not alerts:
        st.info("No recorded fall events logged yet.")
    else:
        for alert in reversed(alerts):
            st.write(f"**Timestamp:** {alert.get('timestamp', 'N/A')} | **Status:** {alert.get('status', 'FALL DETECTED')}")
