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

st.title("🚨 FallDetection.AI Dashboard")
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
mode = st.sidebar.radio("Select Input Mode", ["Upload Video File", "Alert Logs"])
st.sidebar.metric(label="Total Alerts Logged", value=len(alerts))

if mode == "Upload Video File":
    st.subheader("📹 Video File Evaluation")
    uploaded_file = st.file_uploader("Upload a video file to analyze", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        st_frame = st.empty()
        
        st.info("Processing video feed...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st_frame.image(frame_rgb, use_container_width=True)
            time.sleep(0.02)
            
        cap.release()
        st.success("Video processing finished.")

elif mode == "Alert Logs":
    st.subheader("📋 Fall Detection Logs")
    if not alerts:
        st.info("No recorded alerts.")
    else:
        for alert in reversed(alerts):
            st.write(f"**Timestamp:** {alert.get('timestamp', 'N/A')} | **Status:** {alert.get('status', 'FALL DETECTED')}")
