import os
import sys
import time
import json
import tempfile
import traceback
from pathlib import Path
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# ---------------------------------------------------------
# 1. RESOLVE PATHS & IMPORT INFERENCE PIPELINE
# ---------------------------------------------------------
FILE_PATH = Path(__file__).resolve()
# Handles execution whether app.py is in DASHBOARD/ or root directory
PROJECT_ROOT = FILE_PATH.parent.parent if FILE_PATH.parent.name == "DASHBOARD" else FILE_PATH.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Page Configuration
st.set_page_config(
    page_title="FallDetection.AI System",
    page_icon="🚨",
    layout="wide"
)

st.title("🚨 FallDetection.AI System")
st.caption("Real-Time Computer Vision & Machine Learning Fall Detection")

# Debug Import & Model Loading
MODEL_LOADED = False
try:
    from src.realtime_inference import process_frame
    MODEL_LOADED = True
except Exception as e:
    try:
        from realtime_inference import process_frame
        MODEL_LOADED = True
    except Exception as e2:
        st.error(f"⚠️ Model Load / Import Error: {e2}")
        st.code(traceback.format_exc())
        
        def process_frame(frame):
            # Fallback dummy function to keep UI responsive
            return frame, False, "MODEL NOT LOADED"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
ALERT_HISTORY_PATH = OUTPUTS_DIR / "alert_history.json"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# 2. HELPER FUNCTIONS FOR ALERT LOGGING
# ---------------------------------------------------------
def load_alerts():
    if not ALERT_HISTORY_PATH.exists():
        return []
    try:
        with open(ALERT_HISTORY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []

def log_alert(status_text):
    alerts = load_alerts()
    new_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": status_text
    }
    alerts.append(new_entry)
    try:
        with open(ALERT_HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(alerts, f, indent=4)
    except Exception:
        pass

# Sidebar Setup
st.sidebar.header("System Controls")
input_mode = st.sidebar.radio(
    "Select Input Mode",
    ["📷 Live Browser Camera", "📹 Upload Video File", "📋 Alert Logs"]
)

alerts = load_alerts()
st.sidebar.markdown("---")
st.sidebar.metric(label="Total Alerts Logged", value=len(alerts))

# Public STUN Server for WebRTC Connection
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ---------------------------------------------------------
# 3. WEBRTC LIVE VIDEO PROCESSOR
# ---------------------------------------------------------
class FallDetectionVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.fall_detected = False
        self.status_text = "NORMAL"

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        try:
            # Process frame using MediaPipe + ML Model
            processed_img, is_fall, status = process_frame(img)
            self.fall_detected = is_fall
            self.status_text = status
        except Exception as e:
            processed_img = img
            cv2.putText(processed_img, f"Inference Error: {str(e)}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Overlay red alert border if fall is detected
        if self.fall_detected:
            h, w, _ = processed_img.shape
            cv2.rectangle(processed_img, (0, 0), (w, h), (0, 0, 255), 10)
            cv2.putText(processed_img, "🚨 ALERT: FALL DETECTED!", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        return frame.from_ndarray(processed_img, format="bgr24")


# =========================================================
# MODE 1: LIVE WEBRTC CAMERA STREAM WITH REAL-TIME ALARM
# =========================================================
if input_mode == "📷 Live Browser Camera":
    st.subheader("📷 Real-Time Live Webcam Detection")
    st.write("Click **START** below. The system continuously processes frames in real time, triggers visual warnings, and sounds an alarm when a fall occurs.")

    ctx = webrtc_streamer(
        key="fall-detection-live",
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=FallDetectionVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )

    status_container = st.empty()
    audio_container = st.empty()

    if ctx.video_processor:
        while ctx.state.playing:
            is_fall = ctx.video_processor.fall_detected
            status_text = ctx.video_processor.status_text

            if is_fall:
                status_container.error(f"🚨 ALERT: FALL DETECTED! ({status_text})")
                
                # HTML5 Audio Alarm Trigger
                audio_html = """
                
                    
                
                """
                audio_container.markdown(audio_html, unsafe_allow_html=True)
                log_alert(status_text)
            else:
                status_container.success(f"🟢 Monitoring Active — Status: {status_text}")
                audio_container.empty()

            time.sleep(0.2)

# =========================================================
# MODE 2: UPLOAD VIDEO FILE EVALUATION
# =========================================================
elif input_mode == "📹 Upload Video File":
    st.subheader("📹 Video File Evaluation")
    st.write("Upload a video file to analyze sequence frame-by-frame.")

    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 60:
            fps = 30.0

        if not cap.isOpened():
            st.error("Error opening uploaded video file.")
        else:
            st.success(f"Video loaded. Total Frames: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")

            if st.button("▶️ Start Video Analysis"):
                st_frame = st.empty()
                status_box = st.empty()
                audio_box = st.empty()
                progress_bar = st.progress(0)

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                current_frame = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    current_frame += 1

                    # Run inference pipeline
                    processed_frame, fall_detected, status_text = process_frame(frame)

                    # Convert BGR to RGB for Streamlit rendering
                    frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    st_frame.image(frame_rgb, use_container_width=True)

                    if fall_detected:
                        status_box.error(f"🚨 ALERT DETECTED: {status_text}")
                        audio_html = """
                        
                            
                        
                        """
                        audio_box.markdown(audio_html, unsafe_allow_html=True)
                        log_alert(status_text)
                    else:
                        status_box.info(f"🟢 Status: {status_text}")
                        audio_box.empty()

                    if total_frames > 0:
                        progress_bar.progress(min(current_frame / total_frames, 1.0))

                    time.sleep(1.0 / fps)

                cap.release()
                st.success("✅ Video evaluation completed.")

# =========================================================
# MODE 3: ALERT LOGS
# =========================================================
elif input_mode == "📋 Alert Logs":
    st.subheader("📋 Recorded Fall Event Logs")
    alerts = load_alerts()
    if not alerts:
        st.info("No recorded fall events logged yet.")
    else:
        for alert in reversed(alerts):
            st.write(f"⏱️ **Timestamp:** {alert.get('timestamp', 'N/A')} | 🚨 **Status:** {alert.get('status', 'FALL DETECTED')}")
