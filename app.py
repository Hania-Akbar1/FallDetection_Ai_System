import os
import tempfile
import cv2
import pandas as pd
import streamlit as st
import numpy as np
import av

# Streamlit WebRTC compatibility
try:
    from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode
except ImportError:
    from streamlit_webrtc import streamlit_webrtc_streamer as webrtc_streamer, RTCConfiguration, WebRtcMode

from src.realtime_inference import (
    process_frame,
    reset_pipeline_state,
    get_fall_history
)

# Page Setup
st.set_page_config(
    page_title="Fall Detection AI System",
    page_icon="🚨",
    layout="wide"
)

st.title("🚨 Workplace Safety - AI Fall Detection & Monitoring System")

# WebRTC STUN Configuration
RTC_CONFIG = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302", "stun:stun3.l.google.com:19302"]},
        ]
    }
)

# HTML5 Browser Audio Alarm
AUDIO_ALARM_HTML = """

    

"""

# Sidebar Navigation & System Controls
st.sidebar.title("⚙️ System Controls")
mode = st.sidebar.radio(
    "Select Input Mode:",
    ["📹 Live Webcam Stream", "📼 Upload Video File"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("📋 Log Management")

if st.sidebar.button("🗑️ Reset & Clear History Logs"):
    reset_pipeline_state()
    st.sidebar.success("Logs and system history cleared!")

# Main Video Layout
col1, col2 = st.columns([2, 1])

with col1:
    # -------------------------------------------------------------------------
    # MODE 1: LIVE WEBCAM STREAMING
    # -------------------------------------------------------------------------
    if mode == "📹 Live Webcam Stream":
        st.subheader("📹 Real-Time Live Camera Feed")

        class VideoProcessor:
            def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                img = frame.to_ndarray(format="bgr24")
                img_resized = cv2.resize(img, (640, 480))

                processed_img, is_fall, status_text = process_frame(img_resized, source_mode="Live Webcam")

                # Visual Overlay
                if is_fall:
                    cv2.rectangle(processed_img, (0, 0), (640, 60), (0, 0, 255), -1)
                    cv2.putText(
                        processed_img,
                        "CRITICAL: FALL DETECTED!",
                        (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (255, 255, 255),
                        3,
                        cv2.LINE_AA
                    )
                else:
                    cv2.rectangle(processed_img, (0, 0), (640, 45), (0, 180, 0), -1)
                    cv2.putText(
                        processed_img,
                        f"STATUS: {status_text}",
                        (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA
                    )

                return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

        webrtc_streamer(
            key="fall-detection-live",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIG,
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    # -------------------------------------------------------------------------
    # MODE 2: RECORDED VIDEO FILE ANALYSIS
    # -------------------------------------------------------------------------
    elif mode == "📼 Upload Video File":
        st.subheader("📼 Video File Batch Analysis")
        uploaded_file = st.file_uploader("Upload test video (.mp4, .avi, .mov)", type=["mp4", "avi", "mov"])

        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            try:
                tfile.write(uploaded_file.read())
                tfile.close()

                cap = cv2.VideoCapture(tfile.name)
                st_frame = st.empty()
                status_banner = st.empty()
                audio_placeholder = st.empty()

                stop_processing = st.button("Stop Analysis")

                while cap.isOpened() and not stop_processing:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_resized = cv2.resize(frame, (640, 480))
                    processed_img, is_fall, status_text = process_frame(frame_resized, source_mode="Uploaded Video")

                    if is_fall:
                        status_banner.error("🚨 EMERGENCY ALERT: FALL DETECTED IN VIDEO FEED!")
                        audio_placeholder.markdown(AUDIO_ALARM_HTML, unsafe_allow_html=True)
                    else:
                        status_banner.success(f"✅ SYSTEM STATUS: {status_text}")
                        audio_placeholder.empty()

                    rgb_display = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                    st_frame.image(rgb_display, channels="RGB", use_container_width=True)

                cap.release()
                st.info("Video processing complete.")

            finally:
                if os.path.exists(tfile.name):
                    os.unlink(tfile.name)

# -----------------------------------------------------------------------------
# FALL HISTORY & LOG MONITORING SECTION
# -----------------------------------------------------------------------------
with col2:
    st.subheader("📊 Detection Logs & History")
    
    logs = get_fall_history()
    
    if logs:
        df_logs = pd.DataFrame(logs)
        st.dataframe(df_logs, use_container_width=True)
        
        # Download log option
        csv_data = df_logs.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Logs (CSV)",
            data=csv_data,
            file_name="fall_detection_logs.csv",
            mime="text/csv"
        )
    else:
        st.info("No fall events recorded in current session.")
