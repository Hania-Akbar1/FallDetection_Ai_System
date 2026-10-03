import os
import tempfile
import cv2
import streamlit as st
import numpy as np

# -----------------------------------------------------------------------------
# 1. ROBUST & ENVIRONMENT-AGNOSTIC IMPORTS
# -----------------------------------------------------------------------------
# Handle streamlit-webrtc API variations across versions
try:
    from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode
except ImportError:
    try:
        from streamlit_webrtc import streamlit_webrtc_streamer as webrtc_streamer, RTCConfiguration, WebRtcMode
    except ImportError:
        webrtc_streamer = None

import av

# Import your custom fall detection modules
from src.realtime_inference import process_frame, reset_pipeline_state

# -----------------------------------------------------------------------------
# 2. APP CONFIGURATION & STUN SETTINGS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Fall Detection AI",
    page_icon="🎥",
    layout="wide"
)

st.title("🎥 Fall Detection & Real-time Video Analysis System")

# Primary and fallback STUN servers for robust NAT traversal across different networks
RTC_CONFIG = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302", "stun:stun3.l.google.com:19302"]},
        ]
    }
)

# -----------------------------------------------------------------------------
# 3. NAVIGATION / INPUT SOURCE SELECTION
# -----------------------------------------------------------------------------
mode = st.sidebar.radio(
    "Select Input Source:",
    ["Upload Video File", "Live Webcam Stream (WebRTC)"]
)

# -----------------------------------------------------------------------------
# MODE 1: UPLOADED VIDEO FILE PROCESSING
# -----------------------------------------------------------------------------
if mode == "Upload Video File":
    st.subheader("📼 Video File Fall Detection")
    uploaded_file = st.file_uploader(
        "Upload a video file (.mp4, .avi, .mov)", 
        type=["mp4", "avi", "mov"]
    )

    if uploaded_file is not None:
        # Cross-platform safe temporary file handling
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        try:
            tfile.write(uploaded_file.read())
            tfile.close()

            reset_pipeline_state()

            cap = cv2.VideoCapture(tfile.name)
            if not cap.isOpened():
                st.error("Error opening video file. Please check the codec or format.")
            else:
                st_frame = st.empty()
                status_box = st.empty()
                stop_button = st.button("Stop Processing")

                st.info("Processing video file...")

                while cap.isOpened() and not stop_button:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Resize to standard resolution (640x480) for low latency
                    frame_resized = cv2.resize(frame, (640, 480))

                    # Model Inference
                    processed_img, is_fall, status_text = process_frame(frame_resized)

                    # Update Status Banner
                    if is_fall:
                        status_box.error(f"⚠️ ALERT: {status_text}")
                    else:
                        status_box.success(f"✅ STATUS: {status_text}")

                    # OpenCV BGR -> Streamlit RGB
                    rgb_display = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                    st_frame.image(rgb_display, channels="RGB", use_container_width=True)

                cap.release()
                st.success("Video processing complete.")

        finally:
            # Safe cleanup for Windows & Linux
            if os.path.exists(tfile.name):
                os.unlink(tfile.name)

# -----------------------------------------------------------------------------
# MODE 2: LIVE WEBCAM STREAM (WEBRTC)
# -----------------------------------------------------------------------------
elif mode == "Live Webcam Stream (WebRTC)":
    st.subheader("📹 Live Camera Stream Analysis")

    if webrtc_streamer is None:
        st.error(
            "The `streamlit-webrtc` library is not installed or failed to load. "
            "Please verify your `requirements.txt` and `packages.txt` dependencies."
        )
    else:
        st.caption("Ensure camera access is allowed in your web browser.")

        class VideoProcessor:
            def __init__(self):
                reset_pipeline_state()

            def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                # 1. Convert WebRTC frame -> NumPy BGR array
                img = frame.to_ndarray(format="bgr24")

                # 2. Resize frame for consistent performance across cloud/local GPUs/CPUs
                img_resized = cv2.resize(img, (640, 480))

                # 3. Model Inference
                processed_img, is_fall, status_text = process_frame(img_resized)

                # 4. Draw HUD/Status Overlay onto frame directly
                color = (0, 0, 255) if is_fall else (0, 255, 0)
                cv2.putText(
                    processed_img,
                    f"Status: {status_text}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                    cv2.LINE_AA
                )

                # 5. Convert NumPy array -> WebRTC VideoFrame
                return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

        # Launch WebRTC Streamer
        webrtc_streamer(
            key="fall-detection-live-stream",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIG,
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
