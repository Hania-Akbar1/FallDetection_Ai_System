import tempfile
import cv2
import streamlit as st
import numpy as np
import av
from streamlit_webrtc import streamlit_webrtc_streamer, RTCConfiguration, WebRtcMode
from src.realtime_inference import process_frame, reset_pipeline_state

# Page Configuration
st.set_page_config(page_title="Fall Detection AI", layout="wide")
st.title("🎥 Fall Detection & Real-time Analysis System")

# STUN Servers required for WebRTC connections on Cloud servers
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]}
)

# Sidebar Mode Selection
mode = st.sidebar.radio("Select Input Source:", ["Upload Video File", "Live Webcam Stream"])

# -----------------------------------------------------------------------------
# MODE 1: UPLOADED VIDEO FILE ANALYSIS
# -----------------------------------------------------------------------------
if mode == "Upload Video File":
    st.subheader("📼 Video File Fall Detection")
    uploaded_file = st.file_uploader("Upload a video file (.mp4, .avi, .mov)", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        # 1. Save uploaded file to temporary disk space (OpenCV cannot open Streamlit BytesIO)
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        tfile.close()

        # Reset tracking buffer for clean video analysis
        reset_pipeline_state()

        cap = cv2.VideoCapture(tfile.name)
        st_frame = st.empty()
        status_box = st.empty()

        # Get original video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

        st.info("Processing video stream...")
        stop_button = st.button("Stop Video Processing")

        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret:
                break

            # 2. Resize frame to 640x480 to optimize CPU processing on Streamlit Cloud
            frame_resized = cv2.resize(frame, (640, 480))

            # 3. Process frame via inference model
            processed_img, is_fall, status_text = process_frame(frame_resized)

            # 4. Display Status Banner & Processed Video Frame
            if is_fall:
                status_box.error(f"⚠️ {status_text}")
            else:
                status_box.success(f"✅ {status_text}")

            # Convert BGR (OpenCV) to RGB (Streamlit Display)
            rgb_display = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
            st_frame.image(rgb_display, channels="RGB", use_container_width=True)

        cap.release()
        st.success("Video processing complete!")

# -----------------------------------------------------------------------------
# MODE 2: LIVE WEBCAM STREAM (WEBRTC)
# -----------------------------------------------------------------------------
elif mode == "Live Webcam Stream":
    st.subheader("📹 Live Camera Stream Analysis")
    st.caption("Grant camera permissions in your browser when prompted.")

    class VideoProcessor:
        def __init__(self):
            reset_pipeline_state()

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            # Convert WebRTC incoming frame (NDArray) to OpenCV BGR format
            img = frame.to_ndarray(format="bgr24")

            # Resize frame for real-time latency optimization
            img_resized = cv2.resize(img, (640, 480))

            # Process frame through inference engine
            processed_img, is_fall, status_text = process_frame(img_resized)

            # Overlay status text directly onto video stream frame
            text_color = (0, 0, 255) if is_fall else (0, 255, 0)
            cv2.putText(
                processed_img,
                status_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                text_color,
                2,
                cv2.LINE_AA
            )

            # Return processed frame back to browser WebRTC stream
            return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

    # Launch WebRTC Streamer
    webrtc_ctx = streamlit_webrtc_streamer(
        key="fall-detection-live",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
