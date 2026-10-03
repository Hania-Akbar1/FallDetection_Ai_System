
from flask import Flask, jsonify, send_from_directory, Response

import subprocess
import threading
import sys
import os
import time
import json
from pathlib import Path


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DASHBOARD_DIR = PROJECT_ROOT / "DASHBOARD"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LIVE_FRAME_PATH = OUTPUTS_DIR / "live_frame.jpg"
ALERT_HISTORY_PATH = OUTPUTS_DIR / "alert_history.json"
STOP_REQUEST_PATH = OUTPUTS_DIR / "stop_monitoring.flag"


# =========================================================
# FLASK APP
# =========================================================

app = Flask(
    __name__,
    static_folder=DASHBOARD_DIR,
    static_url_path=""
)


# =========================================================
# BACKEND PROCESS
# =========================================================

process = None
process_lock = threading.Lock()


def stop_inference_process():

    global process

    if process is None:
        return

    if process.poll() is None:

        stop_requested = False

        try:
            STOP_REQUEST_PATH.touch()
            stop_requested = True
        except OSError as error:
            print("Could not request graceful stop:", error)

        try:
            process.wait(
                timeout=30 if stop_requested else 3
            )
        except subprocess.TimeoutExpired:
            print("Inference did not stop in time; terminating process.")
            process.terminate()

            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    process = None


# =========================================================
# ALERT HISTORY
# =========================================================

def load_alert_history():

    if not os.path.exists(
        ALERT_HISTORY_PATH
    ):
        return []

    try:

        with open(
            ALERT_HISTORY_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

            if isinstance(data, list):
                return data

    except Exception as error:

        print(
            "Alert history load error:",
            error
        )

    return []


# =========================================================
# GET PERSISTENT ALERT COUNT
# =========================================================

def get_persistent_alert_count():

    alerts = load_alert_history()

    return len(alerts)


# =========================================================
# GET LATEST ALERT
# =========================================================

def get_latest_alert():

    alerts = load_alert_history()

    if not alerts:
        return None

    return alerts[-1]


# =========================================================
# SYSTEM STATE
# =========================================================

system_state = {

    "running": False,

    "status": "OFFLINE",

    "detection": "SYSTEM READY",

    # IMPORTANT:
    # Initial count comes from persistent history
    "alert_count": get_persistent_alert_count(),

    "last_alert": None,

    "telegram": "Ready"
}


# =========================================================
# INITIAL LAST ALERT
# =========================================================

latest_alert = get_latest_alert()

if latest_alert:

    timestamp = latest_alert.get(
        "timestamp"
    )

    if timestamp:

        system_state["last_alert"] = timestamp


# =========================================================
# ALERT HISTORY API
# =========================================================

@app.route("/api/alerts")
def get_alerts():

    alerts = load_alert_history()

    # Latest alerts first
    alerts = list(reversed(alerts))

    return jsonify({

        "success": True,

        "alerts": alerts,

        "count": len(alerts)

    })


# =========================================================
# MONITOR REALTIME INFERENCE OUTPUT
# =========================================================

def monitor_process_output(proc):

    global system_state

    try:

        for raw_line in iter(
            proc.stdout.readline,
            ""
        ):

            if not raw_line:
                break

            line = raw_line.strip()

            print(
                "[FallDetection]",
                line
            )


            # =================================================
            # FALL DETECTED
            # =================================================

            if (
                "ALERT RESPONSE AGENT ACTIVATED"
                in line
            ):

                system_state["detection"] = (
                    "FALL DETECTED"
                )

                system_state["status"] = (
                    "ALERT"
                )

                # IMPORTANT:
                # Do NOT increment an in-memory counter.
                # Read the real persistent history instead.

                system_state["alert_count"] = (
                    get_persistent_alert_count()
                )

                latest = get_latest_alert()

                if latest:

                    system_state["last_alert"] = (
                        latest.get(
                            "timestamp"
                        )
                    )


            # =================================================
            # TELEGRAM ALERT
            # =================================================

            if (
                "TELEGRAM ALERT SENT"
                in line
            ):

                system_state["telegram"] = (
                    "Sent"
                )


            # =================================================
            # MONITORING STARTED
            # =================================================

            if "Starting" in line:

                system_state["status"] = (
                    "MONITORING"
                )


    except Exception as error:

        print(
            "Process monitor error:",
            error
        )


    finally:

        proc.wait()

        with process_lock:

            if process is proc:

                system_state["running"] = False

                system_state["status"] = (
                    "OFFLINE"
                )

                if (
                    system_state["detection"]
                    != "FALL DETECTED"
                ):

                    system_state["detection"] = (
                        "SYSTEM READY"
                    )


# =========================================================
# START FALL DETECTION
# =========================================================

@app.route(
    "/api/start",
    methods=["POST"]
)
def start_monitoring():

    global process

    with process_lock:


        # =================================================
        # ALREADY RUNNING
        # =================================================

        if process is not None:

            if process.poll() is None:

                return jsonify({

                    "success": True,

                    "message":
                        "Monitoring already running."

                })


        # =================================================
        # REMOVE OLD LIVE FRAME
        # =================================================

        try:
            STOP_REQUEST_PATH.unlink(missing_ok=True)
        except OSError as error:
            print("Could not clear old stop request:", error)

        try:

            if os.path.exists(
                LIVE_FRAME_PATH
            ):

                os.remove(
                    LIVE_FRAME_PATH
                )

        except Exception as error:

            print(
                "Could not remove old live frame:",
                error
            )


        # =================================================
        # REFRESH PERSISTENT COUNT
        # =================================================

        system_state["alert_count"] = (
            get_persistent_alert_count()
        )


        # =================================================
        # RESET CURRENT MONITORING STATE
        # =================================================

        system_state["running"] = True

        system_state["status"] = (
            "MONITORING"
        )

        system_state["detection"] = (
            "MONITORING"
        )

        system_state["telegram"] = (
            "Connected"
        )


        # =================================================
        # START REALTIME INFERENCE
        # =================================================

        try:

            process = subprocess.Popen(

                [

                    sys.executable,

                    "-u",

                    "-m",

                    "src.realtime_inference"

                ],

                cwd=PROJECT_ROOT,

                stdout=subprocess.PIPE,

                stderr=subprocess.STDOUT,

                text=True,

                encoding="utf-8",

                errors="replace",

                bufsize=1,

                creationflags=(
                    subprocess.CREATE_NEW_PROCESS_GROUP
                    if os.name == "nt"
                    else 0
                )

            )

        except Exception as error:

            system_state["running"] = False

            system_state["status"] = (
                "OFFLINE"
            )

            system_state["detection"] = (
                "SYSTEM READY"
            )

            return jsonify({

                "success": False,

                "message": str(error)

            }), 500


        # =================================================
        # MONITOR SUBPROCESS OUTPUT
        # =================================================

        threading.Thread(

            target=monitor_process_output,

            args=(process,),

            daemon=True

        ).start()


        return jsonify({

            "success": True,

            "message":
                "Fall detection started."

        })


# =========================================================
# STOP FALL DETECTION
# =========================================================

@app.route(
    "/api/stop",
    methods=["POST"]
)
def stop_monitoring():

    with process_lock:

        try:
            stop_inference_process()
        except Exception as error:
            print("Stop process error:", error)


        system_state["running"] = False

        system_state["status"] = (
            "OFFLINE"
        )

        system_state["detection"] = (
            "SYSTEM READY"
        )

        system_state["telegram"] = (
            "Ready"
        )


        # IMPORTANT:
        # Keep persistent alert count.
        system_state["alert_count"] = (
            get_persistent_alert_count()
        )


    return jsonify({

        "success": True,

        "message":
            "Monitoring stopped."

    })


# =========================================================
# SYSTEM STATUS
# =========================================================

@app.route("/api/status")
def get_status():

    # =====================================================
    # ALWAYS SYNC WITH JSON HISTORY
    # =====================================================

    system_state["alert_count"] = (
        get_persistent_alert_count()
    )


    latest = get_latest_alert()

    if latest:

        system_state["last_alert"] = (
            latest.get("timestamp")
        )


    return jsonify(
        system_state
    )


# =========================================================
# LIVE CAMERA STREAM
# =========================================================

def generate_camera_frames():

    while True:


        # =================================================
        # STOP BROWSER STREAM WHEN MONITORING STOPS
        # =================================================

        if not system_state["running"]:

            time.sleep(0.1)

            continue


        # =================================================
        # WAIT FOR REALTIME FRAME
        # =================================================

        if not os.path.exists(
            LIVE_FRAME_PATH
        ):

            time.sleep(0.05)

            continue


        try:

            with open(
                LIVE_FRAME_PATH,
                "rb"
            ) as image_file:

                frame_bytes = (
                    image_file.read()
                )


            if not frame_bytes:

                time.sleep(0.05)

                continue


            yield (

                b"--frame\r\n"

                b"Content-Type: image/jpeg\r\n\r\n"

                + frame_bytes

                + b"\r\n"

            )


        except Exception as error:

            print(
                "Camera stream error:",
                error
            )

            time.sleep(0.05)


        time.sleep(0.03)


# =========================================================
# VIDEO FEED ROUTE
# =========================================================

@app.route("/video_feed")
def video_feed():

    return Response(

        generate_camera_frames(),

        mimetype=(

            "multipart/x-mixed-replace; "

            "boundary=frame"

        )

    )


# =========================================================
# DASHBOARD
# =========================================================

@app.route("/")
def dashboard():

    return send_from_directory(

        DASHBOARD_DIR,

        "index.html"

    )


# =========================================================
# DASHBOARD STATIC FILES
# =========================================================

@app.route("/<path:path>")
def serve_dashboard_file(path):

    return send_from_directory(

        DASHBOARD_DIR,

        path

    )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    print()

    print("=" * 60)

    print(
        "       FallDetection.AI Dashboard"
    )

    print("=" * 60)

    print()

    print("Dashboard:")

    print(
        f"http://127.0.0.1:{port}"
    )

    print()

    print("Persistent Alerts:")

    print(
        get_persistent_alert_count()
    )

    print()

    print("Camera architecture:")

    print("Webcam")

    print("   ↓")

    print("realtime_inference.py")

    print("   ↓")

    print("live_frame.jpg")

    print("   ↓")

    print("Flask")

    print("   ↓")

    print("Browser Dashboard")

    print()

    print("=" * 60)

    print()

    try:
        app.run(
            host="0.0.0.0",
            port=port,
            debug=False,
            threaded=True
        )
    finally:
        with process_lock:
            stop_inference_process()
