import os
import sys
import time
import json
import signal
import threading
import subprocess
from pathlib import Path
from flask import Flask, jsonify, send_from_directory, Response


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DASHBOARD_DIR = PROJECT_ROOT / "DASHBOARD"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LIVE_FRAME_PATH = OUTPUTS_DIR / "live_frame.jpg"
ALERT_HISTORY_PATH = OUTPUTS_DIR / "alert_history.json"
STOP_REQUEST_PATH = OUTPUTS_DIR / "stop_monitoring.flag"

# Ensure output directories exist
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# FLASK APP
# =========================================================

app = Flask(
    __name__,
    static_folder=str(DASHBOARD_DIR),
    static_url_path=""
)


# =========================================================
# BACKEND PROCESS CONTROL
# =========================================================

process = None
process_lock = threading.Lock()


def stop_inference_process():
    global process

    if process is None:
        return

    if process.poll() is None:
        stop_requested = False

        # Attempt 1: Signal graceful shutdown via flag file
        try:
            STOP_REQUEST_PATH.touch(exist_ok=True)
            stop_requested = True
        except OSError as error:
            print("[FallDetection] Could not create stop request flag:", error)

        # Wait for graceful exit
        try:
            process.wait(timeout=10 if stop_requested else 3)
        except subprocess.TimeoutExpired:
            print("[FallDetection] Inference process did not exit gracefully; terminating...")
            
            # Attempt 2: Soft SIGINT / SIGTERM
            try:
                if os.name == "nt":
                    process.terminate()
                else:
                    process.send_signal(signal.SIGINT)
                process.wait(timeout=5)
            except (subprocess.TimeoutExpired, OSError):
                print("[FallDetection] Force killing process...")
                try:
                    process.kill()
                    process.wait(timeout=3)
                except Exception as err:
                    print("[FallDetection] Process kill error:", err)

    process = None


# =========================================================
# ALERT HISTORY
# =========================================================

def load_alert_history():
    if not ALERT_HISTORY_PATH.exists():
        return []

    try:
        with open(ALERT_HISTORY_PATH, "r", encoding="utf-8") as file:
            data = json.load(file)
            if isinstance(data, list):
                return data
    except Exception as error:
        print("[FallDetection] Alert history load error:", error)

    return []


def get_persistent_alert_count():
    alerts = load_alert_history()
    return len(alerts)


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
    "alert_count": get_persistent_alert_count(),
    "last_alert": None,
    "telegram": "Ready"
}

latest_alert = get_latest_alert()
if latest_alert and latest_alert.get("timestamp"):
    system_state["last_alert"] = latest_alert.get("timestamp")


# =========================================================
# ALERT HISTORY API
# =========================================================

@app.route("/api/alerts")
def get_alerts():
    alerts = load_alert_history()
    alerts_reversed = list(reversed(alerts))
    return jsonify({
        "success": True,
        "alerts": alerts_reversed,
        "count": len(alerts)
    })


# =========================================================
# MONITOR REALTIME INFERENCE OUTPUT
# =========================================================

def monitor_process_output(proc):
    global system_state

    try:
        for raw_line in iter(proc.stdout.readline, ""):
            if not raw_line:
                break

            line = raw_line.strip()
            print("[FallDetection]", line)

            # Detect alert trigger
            if "ALERT RESPONSE AGENT ACTIVATED" in line:
                system_state["detection"] = "FALL DETECTED"
                system_state["status"] = "ALERT"
                system_state["alert_count"] = get_persistent_alert_count()
                
                latest = get_latest_alert()
                if latest:
                    system_state["last_alert"] = latest.get("timestamp")

            # Detect telegram status
            if "TELEGRAM ALERT SENT" in line:
                system_state["telegram"] = "Sent"

            # Detect monitoring start
            if "Starting" in line or "Inference loop started" in line:
                system_state["status"] = "MONITORING"

    except Exception as error:
        print("[FallDetection] Process monitor error:", error)

    finally:
        proc.wait()
        with process_lock:
            if process is proc:
                system_state["running"] = False
                system_state["status"] = "OFFLINE"
                if system_state["detection"] != "FALL DETECTED":
                    system_state["detection"] = "SYSTEM READY"


# =========================================================
# START FALL DETECTION
# =========================================================

@app.route("/api/start", methods=["POST"])
def start_monitoring():
    global process

    with process_lock:
        # Check if already running
        if process is not None and process.poll() is None:
            return jsonify({
                "success": True,
                "message": "Monitoring is already running."
            })

        # Cleanup stop flags & previous live frame
        try:
            STOP_REQUEST_PATH.unlink(missing_ok=True)
        except OSError as error:
            print("[FallDetection] Could not clear stop request flag:", error)

        try:
            if LIVE_FRAME_PATH.exists():
                LIVE_FRAME_PATH.unlink(missing_ok=True)
        except OSError as error:
            print("[FallDetection] Could not remove old live frame:", error)

        # Refresh state
        system_state["alert_count"] = get_persistent_alert_count()
        system_state["running"] = True
        system_state["status"] = "MONITORING"
        system_state["detection"] = "MONITORING"
        system_state["telegram"] = "Connected"

        # Start realtime inference subprocess
        try:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "src.realtime_inference"
                ],
                cwd=str(PROJECT_ROOT),
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
            system_state["status"] = "OFFLINE"
            system_state["detection"] = "SYSTEM READY"
            return jsonify({
                "success": False,
                "message": f"Failed to start process: {str(error)}"
            }), 500

        # Start thread to monitor output logs
        threading.Thread(
            target=monitor_process_output,
            args=(process,),
            daemon=True
        ).start()

        return jsonify({
            "success": True,
            "message": "Fall detection started."
        })


# =========================================================
# STOP FALL DETECTION
# =========================================================

@app.route("/api/stop", methods=["POST"])
def stop_monitoring():
    with process_lock:
        try:
            stop_inference_process()
        except Exception as error:
            print("[FallDetection] Stop process error:", error)

        system_state["running"] = False
        system_state["status"] = "OFFLINE"
        system_state["detection"] = "SYSTEM READY"
        system_state["telegram"] = "Ready"
        system_state["alert_count"] = get_persistent_alert_count()

    return jsonify({
        "success": True,
        "message": "Monitoring stopped."
    })


# =========================================================
# SYSTEM STATUS API
# =========================================================

@app.route("/api/status")
def get_status():
    system_state["alert_count"] = get_persistent_alert_count()
    latest = get_latest_alert()
    if latest:
        system_state["last_alert"] = latest.get("timestamp")

    return jsonify(system_state)


# =========================================================
# LIVE CAMERA STREAM
# =========================================================

def generate_camera_frames():
    while True:
        if not system_state["running"]:
            time.sleep(0.1)
            continue

        if not LIVE_FRAME_PATH.exists():
            time.sleep(0.05)
            continue

        try:
            with open(LIVE_FRAME_PATH, "rb") as image_file:
                frame_bytes = image_file.read()

            if not frame_bytes:
                time.sleep(0.05)
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )

        except (PermissionError, FileNotFoundError, OSError):
            # Safe retry if file is being written concurrently
            time.sleep(0.03)
            continue
        except Exception as error:
            print("[FallDetection] Camera stream error:", error)
            time.sleep(0.05)

        time.sleep(0.03)


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_camera_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# =========================================================
# DASHBOARD SERVING
# =========================================================

@app.route("/")
def dashboard():
    return send_from_directory(DASHBOARD_DIR, "index.html")


@app.route("/<path:path>")
def serve_dashboard_file(path):
    return send_from_directory(DASHBOARD_DIR, path)


# =========================================================
# MAIN ENTRYPOINT
# =========================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    print("\n" + "=" * 60)
    print("         FallDetection.AI Dashboard")
    print("=" * 60)
    print(f"Dashboard URL:      http://127.0.0.1:{port}")
    print(f"Persistent Alerts:  {get_persistent_alert_count()}")
    print("=" * 60 + "\n")

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
