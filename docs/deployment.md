# Deployment Preparation

## Current Local Deployment

Use Python 3.12.13 from the project root. In PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python app.py
```

Do not overwrite an existing `.env`. Set Telegram values locally if notifications are required. Open <http://127.0.0.1:5000> and start monitoring from the dashboard. Stop monitoring before stopping Flask with `Ctrl+C`.

## Render Preparation

No deployment has been performed. Configure a Python web service with:

**Build command**

```sh
pip install -r requirements.txt
```

**Start command**

```sh
gunicorn app:app --bind 0.0.0.0:$PORT --worker-class gthread --workers 1 --threads 8
```

The threaded worker is needed because `/video_feed` is a long-lived streaming response while dashboard status and alert requests must remain serviceable. Provide `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` as secret environment variables if Telegram notifications are enabled. `UR_DATASET_DIR` is only needed by the optional feature-extraction script and is not needed for live model inference. Python 3.12.13 is recorded in `.python-version`.

The current Flask server can serve the dashboard and its API through WSGI. However, the current inference subprocess is not cloud-camera-ready, so hosting the dashboard does not make remote fall monitoring operational. Do not present a hosted dashboard as a working remote camera detector.

## Important Camera Limitation

`src/realtime_inference.py` calls `cv2.VideoCapture(0)` in the Python process. The camera must be attached to the machine running that process; a Render server cannot access an examiner's physical webcam through the current pipeline. The dashboard's `/video_feed` displays frames produced by that local process; it does not request browser-camera access.

The current alarm also imports Windows-only `winsound`, so starting the inference subprocess on a Linux host such as Render will fail before camera capture. This behavior is intentionally unchanged in this cleanup task. Remote browser-camera support and cross-platform inference are a separate future phase.

## Storage and Secrets

`outputs/alert_history.json`, live frames, and demo videos are local runtime files and are Git-ignored. Render's filesystem is ephemeral by default, so alert history is not durable across redeploys or instance replacement. Persistent cloud storage is a future improvement, not part of this deployment preparation.

Keep `.env` out of Git. Use Render's environment settings for credentials, enable HTTPS, and rotate/revoke any Telegram credential that has been exposed. Never load an untrusted pickle model.
