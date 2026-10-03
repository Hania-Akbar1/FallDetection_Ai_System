# FallDetection.AI

FallDetection.AI is a real-time, computer-vision-based fall detection and alert system. It processes camera frames with MediaPipe Pose, classifies temporal landmark sequences with a trained Random Forest model, and presents monitoring state and incidents in a Flask dashboard.

## Core Features

- Live fall detection from a locally attached camera
- MediaPipe Pose extraction of 33 body landmarks
- A 30-frame temporal buffer for Random Forest classification
- Existing motion-based fall checks and recovery latch
- Flask dashboard with live frame, status, and alert history
- Alert Response Agent with severity and confidence reporting
- Optional Telegram notifications configured through environment variables
- Persistent local alert history in `outputs/alert_history.json`
- Unique MP4 for each monitoring session and each fall incident

## System Architecture

```text
Camera
	↓
MediaPipe Pose
	↓
33 Pose Landmarks
	↓
30-Frame Temporal Buffer
	↓
Random Forest + existing motion checks
	↓
Fall Detection
	↓
Alert Response Agent
	↓
Telegram + Flask Dashboard
```

## Project Structure

```text
FallDetection_Ai_System/
├── app.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
├── .python-version
├── src/
│   ├── __init__.py
│   ├── alert_agent.py
│   ├── evaluate_model.py
│   ├── extract_features.py
│   ├── realtime_inference.py
│   ├── train_from_features.py
│   └── vision_core.py
├── models/
│   └── fall_detection_model.pkl
├── data/
│   └── urfd_extracted_features.csv  (local training data; Git-ignored)
├── DASHBOARD/
│   ├── index.html
│   ├── script.js
│   └── style.css
├── outputs/
│   ├── .gitkeep
│   └── README.md
└── docs/
		└── deployment.md
```

The uppercase `DASHBOARD/` directory is retained from the existing project and is referenced with its exact case for case-sensitive hosts. `vision_core.py` is a standalone inference prototype; the Flask application starts `realtime_inference.py`. Runtime outputs and the large feature CSV are kept locally but excluded from Git. The trained model stays in the repository because inference loads it at runtime.

## Technology Stack

- Python 3.12
- Flask and Gunicorn (WSGI deployment)
- OpenCV and MediaPipe Pose
- NumPy, scikit-learn, and joblib
- pandas for feature training/evaluation
- Matplotlib and Seaborn for evaluation output
- Requests and python-dotenv for optional Telegram configuration

## Installation

Use Python 3.12.13, as verified with the current model and MediaPipe environment. From PowerShell:

```powershell
git clone <your-repository-url>
cd FallDetection_Ai_System
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

If `.env` already exists, do not overwrite it. Edit it locally to configure credentials; never commit it.

## Environment Variables

`src/alert_agent.py` reads `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` from the process environment or `.env`. Notifications are unavailable until both are configured; detection and local alert-history recording continue. `.env.example` contains placeholders only.

For optional feature extraction, set `UR_DATASET_DIR` to the directory containing `adl_files/` and `fall_files/`. Its default is `data/ur_dataset`. The source UR dataset is not bundled. The extracted feature CSV is used by training/evaluation, not live inference.

## Running Locally

From the project root with the virtual environment active:

```powershell
python app.py
```

Open <http://127.0.0.1:5000>, then select **Start Monitoring** in the dashboard. The Flask app starts the existing inference process, which uses the camera attached to the machine running Python. A single timestamped MP4 is written directly under `outputs/`; fall detection still creates dashboard alerts but does not save separate fall-event clips. Select **Stop Monitoring** and allow shutdown to finish so the MP4 is finalized and playable before closing the terminal.

## Dashboard

The browser dashboard is served by Flask at `/`. Its JavaScript uses same-origin relative requests, so local and single-service deployments use the same API URLs. The interface does not access a visitor's browser camera.

## Testing

1. Start the application and open the dashboard.
2. Start monitoring and confirm the camera feed and monitoring status.
3. Use a safe prerecorded clip or a controlled, non-hazardous demonstration; do not attempt an actual fall.
4. Confirm the detection status and incident entry in the dashboard.
5. If Telegram credentials are configured, verify receipt of the notification.
6. Confirm the incident is recorded in `outputs/alert_history.json`.

## API Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/` | Serve the dashboard |
| POST | `/api/start` | Start the local inference subprocess |
| POST | `/api/stop` | Stop the inference subprocess |
| GET | `/api/status` | Return monitoring and latest-alert status |
| GET | `/api/alerts` | Return persistent alert history |
| GET | `/video_feed` | Stream processed frames as multipart MJPEG |

## Model

Inference loads `models/fall_detection_model.pkl` with joblib. It is a trusted local pickle artifact; do not load a model from an untrusted source. Training and evaluation scripts use `data/urfd_extracted_features.csv` and may regenerate model/evaluation artifacts.

## Security

- Keep Telegram credentials in `.env` locally or deployment environment settings.
- `.env` is Git-ignored; never commit credentials or paste them into source files.
- Rotate/revoke credentials if they have been exposed or committed.
- Use HTTPS and a production WSGI server for public hosting.
- Treat confidence as the model's reported score, not a medically calibrated probability.

## Deployment Readiness

The Flask application is importable by a WSGI server, uses the deployment-provided `PORT`, and serves the dashboard using portable project-relative paths. Deployment preparation does not add cloud camera inference: the current OpenCV pipeline can only access the machine where Python runs. See the [product requirements](docs/PRD.md) and [deployment guide](docs/deployment.md) before hosting.

## Limitations

- Fall detection can produce false positives and false negatives.
- Model confidence is not necessarily calibrated.
- Current monitoring requires a locally attached camera and the existing host-side inference process.
- A cloud host such as Render cannot access an examiner's physical webcam through this pipeline.
- This system is not a medical device and does not provide a diagnosis.

## Future Improvements

- Browser-camera support for remote examiner testing
- Persistent cloud alert storage
- Better model calibration
- Additional datasets
- Production monitoring
