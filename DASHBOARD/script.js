
// ============================================================
// FallDetection.AI - Dashboard JavaScript
// ============================================================


// ============================================================
// DOM ELEMENTS
// ============================================================

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");

const systemStatus = document.getElementById("systemStatus");
const cameraStatus = document.getElementById("cameraStatus");

const cameraPlaceholder = document.getElementById("cameraPlaceholder");
const cameraFeed = document.getElementById("cameraFeed");
const monitoringOverlay = document.getElementById("monitoringOverlay");

const detectionStatus = document.getElementById("detectionStatus");
const statusIcon = document.getElementById("statusIcon");
const detectionText = document.getElementById("detectionText");
const detectionDescription = document.getElementById("detectionDescription");

const alertCount = document.getElementById("alertCount");
const lastAlert = document.getElementById("lastAlert");
const telegramStatus = document.getElementById("telegramStatus");

const historyCount = document.getElementById("historyCount");
const alertHistory = document.getElementById("alertHistory");
const currentDateTime = document.getElementById("currentDateTime");


function updateClock() {

    if (!currentDateTime) {
        return;
    }

    const now = new Date();

    currentDateTime.dateTime = now.toISOString();
    currentDateTime.textContent = new Intl.DateTimeFormat(undefined, {
        month: "short",
        day: "numeric",
        year: "numeric",
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
        hour12: false
    }).format(now);

}


function updateStatusIndicators(isOnline, isAlert, isCameraActive) {

    const statusBadge = systemStatus?.closest(".system-status");

    if (statusBadge) {
        statusBadge.classList.toggle("is-online", isOnline);
        statusBadge.classList.toggle("is-alert", isAlert);
    }

    if (cameraStatus) {
        cameraStatus.classList.toggle("is-active", isCameraActive);
    }

}


// ============================================================
// START MONITORING
// ============================================================

async function startMonitoring() {

    try {

        const response = await fetch("/api/start", {
            method: "POST"
        });

        const data = await response.json();

        if (data.success) {

            updateRunningUI();

        } else {

            alert(data.message || "Failed to start monitoring.");

        }

    } catch (error) {

        console.error("Start error:", error);

        alert("Could not connect to the monitoring server.");

    }

}


// ============================================================
// STOP MONITORING
// ============================================================

async function stopMonitoring() {

    try {

        const response = await fetch("/api/stop", {
            method: "POST"
        });

        const data = await response.json();

        if (data.success) {

            updateStoppedUI();

        } else {

            alert(data.message || "Failed to stop monitoring.");

        }

    } catch (error) {

        console.error("Stop error:", error);

        alert("Could not connect to the monitoring server.");

    }

}


// ============================================================
// RUNNING UI
// ============================================================

function updateRunningUI() {

    updateStatusIndicators(true, false, true);

    if (startBtn) {
        startBtn.disabled = true;
    }

    if (stopBtn) {
        stopBtn.disabled = false;
    }

    if (cameraPlaceholder) {
        cameraPlaceholder.style.display = "none";
    }

    if (cameraFeed) {
        cameraFeed.style.display = "block";
    }

    if (monitoringOverlay) {
        monitoringOverlay.style.display = "block";
    }

    if (cameraStatus) {
        cameraStatus.textContent = "Camera Active";
    }

    if (systemStatus) {
        systemStatus.textContent = "MONITORING";
    }

}


// ============================================================
// STOPPED UI
// ============================================================

function updateStoppedUI() {

    updateStatusIndicators(false, false, false);

    if (startBtn) {
        startBtn.disabled = false;
    }

    if (stopBtn) {
        stopBtn.disabled = true;
    }

    if (cameraPlaceholder) {
        cameraPlaceholder.style.display = "flex";
    }

    if (cameraFeed) {
        cameraFeed.style.display = "none";
    }

    if (monitoringOverlay) {
        monitoringOverlay.style.display = "none";
    }

    if (cameraStatus) {
        cameraStatus.textContent = "Camera Offline";
    }

    if (systemStatus) {
        systemStatus.textContent = "OFFLINE";
    }

}


// ============================================================
// UPDATE DASHBOARD STATUS
// ============================================================

async function updateStatus() {

    try {

        const response = await fetch("/api/status");

        const data = await response.json();

        if (!data) {
            return;
        }


        // ----------------------------------------------------
        // System Status
        // ----------------------------------------------------

        if (systemStatus) {

            systemStatus.textContent =
                data.status || "OFFLINE";

        }


        // ----------------------------------------------------
        // Alert Count
        // ----------------------------------------------------

        if (alertCount) {

            alertCount.textContent =
                data.alert_count ?? 0;

        }


        // ----------------------------------------------------
        // Last Alert
        // ----------------------------------------------------

        if (lastAlert) {

            lastAlert.textContent =
                data.last_alert || "—";

        }


        // ----------------------------------------------------
        // Telegram Status
        // ----------------------------------------------------

        if (telegramStatus) {

            telegramStatus.textContent =
                data.telegram || "Ready";

        }


        // ----------------------------------------------------
        // Detection Status
        // ----------------------------------------------------

        updateDetectionUI(data);


        // ----------------------------------------------------
        // Camera UI
        // ----------------------------------------------------

        if (data.running) {

            updateRunningUI();

        } else {

            updateStoppedUI();

        }

        updateStatusIndicators(
            Boolean(data.running || data.status !== "OFFLINE"),
            data.status === "ALERT",
            Boolean(data.running)
        );

    } catch (error) {

        console.error("Status update error:", error);

    }

}


// ============================================================
// DETECTION UI
// ============================================================

function updateDetectionUI(data) {

    if (!detectionStatus ||
        !statusIcon ||
        !detectionText ||
        !detectionDescription) {

        return;

    }


    const detection =
        data.detection || "SYSTEM READY";


    // --------------------------------------------------------
    // FALL DETECTED
    // --------------------------------------------------------

    if (
        detection.includes("FALL") ||
        data.status === "ALERT"
    ) {

        detectionStatus.className =
            "detection-status danger";

        statusIcon.textContent = "!";

        detectionText.textContent =
            "FALL DETECTED";

        detectionDescription.textContent =
            "Alert Response Agent activated. Please check the person immediately.";

        return;
    }


    // --------------------------------------------------------
    // MONITORING
    // --------------------------------------------------------

    if (data.running) {

        detectionStatus.className =
            "detection-status normal";

        statusIcon.textContent = "✓";

        detectionText.textContent =
            detection;

        detectionDescription.textContent =
            "System is actively monitoring the camera.";

        return;
    }


    // --------------------------------------------------------
    // SYSTEM READY
    // --------------------------------------------------------

    detectionStatus.className =
        "detection-status normal";

    statusIcon.textContent = "✓";

    detectionText.textContent =
        "SYSTEM READY";

    detectionDescription.textContent =
        "Waiting for monitoring to start.";

}


// ============================================================
// LOAD ALERT HISTORY
// ============================================================

async function loadAlertHistory() {

    try {

        const response = await fetch("/api/alerts");

        const data = await response.json();

        if (!data || !data.success) {

            console.error("Could not load alert history.");

            return;

        }


        const alerts =
            Array.isArray(data.alerts)
                ? data.alerts
                : [];


        renderAlertHistory(alerts);


    } catch (error) {

        console.error(
            "Alert history error:",
            error
        );

    }

}


// ============================================================
// RENDER ALERT HISTORY
// ============================================================

function renderAlertHistory(alerts) {

    if (!alertHistory) {

        console.error(
            'Element with id="alertHistory" was not found.'
        );

        return;

    }


    // --------------------------------------------------------
    // IMPORTANT:
    // Always calculate count from actual history array.
    // This fixes 4 Alerts vs 5 Alerts mismatch.
    // --------------------------------------------------------

    if (historyCount) {

        historyCount.textContent =
            `${alerts.length} Alerts`;

    }


    // --------------------------------------------------------
    // EMPTY HISTORY
    // --------------------------------------------------------

    if (alerts.length === 0) {

        alertHistory.innerHTML = `

            <div class="empty-history">

                <span>✓</span>

                <p>
                    No alerts detected
                </p>

            </div>

        `;

        return;

    }


    // --------------------------------------------------------
    // ALERT CARDS
    // --------------------------------------------------------

    alertHistory.innerHTML = alerts.map(alert => {

        const incidentId =
            alert.incident_id || "N/A";

        const event =
            alert.event || "FALL DETECTED";

        const timestamp =
            alert.timestamp || "N/A";

        const confidence =
            alert.confidence !== null &&
            alert.confidence !== undefined
                ? `${alert.confidence}%`
                : "N/A";

        const severity =
            alert.severity || "N/A";

        const telegram =
            alert.telegram_status || "N/A";


        // Severity class
        let severityClass = "severity-low";

        if (severity.toUpperCase() === "HIGH") {

            severityClass = "severity-high";

        } else if (
            severity.toUpperCase() === "MEDIUM"
        ) {

            severityClass = "severity-medium";

        }


        // Telegram class
        const telegramClass =
            telegram.toUpperCase() === "SENT"
                ? "telegram-sent"
                : "telegram-failed";


        return `

            <div class="history-item">

                <div class="history-main">

                    <div class="history-title-row">

                        <strong>
                            ${escapeHtml(event)}
                        </strong>

                        <span class="incident-id">
                            ${escapeHtml(incidentId)}
                        </span>

                    </div>


                    <div class="history-details">

                        <span>
                            ${escapeHtml(timestamp)}
                        </span>

                        <span>
                            Confidence:
                            ${escapeHtml(confidence)}
                        </span>

                        <span class="${severityClass}">
                            ${escapeHtml(severity)}
                        </span>

                        <span class="${telegramClass}">
                            Telegram:
                            ${escapeHtml(telegram)}
                        </span>

                    </div>

                </div>

            </div>

        `;

    }).join("");

}


// ============================================================
// HTML ESCAPE
// ============================================================

function escapeHtml(value) {

    return String(value)

        .replace(/&/g, "&amp;")

        .replace(/</g, "&lt;")

        .replace(/>/g, "&gt;")

        .replace(/"/g, "&quot;")

        .replace(/'/g, "&#039;");

}


// ============================================================
// INITIALIZE DASHBOARD
// ============================================================

document.addEventListener(
    "DOMContentLoaded",
    async function () {

        console.log(
            "FallDetection.AI Dashboard Loaded"
        );

        updateClock();

        setInterval(
            updateClock,
            1000
        );


        // Initial UI
        updateStoppedUI();


        // Load current system status
        await updateStatus();


        // Load persistent alert history
        await loadAlertHistory();


        // ----------------------------------------------------
        // STATUS REFRESH
        // ----------------------------------------------------

        setInterval(
            updateStatus,
            1000
        );


        // ----------------------------------------------------
        // ALERT HISTORY REFRESH
        // ----------------------------------------------------

        setInterval(
            loadAlertHistory,
            2000
        );

    }
);

