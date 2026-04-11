"""
G_One_Sync AI — Clinical Monitoring Dashboard
=================================================
FastAPI-served real-time dashboard for ICU deterioration monitoring.
Provides WebSocket updates, REST APIs for metrics, and a premium
dark-themed clinical UI.

Routes:
    GET  /                  → Dashboard HTML
    GET  /api/metrics       → Current system + model metrics
    GET  /api/alerts        → Recent alerts
    GET  /api/drift         → Drift report
    GET  /api/patients      → Patient risk overview
    WS   /ws/live           → Real-time WebSocket feed
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from src.monitoring.metrics_collector import MetricsCollector
from src.monitoring.drift_detector import DriftDetector
from src.monitoring.alerting import AlertingEngine


# ── App Setup ────────────────────────────────────────────────────────

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="G_One_Sync AI – Clinical Dashboard",
    version="1.0.0",
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Global state
metrics_collector = MetricsCollector(port=9090)
alerting_engine = AlertingEngine()
drift_detector = DriftDetector()

# WebSocket connections
ws_connections: list[WebSocket] = []

# Simulated patient state (for demo; production would use a database)
_patient_state: dict[int, dict] = {}


def _generate_demo_patients(n: int = 20) -> dict[int, dict]:
    """Generate simulated patient data for dashboard demo."""
    np.random.seed(int(time.time()) % 1000)
    patients = {}
    for i in range(1, n + 1):
        prob = np.clip(np.random.beta(2, 8), 0.02, 0.98)
        if prob >= 0.75:
            risk = "CRITICAL"
        elif prob >= 0.50:
            risk = "HIGH"
        elif prob >= 0.25:
            risk = "MODERATE"
        else:
            risk = "LOW"

        patients[i] = {
            "patient_id": i,
            "bed": f"ICU-{chr(65 + i // 10)}{i % 10}",
            "probability": round(float(prob), 4),
            "risk_level": risk,
            "heart_rate": int(60 + np.random.normal(0, 15)),
            "spo2": round(float(np.clip(97 + np.random.normal(0, 3), 85, 100)), 1),
            "systolic_bp": int(120 + np.random.normal(0, 20)),
            "respiratory_rate": int(16 + np.random.normal(0, 4)),
            "lactate": round(float(np.clip(1.0 + np.random.exponential(0.8), 0.5, 8)), 1),
            "last_updated": datetime.now().isoformat(),
            "trend": np.random.choice(["↑", "↓", "→"], p=[0.3, 0.2, 0.5]),
        }
    return patients


_patient_state = _generate_demo_patients(30)


# ── REST Endpoints ───────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the main dashboard page."""
    index_path = STATIC_DIR / "index.html"
    return FileResponse(str(index_path))


@app.get("/api/metrics")
async def get_metrics():
    """System and model metrics snapshot."""
    import torch

    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "name": torch.cuda.get_device_name(0),
            "memory_used_gb": round(torch.cuda.memory_allocated(0) / 1e9, 2),
            "memory_total_gb": round(torch.cuda.get_device_properties(0).total_mem / 1e9, 2),
            "utilization_pct": round(
                torch.cuda.memory_allocated(0) / torch.cuda.get_device_properties(0).total_mem * 100, 1
            ),
        }

    # Load training results if available
    summary_path = Path("models/artifacts/training_summary.json")
    model_perf = {}
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        for model_name in ["xgboost", "bilstm", "transformer", "ensemble"]:
            if model_name in summary and "test" in summary[model_name]:
                model_perf[model_name] = {
                    "auroc": summary[model_name]["test"].get("auroc", 0),
                    "auprc": summary[model_name]["test"].get("auprc", 0),
                    "f1": summary[model_name]["test"].get("f1", 0),
                    "precision": summary[model_name]["test"].get("precision", 0),
                    "recall": summary[model_name]["test"].get("recall", 0),
                }

    # Risk distribution
    risk_dist = {"LOW": 0, "MODERATE": 0, "HIGH": 0, "CRITICAL": 0}
    for p in _patient_state.values():
        risk_dist[p["risk_level"]] += 1

    return {
        "timestamp": datetime.now().isoformat(),
        "gpu": gpu_info,
        "models": model_perf,
        "risk_distribution": risk_dist,
        "total_patients": len(_patient_state),
        "alerts": alerting_engine.get_stats(),
    }


@app.get("/api/patients")
async def get_patients():
    """Current patient risk overview."""
    patients = list(_patient_state.values())
    patients.sort(key=lambda p: p["probability"], reverse=True)
    return {"patients": patients, "count": len(patients)}


@app.get("/api/alerts")
async def get_alerts(limit: int = 50, severity: Optional[str] = None):
    """Recent alerts."""
    return {
        "alerts": alerting_engine.get_recent_alerts(limit=limit),
        "stats": alerting_engine.get_stats(),
    }


@app.get("/api/drift")
async def get_drift_report():
    """Current drift analysis."""
    if len(drift_detector.live_buffer) >= 100:
        return drift_detector.generate_drift_report()
    return {"status": "insufficient_data", "message": "Need ≥100 live samples for drift analysis"}


@app.get("/api/shap")
async def get_shap_features():
    """Top SHAP feature importance."""
    summary_path = Path("models/artifacts/training_summary.json")
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        if "shap" in summary:
            return summary["shap"]
    return {"top_features": []}


# ── WebSocket ────────────────────────────────────────────────────────

@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    """Real-time dashboard updates via WebSocket — updates 2-3 patients per tick."""
    await websocket.accept()
    ws_connections.append(websocket)
    logger.info("Dashboard WebSocket connected. Total: {}", len(ws_connections))

    try:
        while True:
            await asyncio.sleep(2)

            # Update 2-3 random patients per tick
            n_updates = np.random.randint(2, 4)
            pids = np.random.choice(list(_patient_state.keys()), size=min(n_updates, len(_patient_state)), replace=False)

            for pid in pids:
                p = _patient_state[int(pid)]
                prev_prob = p["probability"]

                # Random walk on probability
                delta = np.random.normal(0, 0.025)
                p["probability"] = round(float(np.clip(p["probability"] + delta, 0.02, 0.98)), 4)

                # Update risk level
                if p["probability"] >= 0.75:
                    p["risk_level"] = "CRITICAL"
                elif p["probability"] >= 0.50:
                    p["risk_level"] = "HIGH"
                elif p["probability"] >= 0.25:
                    p["risk_level"] = "MODERATE"
                else:
                    p["risk_level"] = "LOW"

                # Random walk vitals for realism
                p["heart_rate"] = int(np.clip(p["heart_rate"] + np.random.normal(0, 2), 45, 160))
                p["spo2"] = round(float(np.clip(p["spo2"] + np.random.normal(0, 0.5), 82, 100)), 1)
                p["systolic_bp"] = int(np.clip(p["systolic_bp"] + np.random.normal(0, 3), 60, 200))
                p["respiratory_rate"] = int(np.clip(p["respiratory_rate"] + np.random.normal(0, 1), 8, 40))
                p["lactate"] = round(float(np.clip(p["lactate"] + np.random.normal(0, 0.1), 0.3, 10)), 1)

                p["trend"] = "↑" if delta > 0.01 else ("↓" if delta < -0.01 else "→")
                p["last_updated"] = datetime.now().isoformat()

                # Check alerts
                alerts = alerting_engine.evaluate_prediction(int(pid), p["probability"], prev_prob)

                await websocket.send_json({
                    "type": "patient_update",
                    "patient": p,
                    "alerts": [a.to_dict() for a in alerts],
                })

    except WebSocketDisconnect:
        ws_connections.remove(websocket)
        logger.info("Dashboard WebSocket disconnected. Total: {}", len(ws_connections))


# ── Entrypoint ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    logger.info("🖥️  Starting G_One_Sync Clinical Dashboard on http://localhost:8002")
    uvicorn.run(
        "src.dashboard.app:app",
        host="0.0.0.0",
        port=8002,
        reload=False,
    )

