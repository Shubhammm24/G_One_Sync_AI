"""
G_One_Sync AI — Model Serving API
====================================
Production FastAPI server for real-time clinical deterioration predictions.
Serves XGBoost, BiLSTM, and Transformer models with ensemble capability.

Endpoints:
    POST /predict          — Single patient prediction
    POST /predict/batch    — Batch predictions
    POST /predict/explain  — Prediction + SHAP explanation
    GET  /health           — Health check
    GET  /model/info       — Model metadata
"""

from __future__ import annotations

import time
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from config.settings import serving_settings, model_settings


# ── Request/Response Models ──────────────────────────────────────────────


class ModelType(str, Enum):
    XGBOOST = "xgboost"
    BILSTM = "bilstm"
    TRANSFORMER = "transformer"
    ENSEMBLE = "ensemble"


class VitalSigns(BaseModel):
    """Current vital signs for prediction."""
    heart_rate: float = Field(..., ge=20, le=300)
    respiratory_rate: float = Field(..., ge=4, le=60)
    spo2_pct: float = Field(..., ge=50, le=100)
    temperature_c: float = Field(..., ge=30, le=45)
    systolic_bp: float = Field(..., ge=40, le=300)
    diastolic_bp: float = Field(..., ge=20, le=200)
    oxygen_flow: float = Field(default=0.0, ge=0)
    mobility_score: int = Field(default=3, ge=0, le=5)
    nurse_alert: int = Field(default=0, ge=0, le=1)


class LabResults(BaseModel):
    """Current lab results for prediction."""
    wbc_count: float = Field(default=7.5, ge=0, le=100)
    lactate: float = Field(default=1.0, ge=0, le=30)
    creatinine: float = Field(default=1.0, ge=0, le=30)
    crp_level: float = Field(default=10.0, ge=0, le=500)
    hemoglobin: float = Field(default=13.0, ge=3, le=25)
    sepsis_risk_score: float = Field(default=0.1, ge=0, le=1)


class PredictionRequest(BaseModel):
    """Single patient prediction request."""
    patient_id: int
    vitals: VitalSigns
    labs: LabResults
    model_type: ModelType = ModelType.ENSEMBLE
    window_history: Optional[list[dict]] = None  # Previous timesteps


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    patients: list[PredictionRequest]
    model_type: ModelType = ModelType.ENSEMBLE


class RiskLevel(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class PredictionResponse(BaseModel):
    """Prediction result for a single patient."""
    patient_id: int
    deterioration_probability: float
    risk_level: RiskLevel
    risk_score: int = Field(..., ge=0, le=100)
    model_used: str
    prediction_time_ms: float
    clinical_alerts: list[str] = []
    top_risk_factors: list[dict] = []


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    total_patients: int
    high_risk_count: int
    avg_prediction_time_ms: float


class ModelInfo(BaseModel):
    model_name: str
    model_type: str
    is_loaded: bool
    feature_count: int
    last_updated: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    models_loaded: dict[str, bool]
    gpu_available: bool
    gpu_name: Optional[str] = None


# ── Model Registry ──────────────────────────────────────────────────────


class ModelRegistry:
    """
    Manages loaded models for serving.
    Handles model loading, caching, and prediction routing.
    """

    def __init__(self):
        self.models: dict[str, Any] = {}
        self.feature_names: dict[str, list[str]] = {}
        self._loaded = False

    def load_models(self, models_dir: Optional[Path] = None) -> None:
        """Load all available models from disk."""
        models_dir = models_dir or model_settings.artifacts_dir

        # Try loading XGBoost
        xgb_path = models_dir / "xgboost" / "xgboost_model.json"
        if xgb_path.exists():
            try:
                import xgboost as xgb
                model = xgb.XGBClassifier()
                model.load_model(str(xgb_path))
                self.models["xgboost"] = model

                # Load feature names
                feat_path = xgb_path.with_suffix(".features.json")
                if feat_path.exists():
                    import json
                    with open(feat_path) as f:
                        self.feature_names["xgboost"] = json.load(f)

                logger.info("✅ XGBoost model loaded")
            except Exception as e:
                logger.warning("Failed to load XGBoost: {}", e)

        # Try loading BiLSTM
        lstm_path = models_dir / "bilstm" / "bilstm_model.pt"
        if lstm_path.exists():
            try:
                import torch
                from src.modeling.lstm_trainer import BiLSTMAttentionModel
                checkpoint = torch.load(lstm_path, map_location="cpu", weights_only=False)
                hp = checkpoint["hyperparams"]
                model = BiLSTMAttentionModel(
                    input_size=hp["input_size"],
                    hidden_size=hp["hidden_size"],
                    num_layers=hp["num_layers"],
                    dropout=hp["dropout"],
                )
                model.load_state_dict(checkpoint["model_state_dict"])
                model.eval()
                if torch.cuda.is_available():
                    model = model.to("cuda")
                self.models["bilstm"] = model
                logger.info("✅ BiLSTM model loaded")
            except Exception as e:
                logger.warning("Failed to load BiLSTM: {}", e)

        # Try loading Transformer
        tf_path = models_dir / "transformer" / "transformer_model.pt"
        if tf_path.exists():
            try:
                import torch
                from src.modeling.transformer_trainer import TemporalTransformerModel
                checkpoint = torch.load(tf_path, map_location="cpu", weights_only=False)
                hp = checkpoint["hyperparams"]
                model = TemporalTransformerModel(
                    input_size=hp["input_size"],
                    d_model=hp["d_model"],
                    nhead=hp["nhead"],
                    num_layers=hp["num_layers"],
                    dim_feedforward=hp["dim_feedforward"],
                    dropout=hp["dropout"],
                )
                model.load_state_dict(checkpoint["model_state_dict"])
                model.eval()
                if torch.cuda.is_available():
                    model = model.to("cuda")
                self.models["transformer"] = model
                logger.info("✅ Transformer model loaded")
            except Exception as e:
                logger.warning("Failed to load Transformer: {}", e)

        self._loaded = True
        logger.info("Model Registry: {} models loaded", len(self.models))

    def predict(
        self,
        features: np.ndarray,
        model_type: str = "xgboost",
    ) -> np.ndarray:
        """Get prediction from a specific model."""
        if model_type not in self.models:
            raise ValueError(f"Model '{model_type}' not loaded")

        model = self.models[model_type]

        if model_type == "xgboost":
            return model.predict_proba(features)[:, 1]

        elif model_type in ("bilstm", "transformer"):
            import torch
            with torch.no_grad():
                tensor = torch.tensor(features, dtype=torch.float32)
                if torch.cuda.is_available():
                    tensor = tensor.to("cuda")
                if model_type == "bilstm":
                    logits, _ = model(tensor)
                else:
                    logits = model(tensor)
                return torch.sigmoid(logits).cpu().numpy()

        raise ValueError(f"Unknown model type: {model_type}")

    def ensemble_predict(self, features: np.ndarray) -> np.ndarray:
        """Average predictions from all loaded models."""
        predictions = []
        for name in self.models:
            try:
                pred = self.predict(features, name)
                predictions.append(pred)
            except Exception as e:
                logger.warning("Ensemble skip {}: {}", name, e)

        if not predictions:
            raise RuntimeError("No models available for ensemble")

        return np.mean(predictions, axis=0)


# ── FastAPI Application ──────────────────────────────────────────────────


app = FastAPI(
    title="G_One_Sync AI — Clinical Deterioration Prediction API",
    version="2.0.0",
    description="Real-time ICU deterioration risk scoring with explainability",
)

registry = ModelRegistry()


def _classify_risk(probability: float) -> tuple[RiskLevel, int, list[str]]:
    """Convert probability to risk level, score, and alerts."""
    score = int(probability * 100)
    alerts = []

    if probability >= 0.75:
        level = RiskLevel.CRITICAL
        alerts.append("⚠️ CRITICAL: Immediate clinical review recommended")
        alerts.append("Consider ICU escalation protocols")
    elif probability >= 0.50:
        level = RiskLevel.HIGH
        alerts.append("🔴 HIGH RISK: Close monitoring required (q15min vitals)")
        alerts.append("Notify attending physician")
    elif probability >= 0.25:
        level = RiskLevel.MODERATE
        alerts.append("🟡 MODERATE RISK: Increased surveillance recommended")
    else:
        level = RiskLevel.LOW
        alerts.append("🟢 LOW RISK: Continue standard monitoring")

    return level, score, alerts


def _build_feature_vector(req: PredictionRequest) -> np.ndarray:
    """Build feature vector from request vitals + labs."""
    v = req.vitals
    l = req.labs
    features = np.array([[
        v.heart_rate, v.respiratory_rate, v.spo2_pct,
        v.temperature_c, v.systolic_bp, v.diastolic_bp,
        v.oxygen_flow, v.mobility_score, v.nurse_alert,
        l.wbc_count, l.lactate, l.creatinine,
        l.crp_level, l.hemoglobin, l.sepsis_risk_score,
    ]], dtype=np.float32)
    return features


@app.on_event("startup")
async def startup():
    """Load models on startup."""
    logger.info("🚀 Starting G_One_Sync AI Prediction Server")
    try:
        registry.load_models()
    except Exception as e:
        logger.error("Failed to load models: {}", e)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    import torch
    gpu_available = torch.cuda.is_available()
    return HealthResponse(
        status="healthy",
        models_loaded={name: True for name in registry.models},
        gpu_available=gpu_available,
        gpu_name=torch.cuda.get_device_name(0) if gpu_available else None,
    )


@app.get("/model/info")
async def model_info():
    """Get loaded model information."""
    info = []
    for name, model in registry.models.items():
        feat_count = len(registry.feature_names.get(name, []))
        info.append(ModelInfo(
            model_name=name,
            model_type=name,
            is_loaded=True,
            feature_count=feat_count,
        ))
    return info


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Single patient deterioration prediction."""
    start = time.time()

    features = _build_feature_vector(request)
    model_type = request.model_type.value

    try:
        if model_type == "ensemble":
            prob = float(registry.ensemble_predict(features)[0])
            model_used = "ensemble"
        else:
            prob = float(registry.predict(features, model_type)[0])
            model_used = model_type
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    risk_level, risk_score, alerts = _classify_risk(prob)
    elapsed_ms = (time.time() - start) * 1000

    return PredictionResponse(
        patient_id=request.patient_id,
        deterioration_probability=round(prob, 4),
        risk_level=risk_level,
        risk_score=risk_score,
        model_used=model_used,
        prediction_time_ms=round(elapsed_ms, 2),
        clinical_alerts=alerts,
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Batch prediction for multiple patients."""
    predictions = []
    total_time = 0

    for patient_req in request.patients:
        patient_req.model_type = request.model_type
        result = await predict(patient_req)
        predictions.append(result)
        total_time += result.prediction_time_ms

    high_risk = sum(1 for p in predictions if p.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL))

    return BatchPredictionResponse(
        predictions=predictions,
        total_patients=len(predictions),
        high_risk_count=high_risk,
        avg_prediction_time_ms=round(total_time / max(len(predictions), 1), 2),
    )


# ── CLI Entrypoint ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.serving.model_server:app",
        host=serving_settings.serving_host,
        port=serving_settings.serving_port,
        reload=False,
        workers=1,
    )
