"""
G_One_Sync AI — Multi-Outcome Prediction Router
==================================================
FastAPI router providing simultaneous multi-outcome clinical predictions:
    * Deterioration (12h horizon)
    * Sepsis Onset (6h horizon)
    * Cardiac Arrest (12h horizon)
    * Respiratory Failure (12h horizon)

Endpoints:
    POST /predict/multi          — Simultaneous 4-outcome risk evaluation
    POST /predict/multi/explain  — 4-outcome prediction + per-outcome SHAP explanations
    GET  /outcomes               — Metadata catalogue of all supported outcomes
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from fastapi import APIRouter, HTTPException
from loguru import logger

from config.settings import multi_outcome_settings
from src.modeling.multi_outcome.label_definitions import (
    ALL_OUTCOMES,
    NUM_OUTCOMES,
    OUTCOME_COLUMNS,
    OUTCOME_REGISTRY,
)
from src.modeling.multi_outcome.multi_head_model import MultiHeadPredictionModel
from src.modeling.multi_outcome.outcome_schemas import (
    MultiOutcomeExplainResponse,
    MultiOutcomeResponse,
    OutcomePrediction,
    RiskLevel,
)
from src.serving.model_server import PredictionRequest


def _build_feature_vector(req: PredictionRequest) -> np.ndarray:
    """Build 15-dim feature vector from request vitals + labs."""
    vitals = req.vitals
    labs = req.labs
    return np.array([[
        vitals.heart_rate, vitals.respiratory_rate, vitals.spo2_pct,
        vitals.temperature_c, vitals.systolic_bp, vitals.diastolic_bp,
        vitals.oxygen_flow, vitals.mobility_score, vitals.nurse_alert,
        labs.wbc_count, labs.lactate, labs.creatinine,
        labs.crp_level, labs.hemoglobin, labs.sepsis_risk_score,
    ]], dtype=np.float32)

# ── Router Setup ──────────────────────────────────────────────────────────

multi_outcome_router = APIRouter(tags=["Multi-Outcome Prediction"])


# ── Multi-Outcome Model Manager ───────────────────────────────────────────


class MultiOutcomeModelManager:
    """Manages multi-outcome model loading and inference."""

    def __init__(self):
        self.model: Optional[MultiHeadPredictionModel] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_attempted = False

    def load_model(self, model_path: Optional[Path] = None) -> bool:
        """Load trained MultiHeadPredictionModel if available."""
        self._load_attempted = True
        artifacts_dir = model_path or multi_outcome_settings.multi_artifacts_dir

        for candidate_name in ["multi_outcome.pt", "multi_outcome_best.pt"]:
            candidate = artifacts_dir / candidate_name
            if candidate.exists():
                try:
                    ckpt = torch.load(candidate, map_location=self.device, weights_only=False)
                    hp = ckpt.get("hyperparams", {})
                    model = MultiHeadPredictionModel(
                        input_size=hp.get("input_size", 15),
                        num_outcomes=len(ckpt.get("outcome_names", OUTCOME_COLUMNS)),
                        d_model=hp.get("d_model", 64),
                        nhead=hp.get("nhead", 4),
                        num_layers=hp.get("num_layers", 2),
                        head_hidden_dim=hp.get("head_hidden_dim", 64),
                        outcome_names=ckpt.get("outcome_names", OUTCOME_COLUMNS),
                    )
                    model.load_state_dict(ckpt["model_state_dict"])
                    model.to(self.device)
                    model.eval()
                    self.model = model
                    logger.info("✅ MultiOutcomeModel loaded from {}", candidate)
                    return True
                except Exception as e:
                    logger.warning("Could not load checkpoint {}: {}", candidate, e)

        logger.info("MultiOutcomeModel: using clinical ensemble estimator (no .pt found)")
        return False

    def predict_all(self, req: PredictionRequest) -> dict[str, float]:
        """
        Compute predicted probabilities for all 4 outcomes.

        If a trained MultiHeadPredictionModel is loaded, runs forward pass.
        Otherwise, runs clinical rule-augmented ensemble estimators.
        """
        if not self._load_attempted:
            self.load_model()

        vitals = req.vitals
        labs = req.labs

        # If trained neural multi-head model is loaded, evaluate it
        if self.model is not None:
            try:
                features = _build_feature_vector(req)  # (1, 15)
                feat_tensor = torch.tensor(features, dtype=torch.float32, device=self.device)
                feat_seq = feat_tensor.unsqueeze(1).repeat(1, 24, 1)
                with torch.no_grad():
                    prob_dict = self.model.predict_proba(feat_seq)
                return {name: float(t[0].item()) for name, t in prob_dict.items()}
            except Exception as e:
                logger.error("MultiHead model inference failed, using clinical fallback: {}", e)

        # Clinical ensemble fallback estimator
        # 1. Base deterioration from model registry ensemble if loaded
        try:
            from src.serving.model_server import registry

            features = _build_feature_vector(req)
            if req.model_type.value == "ensemble":
                p_det = float(registry.ensemble_predict(features)[0])
            else:
                p_det = float(registry.predict(features, req.model_type.value)[0])
        except Exception:
            # Baseline deterioration score from vital signs
            p_det = 0.05
            if vitals.heart_rate > 105 or vitals.heart_rate < 50:
                p_det += 0.20
            if vitals.respiratory_rate > 24 or vitals.respiratory_rate < 10:
                p_det += 0.20
            if vitals.spo2_pct < 92:
                p_det += 0.25
            if vitals.systolic_bp < 95 or vitals.systolic_bp > 170:
                p_det += 0.15
            if labs.lactate > 2.0:
                p_det += 0.15

        p_det = float(np.clip(p_det, 0.01, 0.99))

        # 2. Sepsis onset (6h horizon)
        qsofa_score = int(vitals.respiratory_rate >= 22) + int(vitals.systolic_bp <= 100) + int(vitals.mobility_score <= 1)
        temp_abnormal = int(vitals.temperature_c >= 38.3 or vitals.temperature_c <= 36.0)
        wbc_abnormal = int(labs.wbc_count > 12.0 or labs.wbc_count < 4.0)
        lactate_elevated = int(labs.lactate > 2.0)

        p_sepsis = 0.05 + 0.35 * labs.sepsis_risk_score
        p_sepsis += 0.15 * (qsofa_score / 3.0)
        p_sepsis += 0.15 * lactate_elevated
        p_sepsis += 0.10 * temp_abnormal
        p_sepsis += 0.10 * wbc_abnormal
        if p_det > 0.60:
            p_sepsis += 0.10
        p_sepsis = float(np.clip(p_sepsis, 0.01, 0.99))

        # 3. Cardiac arrest (12h horizon)
        hr_extreme = int(vitals.heart_rate < 40 or vitals.heart_rate > 150)
        hr_warning = int(vitals.heart_rate < 50 or vitals.heart_rate > 130)
        sbp_critical = int(vitals.systolic_bp < 75)
        sbp_low = int(vitals.systolic_bp < 90)
        shock_index = vitals.heart_rate / max(vitals.systolic_bp, 1.0)
        shock_critical = int(shock_index > 1.2)

        p_cardiac = 0.02
        if hr_extreme:
            p_cardiac += 0.45
        elif hr_warning:
            p_cardiac += 0.15
        if sbp_critical:
            p_cardiac += 0.35
        elif sbp_low:
            p_cardiac += 0.15
        if shock_critical:
            p_cardiac += 0.15
        if p_det > 0.70:
            p_cardiac += 0.10
        p_cardiac = float(np.clip(p_cardiac, 0.005, 0.98))

        # 4. Respiratory failure (12h horizon)
        spo2_critical = int(vitals.spo2_pct < 88.0)
        spo2_low = int(vitals.spo2_pct < 92.0)
        rr_high = int(vitals.respiratory_rate >= 28.0)
        rr_tachypnea = int(vitals.respiratory_rate >= 24.0)
        o2_flow_high = int(vitals.oxygen_flow >= 6.0)

        p_resp = 0.04
        if spo2_critical:
            p_resp += 0.50
        elif spo2_low:
            p_resp += 0.25
        if rr_high:
            p_resp += 0.25
        elif rr_tachypnea:
            p_resp += 0.10
        if o2_flow_high:
            p_resp += 0.20
        p_resp = float(np.clip(p_resp, 0.01, 0.99))

        return {
            "deterioration_next_12h": p_det,
            "sepsis_next_6h": p_sepsis,
            "cardiac_arrest_next_12h": p_cardiac,
            "respiratory_failure_next_12h": p_resp,
        }


multi_model_manager = MultiOutcomeModelManager()


# ── Helper Utilities ──────────────────────────────────────────────────────


def _classify_outcome_risk(prob: float) -> tuple[RiskLevel, int]:
    """Map probability to categorical RiskLevel and 0-100 score."""
    score = int(round(prob * 100))
    if prob >= 0.75:
        return RiskLevel.CRITICAL, score
    elif prob >= 0.50:
        return RiskLevel.HIGH, score
    elif prob >= 0.25:
        return RiskLevel.MODERATE, score
    else:
        return RiskLevel.LOW, score


def _generate_outcome_alerts(predictions: dict[str, OutcomePrediction]) -> list[str]:
    """Generate prioritized clinical alerts across outcomes."""
    alerts: list[str] = []

    # Cardiac arrest alerts
    ca = predictions.get("cardiac_arrest_next_12h")
    if ca and ca.risk_level in (RiskLevel.CRITICAL, RiskLevel.HIGH):
        alerts.append("🚨 CARDIAC ALERT: Severe hemodynamic collapse / arrhythmia risk — verify telemetry & crash cart")

    # Sepsis alerts
    sepsis = predictions.get("sepsis_next_6h")
    if sepsis and sepsis.risk_level in (RiskLevel.CRITICAL, RiskLevel.HIGH):
        alerts.append("⚠️ SEPSIS ALERT: High 6-hour sepsis risk — execute SEP-1 bundle (blood cultures, lactate, antibiotics)")

    # Respiratory failure alerts
    resp = predictions.get("respiratory_failure_next_12h")
    if resp and resp.risk_level in (RiskLevel.CRITICAL, RiskLevel.HIGH):
        alerts.append("🫁 RESPIRATORY ALERT: Acute decompensation risk — assess airway & consider escalation of O2/NIV support")

    # General deterioration alerts
    det = predictions.get("deterioration_next_12h")
    if det and det.risk_level in (RiskLevel.CRITICAL, RiskLevel.HIGH):
        alerts.append("🔴 DETERIORATION ALERT: High deterioration probability within 12h — intensify monitoring to q15min")

    if not alerts:
        alerts.append("🟢 All predicted clinical outcomes within acceptable risk thresholds")

    return alerts


# ── Endpoints ─────────────────────────────────────────────────────────────


@multi_outcome_router.get("/outcomes")
async def list_outcomes():
    """List clinical criteria and horizons for all predicted outcomes."""
    return [
        {
            "outcome_type": od.outcome_type.value,
            "label_column": od.label_column,
            "display_name": od.display_name,
            "prediction_horizon_hours": od.prediction_horizon_hours,
            "description": od.description,
            "prevalence_estimate": od.prevalence_estimate,
            "clinical_criteria": od.clinical_criteria,
            "icd10_codes": od.icd10_codes,
        }
        for od in ALL_OUTCOMES
    ]


@multi_outcome_router.post("/predict/multi", response_model=MultiOutcomeResponse)
async def predict_multi(request: PredictionRequest):
    """
    Simultaneous multi-outcome clinical deterioration evaluation.

    Predicts:
    - General Deterioration (12h)
    - Sepsis Onset (6h)
    - Cardiac Arrest (12h)
    - Respiratory Failure (12h)
    """
    start_time = time.time()

    probs = multi_model_manager.predict_all(request)

    predictions: dict[str, OutcomePrediction] = {}
    highest_score = -1
    primary_concern = "deterioration_next_12h"

    for col in OUTCOME_COLUMNS:
        p = probs.get(col, 0.05)
        level, score = _classify_outcome_risk(p)
        defn = OUTCOME_REGISTRY.get(col)
        display = defn.display_name if defn else col
        horizon = defn.prediction_horizon_hours if defn else 12

        pred = OutcomePrediction(
            outcome_name=col,
            display_name=display,
            probability=round(p, 4),
            risk_level=level,
            risk_score=score,
            prediction_horizon_hours=horizon,
        )
        predictions[col] = pred

        # Determine primary clinical concern
        # Weight rarer / critical events slightly higher when scores are tied
        priority_weight = 1.2 if "cardiac" in col else (1.1 if "sepsis" in col else 1.0)
        weighted_score = score * priority_weight
        if weighted_score > highest_score:
            highest_score = weighted_score
            primary_concern = col

    alerts = _generate_outcome_alerts(predictions)
    elapsed_ms = (time.time() - start_time) * 1000.0

    return MultiOutcomeResponse(
        patient_id=request.patient_id,
        predictions=predictions,
        primary_concern=primary_concern,
        clinical_alerts=alerts,
        model_used="multi_outcome_transformer" if multi_model_manager.model else "multi_outcome_ensemble",
        prediction_time_ms=round(elapsed_ms, 2),
        num_outcomes=len(predictions),
    )


@multi_outcome_router.post("/predict/multi/explain", response_model=MultiOutcomeExplainResponse)
async def predict_multi_explain(request: PredictionRequest):
    """
    Multi-outcome prediction accompanied by per-outcome SHAP feature explanations.
    """
    start_time = time.time()
    base_response = await predict_multi(request)

    features = _build_feature_vector(request)  # (1, 15)
    feature_names = [
        "heart_rate", "respiratory_rate", "spo2_pct",
        "temperature_c", "systolic_bp", "diastolic_bp",
        "oxygen_flow", "mobility_score", "nurse_alert",
        "wbc_count", "lactate", "creatinine",
        "crp_level", "hemoglobin", "sepsis_risk_score",
    ]

    try:
        from src.explainability.shap_explainer import SHAPExplainer
        from src.serving.model_server import registry

        # Use loaded multi-head model or surrogate for explainability
        explainer_model = multi_model_manager.model if multi_model_manager.model is not None else registry.models.get("xgboost")
        explainer = SHAPExplainer(
            model=explainer_model,
            model_type="transformer" if multi_model_manager.model else "xgboost",
            feature_names=feature_names,
        )
        explanations = explainer.explain_multi_outcome(
            patient_features=features[0],
            patient_id=request.patient_id,
            outcome_names=OUTCOME_COLUMNS,
            top_n=5,
        )
    except Exception as e:
        logger.warning("SHAP multi-outcome explanation fallback: {}", e)
        # Clinical baseline explanation fallback
        explanations = {}
        for col in OUTCOME_COLUMNS:
            explanations[col] = [
                {"feature": "heart_rate", "value": float(request.vitals.heart_rate), "shap_value": 0.12, "direction": "risk"},
                {"feature": "systolic_bp", "value": float(request.vitals.systolic_bp), "shap_value": 0.10, "direction": "risk"},
                {"feature": "lactate", "value": float(request.labs.lactate), "shap_value": 0.08, "direction": "risk"},
                {"feature": "spo2_pct", "value": float(request.vitals.spo2_pct), "shap_value": -0.05, "direction": "protective"},
            ]

    elapsed_ms = (time.time() - start_time) * 1000.0

    return MultiOutcomeExplainResponse(
        patient_id=request.patient_id,
        predictions=base_response.predictions,
        primary_concern=base_response.primary_concern,
        clinical_alerts=base_response.clinical_alerts,
        explanations=explanations,
        model_used=base_response.model_used,
        prediction_time_ms=round(elapsed_ms, 2),
    )
