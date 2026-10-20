"""
G_One_Sync AI — Multi-Outcome Prediction Schemas
===================================================
Pydantic models for multi-outcome API requests and responses.
Used by the /predict/multi endpoint in the serving API.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from src.modeling.multi_outcome.label_definitions import (
    NUM_OUTCOMES,
    OUTCOME_COLUMNS,
)


class RiskLevel(str, Enum):
    """Risk classification levels."""

    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class OutcomePrediction(BaseModel):
    """Prediction result for a single clinical outcome."""

    outcome_name: str = Field(..., description="Clinical outcome identifier")
    display_name: str = Field(..., description="Human-readable outcome name")
    probability: float = Field(..., ge=0.0, le=1.0, description="Calibrated probability")
    risk_level: RiskLevel = Field(..., description="Risk classification")
    risk_score: int = Field(..., ge=0, le=100, description="Risk score 0-100")
    prediction_horizon_hours: int = Field(..., description="Prediction horizon in hours")


class MultiOutcomeResponse(BaseModel):
    """Response for multi-outcome prediction endpoint."""

    patient_id: int
    predictions: dict[str, OutcomePrediction] = Field(
        ..., description="Per-outcome prediction results"
    )
    primary_concern: str = Field(
        ..., description="Outcome with highest risk score"
    )
    clinical_alerts: list[str] = Field(
        default_factory=list, description="Consolidated clinical alerts"
    )
    model_used: str = "multi_outcome_ensemble"
    prediction_time_ms: float = Field(..., description="Inference latency in milliseconds")
    num_outcomes: int = Field(default=NUM_OUTCOMES, description="Number of predicted outcomes")


class MultiOutcomeExplainResponse(BaseModel):
    """Response for multi-outcome prediction with SHAP explanations."""

    patient_id: int
    predictions: dict[str, OutcomePrediction]
    primary_concern: str
    clinical_alerts: list[str] = []
    explanations: dict[str, list[dict]] = Field(
        default_factory=dict,
        description="Per-outcome SHAP feature importance: {outcome: [{feature, value, shap_value}]}",
    )
    model_used: str = "multi_outcome_ensemble"
    prediction_time_ms: float = 0.0
