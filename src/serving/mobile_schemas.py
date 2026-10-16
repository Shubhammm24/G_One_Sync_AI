"""
G_One_Sync AI — Mobile-Optimized Response Schemas
=====================================================
Pydantic models for the mobile companion app API. These are
deliberately compact — mobile clients need smaller payloads
with only the most clinically relevant fields.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

# ── Patient Summary (compact) ──────────────────────────────────────────


class PatientVitalsSummary(BaseModel):
    """Compact vital signs for mobile display."""

    heart_rate: float
    respiratory_rate: float
    spo2_pct: float
    temperature_c: float
    systolic_bp: float
    diastolic_bp: float


class PatientLabsSummary(BaseModel):
    """Compact lab results for mobile display."""

    wbc_count: float
    lactate: float
    creatinine: float


class RiskFactor(BaseModel):
    """Single SHAP-derived risk factor for mobile display."""

    feature_name: str
    display_name: str
    shap_value: float
    current_value: float
    direction: str = Field(..., pattern="^(increasing|decreasing)$")


class PatientRiskSummary(BaseModel):
    """Mobile-optimized single patient risk summary.

    Returns only the most clinically relevant fields to
    minimise payload size for mobile data connections.
    """

    patient_id: int
    risk_level: str  # LOW, MODERATE, HIGH, CRITICAL
    risk_score: int = Field(..., ge=0, le=100)
    deterioration_probability: float = Field(..., ge=0.0, le=1.0)
    vitals: PatientVitalsSummary
    labs: PatientLabsSummary
    top_risk_factors: list[RiskFactor] = []
    last_updated: str
    alerts: list[str] = []


# ── Watchlist (physician's patient list) ────────────────────────────────


class WatchlistEntry(BaseModel):
    """Single entry in the physician's watchlist."""

    patient_id: int
    risk_level: str
    risk_score: int = Field(..., ge=0, le=100)
    deterioration_probability: float
    primary_concern: str = ""  # Top risk factor display name
    last_updated: str
    trend: str = Field(
        default="stable",
        pattern="^(improving|stable|worsening)$",
        description="Risk trend over last 6 hours",
    )


class WatchlistResponse(BaseModel):
    """Physician's patient watchlist for mobile dashboard."""

    physician_id: str
    patients: list[WatchlistEntry]
    total_patients: int
    critical_count: int
    high_count: int
    last_refreshed: str


# ── Notification schemas ────────────────────────────────────────────────


class NotificationHistoryEntry(BaseModel):
    """Single notification in the alert history."""

    notification_id: str
    patient_id: Optional[int] = None
    risk_level: Optional[str] = None
    title: str
    body: str
    sent_at: str
    read: bool = False


class NotificationHistoryResponse(BaseModel):
    """Push notification history for alert screen."""

    notifications: list[NotificationHistoryEntry]
    total_count: int
    unread_count: int


# ── Device registration response ────────────────────────────────────────


class DeviceRegistrationResponse(BaseModel):
    """Response after registering a device for push notifications."""

    status: str
    device_token: str
    device_type: str
    physician_id: str
    registered_at: str


class TestNotificationResponse(BaseModel):
    """Response after sending a test push notification."""

    success: bool
    message_id: Optional[str] = None
    error: Optional[str] = None
