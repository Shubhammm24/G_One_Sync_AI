"""
G_One_Sync AI — Mobile API Router
=====================================
FastAPI router providing mobile-optimized endpoints for the
companion physician app. Includes:

  * ``POST /auth/login``              — Authenticate, return JWT pair
  * ``POST /auth/refresh``            — Refresh access token
  * ``GET  /mobile/watchlist``        — Physician's patient watchlist
  * ``GET  /mobile/patient/{id}``     — Single patient risk summary
  * ``POST /notifications/register``  — Register FCM device token
  * ``POST /notifications/test``      — Send test push notification
  * ``DELETE /notifications/{token}`` — Unregister device

All endpoints respect the ``auth_enabled`` feature flag. When
disabled, auth endpoints still work (for client development)
but protected endpoints skip token validation.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger

from config.settings import mobile_settings
from src.serving.auth import (
    LoginRequest,
    RefreshRequest,
    TokenPayload,
    TokenResponse,
    create_access_token,
    create_refresh_token,
    decode_token,
    require_auth,
    validate_credentials,
)
from src.serving.mobile_schemas import (
    DeviceRegistrationResponse,
    PatientLabsSummary,
    PatientRiskSummary,
    PatientVitalsSummary,
    RiskFactor,
    TestNotificationResponse,
    WatchlistEntry,
    WatchlistResponse,
)
from src.serving.push_notifications import (
    DeviceRegistration,
    DeviceRegistry,
    PushNotificationService,
)

# ── Router Setup ────────────────────────────────────────────────────────

auth_router = APIRouter(prefix="/auth", tags=["Authentication"])
mobile_router = APIRouter(prefix="/mobile", tags=["Mobile"])
notification_router = APIRouter(prefix="/notifications", tags=["Push Notifications"])

# Shared service instances (created once, reused across requests)
_device_registry: Optional[DeviceRegistry] = None
_push_service: Optional[PushNotificationService] = None


def _get_device_registry() -> DeviceRegistry:
    """Lazy-init device registry singleton."""
    global _device_registry
    if _device_registry is None:
        _device_registry = DeviceRegistry()
    return _device_registry


def _get_push_service() -> PushNotificationService:
    """Lazy-init push notification service singleton."""
    global _push_service
    if _push_service is None:
        _push_service = PushNotificationService(_get_device_registry())
    return _push_service


# ── Simulated Patient Store ─────────────────────────────────────────────
# In production, this would query the data lake or a patient database.
# For Phase 8, we use an in-memory store that the serving API can update.

_patient_store: dict[int, dict] = {}


def _seed_demo_patients() -> None:
    """Populate default ICU demo patients if patient store is empty."""
    if _patient_store:
        return
    now_iso = datetime.now(tz=timezone.utc).isoformat()
    demo_data = [
        {
            "patient_id": 1042,
            "risk_level": "CRITICAL",
            "risk_score": 88,
            "deterioration_probability": 0.88,
            "vitals": {"heart_rate": 128.0, "respiratory_rate": 32.0, "spo2_pct": 86.0, "temperature_c": 38.9, "systolic_bp": 82.0, "diastolic_bp": 48.0},
            "labs": {"wbc_count": 19.5, "lactate": 4.2, "creatinine": 2.1},
            "alerts": ["Severe Hypoxemia: SpO2 86%", "Tachycardia: HR 128 bpm", "Lactate critical: 4.2 mmol/L"],
            "top_risk_factors": [
                {"feature_name": "spo2_pct", "display_name": "SpO2 Saturation", "shap_value": 0.32, "current_value": 86.0, "direction": "increasing"},
                {"feature_name": "lactate", "display_name": "Serum Lactate", "shap_value": 0.28, "current_value": 4.2, "direction": "increasing"},
                {"feature_name": "respiratory_rate", "display_name": "Respiratory Rate", "shap_value": 0.21, "current_value": 32.0, "direction": "increasing"},
            ],
            "last_updated": now_iso,
            "trend": "worsening",
        },
        {
            "patient_id": 1019,
            "risk_level": "CRITICAL",
            "risk_score": 82,
            "deterioration_probability": 0.82,
            "vitals": {"heart_rate": 115.0, "respiratory_rate": 26.0, "spo2_pct": 91.0, "temperature_c": 39.2, "systolic_bp": 85.0, "diastolic_bp": 52.0},
            "labs": {"wbc_count": 21.0, "lactate": 3.8, "creatinine": 2.4},
            "alerts": ["Septic Shock: Refractory Hypotension", "Elevated Lactate: 3.8 mmol/L"],
            "top_risk_factors": [
                {"feature_name": "systolic_bp", "display_name": "Systolic Blood Pressure", "shap_value": 0.35, "current_value": 85.0, "direction": "increasing"},
                {"feature_name": "lactate", "display_name": "Serum Lactate", "shap_value": 0.25, "current_value": 3.8, "direction": "increasing"},
            ],
            "last_updated": now_iso,
            "trend": "worsening",
        },
        {
            "patient_id": 1088,
            "risk_level": "HIGH",
            "risk_score": 68,
            "deterioration_probability": 0.68,
            "vitals": {"heart_rate": 104.0, "respiratory_rate": 28.0, "spo2_pct": 92.0, "temperature_c": 38.1, "systolic_bp": 100.0, "diastolic_bp": 62.0},
            "labs": {"wbc_count": 16.2, "lactate": 2.6, "creatinine": 1.4},
            "alerts": ["Tachypnea: RR 28 breaths/min", "WBC elevated: 16.2 K/uL"],
            "top_risk_factors": [
                {"feature_name": "respiratory_rate", "display_name": "Respiratory Rate", "shap_value": 0.29, "current_value": 28.0, "direction": "increasing"},
                {"feature_name": "wbc_count", "display_name": "WBC Count", "shap_value": 0.22, "current_value": 16.2, "direction": "increasing"},
            ],
            "last_updated": now_iso,
            "trend": "worsening",
        },
        {
            "patient_id": 1023,
            "risk_level": "HIGH",
            "risk_score": 54,
            "deterioration_probability": 0.54,
            "vitals": {"heart_rate": 96.0, "respiratory_rate": 22.0, "spo2_pct": 94.0, "temperature_c": 37.6, "systolic_bp": 94.0, "diastolic_bp": 58.0},
            "labs": {"wbc_count": 12.8, "lactate": 1.9, "creatinine": 1.2},
            "alerts": ["Borderline Hypotension: SBP 94 mmHg"],
            "top_risk_factors": [
                {"feature_name": "systolic_bp", "display_name": "Systolic Blood Pressure", "shap_value": 0.20, "current_value": 94.0, "direction": "increasing"},
            ],
            "last_updated": now_iso,
            "trend": "stable",
        },
        {
            "patient_id": 1055,
            "risk_level": "MODERATE",
            "risk_score": 38,
            "deterioration_probability": 0.38,
            "vitals": {"heart_rate": 92.0, "respiratory_rate": 19.0, "spo2_pct": 96.0, "temperature_c": 37.2, "systolic_bp": 118.0, "diastolic_bp": 72.0},
            "labs": {"wbc_count": 9.4, "lactate": 1.4, "creatinine": 1.0},
            "alerts": ["Mild Tachycardia: HR 92 bpm"],
            "top_risk_factors": [
                {"feature_name": "heart_rate", "display_name": "Heart Rate", "shap_value": 0.15, "current_value": 92.0, "direction": "increasing"},
            ],
            "last_updated": now_iso,
            "trend": "stable",
        },
        {
            "patient_id": 1071,
            "risk_level": "MODERATE",
            "risk_score": 32,
            "deterioration_probability": 0.32,
            "vitals": {"heart_rate": 84.0, "respiratory_rate": 18.0, "spo2_pct": 97.0, "temperature_c": 36.9, "systolic_bp": 122.0, "diastolic_bp": 78.0},
            "labs": {"wbc_count": 8.1, "lactate": 1.2, "creatinine": 0.9},
            "alerts": [],
            "top_risk_factors": [],
            "last_updated": now_iso,
            "trend": "improving",
        },
        {
            "patient_id": 1012,
            "risk_level": "LOW",
            "risk_score": 14,
            "deterioration_probability": 0.14,
            "vitals": {"heart_rate": 72.0, "respiratory_rate": 16.0, "spo2_pct": 98.0, "temperature_c": 36.8, "systolic_bp": 124.0, "diastolic_bp": 76.0},
            "labs": {"wbc_count": 6.8, "lactate": 0.9, "creatinine": 0.8},
            "alerts": [],
            "top_risk_factors": [],
            "last_updated": now_iso,
            "trend": "improving",
        },
        {
            "patient_id": 1030,
            "risk_level": "LOW",
            "risk_score": 8,
            "deterioration_probability": 0.08,
            "vitals": {"heart_rate": 68.0, "respiratory_rate": 14.0, "spo2_pct": 99.0, "temperature_c": 36.6, "systolic_bp": 118.0, "diastolic_bp": 74.0},
            "labs": {"wbc_count": 5.9, "lactate": 0.7, "creatinine": 0.8},
            "alerts": [],
            "top_risk_factors": [],
            "last_updated": now_iso,
            "trend": "stable",
        },
    ]
    for d in demo_data:
        _patient_store[d["patient_id"]] = d


# Initialize demo data on module load
_seed_demo_patients()


def update_patient_risk(
    patient_id: int,
    risk_level: str,
    risk_score: int,
    probability: float,
    vitals: Optional[dict] = None,
    labs: Optional[dict] = None,
    alerts: Optional[list[str]] = None,
) -> None:
    """Update or insert patient risk data in the mobile store.

    Called by the prediction pipeline after each prediction to keep
    the mobile watchlist current.

    Args:
        patient_id: Patient identifier.
        risk_level: Risk classification (LOW/MODERATE/HIGH/CRITICAL).
        risk_score: Integer risk score 0-100.
        probability: Deterioration probability.
        vitals: Current vital signs dict.
        labs: Current lab results dict.
        alerts: Clinical alert strings.
    """
    _patient_store[patient_id] = {
        "patient_id": patient_id,
        "risk_level": risk_level,
        "risk_score": risk_score,
        "deterioration_probability": probability,
        "vitals": vitals or {
            "heart_rate": 80.0,
            "respiratory_rate": 18.0,
            "spo2_pct": 97.0,
            "temperature_c": 37.0,
            "systolic_bp": 120.0,
            "diastolic_bp": 75.0,
        },
        "labs": labs or {
            "wbc_count": 7.5,
            "lactate": 1.0,
            "creatinine": 1.0,
        },
        "alerts": alerts or [],
        "last_updated": datetime.now(tz=timezone.utc).isoformat(),
        "trend": "stable",
    }


# ═════════════════════════════════════════════════════════════════════════
# AUTH ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════


@auth_router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    """Authenticate a physician and return JWT token pair.

    In demo mode (no password hash configured), accepts any password
    for the demo username. In production, validates against bcrypt hash.
    """
    if not validate_credentials(request.username, request.password):
        logger.warning("Failed login attempt for user='{}'", request.username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    access_token = create_access_token(request.username)
    refresh_token = create_refresh_token(request.username)

    logger.info("Physician '{}' logged in successfully", request.username)
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=mobile_settings.jwt_expiry_hours * 3600,
    )


@auth_router.post("/refresh", response_model=TokenResponse)
async def refresh_token(request: RefreshRequest):
    """Exchange a valid refresh token for a new access token pair."""
    payload = decode_token(request.refresh_token)

    if payload.token_type != "refresh":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid token type — expected refresh token",
        )

    access_token = create_access_token(payload.sub)
    refresh_token = create_refresh_token(payload.sub)

    logger.info("Token refreshed for user='{}'", payload.sub)
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=mobile_settings.jwt_expiry_hours * 3600,
    )


# ═════════════════════════════════════════════════════════════════════════
# MOBILE ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════


@mobile_router.get("/watchlist", response_model=WatchlistResponse)
async def get_watchlist(
    auth: Optional[TokenPayload] = Depends(require_auth),
):
    """Get the physician's patient watchlist.

    Returns a compact list of all monitored patients sorted by
    risk score (highest first).
    """
    physician_id = auth.sub if auth else mobile_settings.demo_username

    # Build watchlist from patient store
    entries = []
    for pid, data in _patient_store.items():
        entries.append(WatchlistEntry(
            patient_id=data["patient_id"],
            risk_level=data["risk_level"],
            risk_score=data["risk_score"],
            deterioration_probability=data["deterioration_probability"],
            primary_concern=data["alerts"][0] if data.get("alerts") else "",
            last_updated=data["last_updated"],
            trend=data.get("trend", "stable"),
        ))

    # Sort by risk score descending (most critical first)
    entries.sort(key=lambda e: e.risk_score, reverse=True)

    critical = sum(1 for e in entries if e.risk_level == "CRITICAL")
    high = sum(1 for e in entries if e.risk_level == "HIGH")

    return WatchlistResponse(
        physician_id=physician_id,
        patients=entries,
        total_patients=len(entries),
        critical_count=critical,
        high_count=high,
        last_refreshed=datetime.now(tz=timezone.utc).isoformat(),
    )


@mobile_router.get("/patient/{patient_id}", response_model=PatientRiskSummary)
async def get_patient_summary(
    patient_id: int,
    auth: Optional[TokenPayload] = Depends(require_auth),
):
    """Get a single patient's risk summary for mobile display.

    Returns compact vitals, labs, risk assessment, and top risk
    factors for the patient detail screen.
    """
    if patient_id not in _patient_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient {patient_id} not found in watchlist",
        )

    data = _patient_store[patient_id]
    risk_factors = [
        RiskFactor(**rf) if isinstance(rf, dict) else rf
        for rf in data.get("top_risk_factors", [])
    ]

    return PatientRiskSummary(
        patient_id=data["patient_id"],
        risk_level=data["risk_level"],
        risk_score=data["risk_score"],
        deterioration_probability=data["deterioration_probability"],
        vitals=PatientVitalsSummary(**data["vitals"]),
        labs=PatientLabsSummary(**data["labs"]),
        top_risk_factors=risk_factors,
        last_updated=data["last_updated"],
        alerts=data.get("alerts", []),
    )


# ═════════════════════════════════════════════════════════════════════════
# NOTIFICATION ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════


@notification_router.post("/register", response_model=DeviceRegistrationResponse)
async def register_device(
    registration: DeviceRegistration,
    auth: Optional[TokenPayload] = Depends(require_auth),
):
    """Register a mobile device for push notifications.

    The physician_id in the registration body should match the
    authenticated user. When auth is disabled, the body value is
    used directly.
    """
    # If auth is enabled, override physician_id with token subject
    if auth and registration.physician_id != auth.sub:
        logger.warning(
            "Physician ID mismatch: body='{}' token='{}'",
            registration.physician_id,
            auth.sub,
        )
        registration.physician_id = auth.sub

    registry = _get_device_registry()
    info = registry.register(registration)

    return DeviceRegistrationResponse(
        status="registered",
        device_token=info.device_token,
        device_type=info.device_type,
        physician_id=info.physician_id,
        registered_at=info.registered_at,
    )


@notification_router.post("/test", response_model=TestNotificationResponse)
async def send_test_notification(
    device_token: str,
    auth: Optional[TokenPayload] = Depends(require_auth),
):
    """Send a test push notification to verify connectivity."""
    push = _get_push_service()
    result = await push.send_test_notification(device_token)

    return TestNotificationResponse(
        success=result.success,
        message_id=result.message_id,
        error=result.error,
    )


@notification_router.delete("/{device_token}")
async def unregister_device(
    device_token: str,
    auth: Optional[TokenPayload] = Depends(require_auth),
):
    """Unregister a device from push notifications."""
    registry = _get_device_registry()
    removed = registry.unregister(device_token)

    if not removed:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Device token not found",
        )

    return {"status": "unregistered", "device_token": device_token}


@notification_router.get("/devices")
async def list_devices(
    auth: Optional[TokenPayload] = Depends(require_auth),
):
    """List all registered devices (admin/debug endpoint)."""
    registry = _get_device_registry()
    devices = registry.get_all_devices()

    return {
        "total_devices": len(devices),
        "devices": [d.model_dump() for d in devices],
    }
