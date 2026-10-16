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

    return PatientRiskSummary(
        patient_id=data["patient_id"],
        risk_level=data["risk_level"],
        risk_score=data["risk_score"],
        deterioration_probability=data["deterioration_probability"],
        vitals=PatientVitalsSummary(**data["vitals"]),
        labs=PatientLabsSummary(**data["labs"]),
        top_risk_factors=[],  # Populated by SHAP when available
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
