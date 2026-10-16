"""
G_One_Sync AI — Mobile API Tests
===================================
Tests for mobile companion app endpoints, JWT authentication,
and FCM push notification registry/dispatch services (Phase 8).
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from config.settings import mobile_settings
from src.serving.auth import (
    TokenPayload,
    create_access_token,
    create_refresh_token,
    decode_token,
    validate_credentials,
)
from src.serving.mobile_router import _patient_store, update_patient_risk
from src.serving.model_server import app
from src.serving.push_notifications import (
    DeviceInfo,
    DeviceRegistration,
    DeviceRegistry,
    PushNotificationService,
    PushPayload,
)


@pytest.fixture
def client():
    """Create a FastAPI test client for serving and mobile routes."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_patient_store():
    """Clear mobile in-memory patient store between tests."""
    _patient_store.clear()
    yield
    _patient_store.clear()


# ═════════════════════════════════════════════════════════════════════════
# 1. JWT Authentication Unit Tests
# ═════════════════════════════════════════════════════════════════════════


class TestJWTAuth:
    """Unit tests for token creation, verification, and credentials."""

    def test_create_and_decode_access_token(self):
        token = create_access_token("dr.sharma")
        assert isinstance(token, str)
        payload = decode_token(token)
        assert isinstance(payload, TokenPayload)
        assert payload.sub == "dr.sharma"
        assert payload.token_type == "access"
        assert payload.exp > time.time()

    def test_create_and_decode_refresh_token(self):
        token = create_refresh_token("dr.patel")
        payload = decode_token(token)
        assert payload.sub == "dr.patel"
        assert payload.token_type == "refresh"
        assert payload.exp > time.time()

    def test_decode_invalid_token_raises_401(self):
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            decode_token("invalid.token.payload")
        assert exc_info.value.status_code == 401
        assert "Invalid token" in exc_info.value.detail

    def test_decode_expired_token_raises_401(self):
        import jwt
        from fastapi import HTTPException

        expired_payload = {
            "sub": "dr.sharma",
            "iat": time.time() - 3600,
            "exp": time.time() - 100,
            "token_type": "access",
        }
        expired_token = jwt.encode(
            expired_payload,
            mobile_settings.jwt_secret,
            algorithm=mobile_settings.jwt_algorithm,
        )

        with pytest.raises(HTTPException) as exc_info:
            decode_token(expired_token)
        assert exc_info.value.status_code == 401
        assert "expired" in exc_info.value.detail.lower()

    def test_validate_credentials_demo_mode(self):
        # Demo mode allows any password for demo_username
        assert validate_credentials("dr.sharma", "any_password_123") is True
        assert validate_credentials("unknown_user", "password") is False


# ═════════════════════════════════════════════════════════════════════════
# 2. Auth Endpoints (/auth/login, /auth/refresh)
# ═════════════════════════════════════════════════════════════════════════


class TestAuthEndpoints:
    """Tests for /auth/login and /auth/refresh HTTP routes."""

    def test_login_success_returns_jwt_pair(self, client):
        response = client.post(
            "/auth/login",
            json={"username": "dr.sharma", "password": "password123"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        assert data["expires_in"] > 0

    def test_login_invalid_user_returns_401(self, client):
        response = client.post(
            "/auth/login",
            json={"username": "dr.unknown", "password": "wrongpassword"},
        )
        assert response.status_code == 401
        assert "Invalid username or password" in response.json()["detail"]

    def test_refresh_token_success(self, client):
        refresh_tok = create_refresh_token("dr.sharma")
        response = client.post(
            "/auth/refresh",
            json={"refresh_token": refresh_tok},
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data

    def test_refresh_token_rejects_access_token(self, client):
        access_tok = create_access_token("dr.sharma")
        response = client.post(
            "/auth/refresh",
            json={"refresh_token": access_tok},
        )
        assert response.status_code == 400
        assert "expected refresh token" in response.json()["detail"]


# ═════════════════════════════════════════════════════════════════════════
# 3. Push Notification & Device Registry Tests
# ═════════════════════════════════════════════════════════════════════════


class TestPushNotifications:
    """Tests for FCM device registration, persistence, and alert dispatch."""

    def test_device_registry_crud(self, tmp_path: Path):
        registry_file = tmp_path / "test_registry.json"
        reg = DeviceRegistry(registry_path=registry_file)

        assert reg.device_count() == 0

        # Register device
        info = reg.register(DeviceRegistration(
            device_token="fcm_token_sample_1234567890",
            device_type="android",
            physician_id="dr.sharma",
        ))
        assert info.device_token == "fcm_token_sample_1234567890"
        assert reg.device_count() == 1

        # Retrieve by physician
        devices = reg.get_devices_for_physician("dr.sharma")
        assert len(devices) == 1
        assert devices[0].device_type == "android"

        # Persistence: reload from disk
        reg_reloaded = DeviceRegistry(registry_path=registry_file)
        assert reg_reloaded.device_count() == 1
        assert len(reg_reloaded.get_devices_for_physician("dr.sharma")) == 1

        # Unregister
        assert reg_reloaded.unregister("fcm_token_sample_1234567890") is True
        assert reg_reloaded.device_count() == 0
        assert reg_reloaded.unregister("non_existent_token") is False

    @pytest.mark.asyncio
    async def test_simulated_push_send(self, tmp_path: Path):
        registry_file = tmp_path / "test_registry.json"
        reg = DeviceRegistry(registry_path=registry_file)
        reg.register(DeviceRegistration(
            device_token="fcm_token_sample_sim_test_123",
            device_type="ios",
            physician_id="dr.sharma",
        ))

        service = PushNotificationService(device_registry=reg)
        result = await service.send_to_device(
            "fcm_token_sample_sim_test_123",
            PushPayload(
                title="Critical Alert",
                body="Patient deteriorating",
                patient_id=101,
                risk_level="CRITICAL",
                deterioration_probability=0.88,
            ),
        )
        assert result.success is True
        assert result.message_id.startswith("sim-")

        # Verify device stats were updated
        device = reg.get_devices_for_physician("dr.sharma")[0]
        assert device.push_count == 1
        assert device.last_push_at is not None

    @pytest.mark.asyncio
    async def test_send_risk_alert_multiple_devices(self, tmp_path: Path):
        registry_file = tmp_path / "test_registry.json"
        reg = DeviceRegistry(registry_path=registry_file)
        reg.register(DeviceRegistration(
            device_token="token_phone_123456789",
            device_type="android",
            physician_id="dr.sharma",
        ))
        reg.register(DeviceRegistration(
            device_token="token_tablet_123456789",
            device_type="ios",
            physician_id="dr.sharma",
        ))

        service = PushNotificationService(device_registry=reg)
        results = await service.send_risk_alert(
            physician_id="dr.sharma",
            patient_id=42,
            risk_level="CRITICAL",
            probability=0.92,
        )
        assert len(results) == 2
        assert all(r.success for r in results)


# ═════════════════════════════════════════════════════════════════════════
# 4. Mobile Endpoints (/mobile/watchlist, /mobile/patient/{id})
# ═════════════════════════════════════════════════════════════════════════


class TestMobileEndpoints:
    """Tests for mobile watchlist and patient summary endpoints."""

    def test_watchlist_empty(self, client):
        response = client.get("/mobile/watchlist")
        assert response.status_code == 200
        data = response.json()
        assert data["total_patients"] == 0
        assert data["patients"] == []
        assert data["critical_count"] == 0
        assert data["high_count"] == 0

    def test_watchlist_sorted_by_risk_score(self, client):
        # Insert 3 patients with different risk levels
        update_patient_risk(
            patient_id=101,
            risk_level="LOW",
            risk_score=15,
            probability=0.15,
        )
        update_patient_risk(
            patient_id=102,
            risk_level="CRITICAL",
            risk_score=92,
            probability=0.92,
            alerts=["CRITICAL: Severe Sepsis Alert"],
        )
        update_patient_risk(
            patient_id=103,
            risk_level="HIGH",
            risk_score=68,
            probability=0.68,
            alerts=["HIGH: Elevated Lactate"],
        )

        response = client.get("/mobile/watchlist")
        assert response.status_code == 200
        data = response.json()
        assert data["total_patients"] == 3
        assert data["critical_count"] == 1
        assert data["high_count"] == 1

        # Highest risk first
        scores = [p["risk_score"] for p in data["patients"]]
        assert scores == [92, 68, 15]
        assert data["patients"][0]["patient_id"] == 102
        assert data["patients"][0]["primary_concern"] == "CRITICAL: Severe Sepsis Alert"

    def test_patient_detail_success(self, client):
        update_patient_risk(
            patient_id=201,
            risk_level="HIGH",
            risk_score=72,
            probability=0.72,
            vitals={
                "heart_rate": 115.0,
                "respiratory_rate": 26.0,
                "spo2_pct": 91.0,
                "temperature_c": 38.9,
                "systolic_bp": 95.0,
                "diastolic_bp": 60.0,
            },
            labs={
                "wbc_count": 16.5,
                "lactate": 3.2,
                "creatinine": 1.8,
            },
            alerts=["Tachycardia", "Hypoxemia"],
        )

        response = client.get("/mobile/patient/201")
        assert response.status_code == 200
        data = response.json()
        assert data["patient_id"] == 201
        assert data["risk_level"] == "HIGH"
        assert data["risk_score"] == 72
        assert data["vitals"]["heart_rate"] == 115.0
        assert data["labs"]["lactate"] == 3.2
        assert len(data["alerts"]) == 2

    def test_patient_detail_not_found_returns_404(self, client):
        response = client.get("/mobile/patient/99999")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


# ═════════════════════════════════════════════════════════════════════════
# 5. Notification Router Endpoints
# ═════════════════════════════════════════════════════════════════════════


class TestNotificationEndpoints:
    """Tests for device registration and push endpoints via HTTP."""

    def test_register_device_endpoint(self, client):
        response = client.post(
            "/notifications/register",
            json={
                "device_token": "fcm_test_token_endpoint_999",
                "device_type": "android",
                "physician_id": "dr.sharma",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "registered"
        assert data["device_token"] == "fcm_test_token_endpoint_999"

    def test_register_device_invalid_payload(self, client):
        response = client.post(
            "/notifications/register",
            json={
                "device_token": "short",  # min_length=10 violated
                "device_type": "windows_phone",  # pattern violated
                "physician_id": "dr.sharma",
            },
        )
        assert response.status_code == 422

    def test_test_notification_endpoint(self, client):
        response = client.post(
            "/notifications/test?device_token=fcm_test_token_endpoint_999",
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_list_and_delete_devices_endpoint(self, client):
        # Register a device
        client.post(
            "/notifications/register",
            json={
                "device_token": "token_to_be_deleted_12345",
                "device_type": "ios",
                "physician_id": "dr.sharma",
            },
        )

        # List devices
        list_resp = client.get("/notifications/devices")
        assert list_resp.status_code == 200
        assert list_resp.json()["total_devices"] >= 1

        # Delete device
        del_resp = client.delete("/notifications/token_to_be_deleted_12345")
        assert del_resp.status_code == 200
        assert del_resp.json()["status"] == "unregistered"

        # Delete non-existent device
        del_404 = client.delete("/notifications/non_existent_token_123")
        assert del_404.status_code == 404


# ═════════════════════════════════════════════════════════════════════════
# 6. Auth Protection Feature Flag Tests
# ═════════════════════════════════════════════════════════════════════════


class TestAuthProtectionFlag:
    """Verify behavior when auth_enabled is True vs False."""

    def test_endpoints_pass_freely_when_auth_disabled(self, client):
        with patch.object(mobile_settings, "auth_enabled", False):
            response = client.get("/mobile/watchlist")
            assert response.status_code == 200

    def test_endpoints_require_token_when_auth_enabled(self, client):
        with patch.object(mobile_settings, "auth_enabled", True):
            # 1. No token -> 401
            resp_no_token = client.get("/mobile/watchlist")
            assert resp_no_token.status_code == 401

            # 2. Invalid token -> 401
            resp_bad_token = client.get(
                "/mobile/watchlist",
                headers={"Authorization": "Bearer totally_bogus_token"},
            )
            assert resp_bad_token.status_code == 401

            # 3. Valid access token -> 200
            token = create_access_token("dr.sharma")
            resp_good_token = client.get(
                "/mobile/watchlist",
                headers={"Authorization": f"Bearer {token}"},
            )
            assert resp_good_token.status_code == 200


# ═════════════════════════════════════════════════════════════════════════
# 7. End-to-End: Predict feeds Mobile Watchlist
# ═════════════════════════════════════════════════════════════════════════


class TestPredictionFeedsMobileWatchlist:
    """Test that predictions on /predict automatically update the mobile store."""

    def test_prediction_populates_watchlist(self, client):
        predict_payload = {
            "patient_id": 555,
            "vitals": {
                "heart_rate": 105.0,
                "respiratory_rate": 24.0,
                "spo2_pct": 93.0,
                "temperature_c": 38.5,
                "systolic_bp": 100.0,
                "diastolic_bp": 65.0,
                "oxygen_flow": 2.0,
                "mobility_score": 2,
                "nurse_alert": 1,
            },
            "labs": {
                "wbc_count": 14.0,
                "lactate": 2.8,
                "creatinine": 1.5,
                "crp_level": 45.0,
                "hemoglobin": 11.0,
                "sepsis_risk_score": 0.6,
            },
            "model_type": "xgboost",
        }

        import numpy as np

        from src.serving.model_server import registry

        # Mock model prediction since physical model weights may not be trained in test env
        with patch.object(registry, "predict", return_value=np.array([0.72])):
            pred_resp = client.post("/predict", json=predict_payload)
            assert pred_resp.status_code == 200
            pred_data = pred_resp.json()

        # Check mobile watchlist
        watch_resp = client.get("/mobile/watchlist")
        assert watch_resp.status_code == 200
        watchlist = watch_resp.json()["patients"]

        patient_entry = next((p for p in watchlist if p["patient_id"] == 555), None)
        assert patient_entry is not None
        assert patient_entry["risk_level"] == pred_data["risk_level"]
        assert patient_entry["risk_score"] == pred_data["risk_score"]

        # Check mobile patient detail
        detail_resp = client.get("/mobile/patient/555")
        assert detail_resp.status_code == 200
        detail_data = detail_resp.json()
        assert detail_data["patient_id"] == 555
        assert detail_data["vitals"]["heart_rate"] == 105.0
        assert detail_data["labs"]["lactate"] == 2.8
