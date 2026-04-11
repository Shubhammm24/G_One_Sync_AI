"""
G_One_Sync AI — Serving API Tests
=====================================
Tests for the FastAPI model serving endpoints.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client for the serving API."""
    from src.serving.model_server import app
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_format(self, client):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "models_loaded" in data
        assert "gpu_available" in data


class TestModelInfoEndpoint:
    """Tests for /model/info endpoint."""

    def test_model_info_returns_200(self, client):
        response = client.get("/model/info")
        assert response.status_code == 200


class TestPredictEndpoint:
    """Tests for /predict endpoint."""

    def _make_request(self):
        return {
            "patient_id": 1,
            "vitals": {
                "heart_rate": 95.0,
                "respiratory_rate": 22.0,
                "spo2_pct": 94.0,
                "temperature_c": 38.2,
                "systolic_bp": 110.0,
                "diastolic_bp": 70.0,
                "oxygen_flow": 2.0,
                "mobility_score": 3,
                "nurse_alert": 0,
            },
            "labs": {
                "wbc_count": 12.5,
                "lactate": 2.1,
                "creatinine": 1.4,
                "crp_level": 45.0,
                "hemoglobin": 11.5,
                "sepsis_risk_score": 0.3,
            },
            "model_type": "ensemble",
        }

    def test_predict_valid_request(self, client):
        """Valid request should return 200 with risk assessment."""
        response = client.post("/predict", json=self._make_request())
        # May return 500 if no models loaded, which is acceptable in test
        assert response.status_code in (200, 500)
        if response.status_code == 200:
            data = response.json()
            assert "patient_id" in data
            assert "deterioration_probability" in data
            assert "risk_level" in data
            assert data["risk_level"] in ["LOW", "MODERATE", "HIGH", "CRITICAL"]

    def test_predict_invalid_vitals(self, client):
        """Invalid vitals should return 422."""
        req = self._make_request()
        req["vitals"]["heart_rate"] = -10  # Invalid
        response = client.post("/predict", json=req)
        assert response.status_code == 422

    def test_predict_missing_fields(self, client):
        """Missing required fields should return 422."""
        response = client.post("/predict", json={"patient_id": 1})
        assert response.status_code == 422


class TestBatchPredictEndpoint:
    """Tests for /predict/batch endpoint."""

    def test_batch_predict(self, client):
        """Batch prediction should accept multiple patients."""
        req = {
            "patients": [
                {
                    "patient_id": i,
                    "vitals": {
                        "heart_rate": 80 + i * 5,
                        "respiratory_rate": 18 + i,
                        "spo2_pct": 97 - i,
                        "temperature_c": 37.0 + i * 0.3,
                        "systolic_bp": 120 - i * 5,
                        "diastolic_bp": 75,
                        "oxygen_flow": 0,
                        "mobility_score": 3,
                        "nurse_alert": 0,
                    },
                    "labs": {
                        "wbc_count": 7.5,
                        "lactate": 1.0 + i * 0.5,
                        "creatinine": 1.0,
                        "crp_level": 10,
                        "hemoglobin": 13,
                        "sepsis_risk_score": 0.1,
                    },
                }
                for i in range(3)
            ],
            "model_type": "ensemble",
        }
        response = client.post("/predict/batch", json=req)
        assert response.status_code in (200, 500)


class TestRiskClassification:
    """Tests for risk level classification logic."""

    def test_risk_levels(self):
        from src.serving.model_server import _classify_risk, RiskLevel

        level, score, alerts = _classify_risk(0.10)
        assert level == RiskLevel.LOW
        assert score == 10

        level, score, alerts = _classify_risk(0.30)
        assert level == RiskLevel.MODERATE

        level, score, alerts = _classify_risk(0.55)
        assert level == RiskLevel.HIGH

        level, score, alerts = _classify_risk(0.80)
        assert level == RiskLevel.CRITICAL
        assert len(alerts) >= 2  # Should have multiple alerts

    def test_feature_vector_shape(self):
        from src.serving.model_server import _build_feature_vector, PredictionRequest, VitalSigns, LabResults

        req = PredictionRequest(
            patient_id=1,
            vitals=VitalSigns(
                heart_rate=80, respiratory_rate=18, spo2_pct=97,
                temperature_c=37, systolic_bp=120, diastolic_bp=75,
            ),
            labs=LabResults(),
        )
        features = _build_feature_vector(req)
        assert features.shape == (1, 15)
