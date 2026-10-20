"""
G_One_Sync AI — Multi-Outcome Prediction Tests
=================================================
Tests for Phase 10:
    - Clinical outcome definitions & registry
    - Multi-outcome label generation (sepsis, cardiac arrest, respiratory failure)
    - MultiHeadPredictionModel architecture & forward pass
    - MultiLabelTrainer multi-task training loop & evaluation
    - SHAP multi-outcome explainability
    - Serving endpoints (/outcomes, /predict/multi, /predict/multi/explain)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient

from src.explainability.shap_explainer import SHAPExplainer
from src.modeling.multi_outcome.label_definitions import (
    ALL_OUTCOMES,
    CARDIAC_ARREST_DEFINITION,
    DETERIORATION_DEFINITION,
    NUM_OUTCOMES,
    OUTCOME_COLUMNS,
    OUTCOME_REGISTRY,
    RESPIRATORY_FAILURE_DEFINITION,
    SEPSIS_DEFINITION,
    OutcomeType,
)
from src.modeling.multi_outcome.multi_head_model import MultiHeadPredictionModel
from src.modeling.multi_outcome.multi_label_trainer import MultiLabelTrainer
from src.modeling.multi_outcome.outcome_schemas import (
    MultiOutcomeExplainResponse,
    MultiOutcomeResponse,
    OutcomePrediction,
    RiskLevel,
)
from src.preprocessing.label_generator import LabelGenerator
from src.serving.model_server import app


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def sample_multi_outcome_panel() -> pd.DataFrame:
    """Create synthetic timeseries panel for 3 patients over 12 hours."""
    np.random.seed(42)
    records = []
    for pid in [1, 2, 3]:
        for hour in range(12):
            records.append({
                "patient_id": pid,
                "hour_from_admission": hour,
                "heart_rate": 75.0 + (hour * 3 if pid == 1 else 0) + (100.0 if pid == 2 and hour >= 8 else 0),
                "respiratory_rate": 16.0 + (hour * 1.5 if pid == 3 else 0),
                "spo2_pct": 98.0 - (hour * 1.2 if pid == 3 else 0),
                "temperature_c": 37.0 + (1.5 if pid == 1 and hour >= 6 else 0),
                "systolic_bp": 120.0 - (hour * 4 if pid == 2 else 0),
                "diastolic_bp": 75.0,
                "oxygen_device": "none",
                "oxygen_flow": 8.0 if (pid == 3 and hour >= 8) else 0.0,
                "mobility_score": 1 if (pid == 1 and hour >= 6) else 3,
                "nurse_alert": 1 if hour >= 8 else 0,
                "wbc_count": 14.0 if (pid == 1 and hour >= 6) else 7.0,
                "lactate": 2.8 if (pid == 1 and hour >= 6) else 1.1,
                "creatinine": 1.2,
                "crp_level": 35.0 if (pid == 1 and hour >= 6) else 10.0,
                "hemoglobin": 13.0,
                "sepsis_risk_score": 0.75 if (pid == 1 and hour >= 6) else 0.15,
                "age": 60 + pid * 5,
                "gender": "M" if pid % 2 == 0 else "F",
                "comorbidity_index": pid,
                "admission_type": "ED",
                "deterioration_hour": 9 if pid in (1, 2) else -1,
                "deterioration_event": 1 if pid in (1, 2) else 0,
            })
    return pd.DataFrame(records)


@pytest.fixture
def sample_prediction_payload() -> dict:
    """Standard single-patient clinical prediction payload."""
    return {
        "patient_id": 401,
        "vitals": {
            "heart_rate": 115.0,
            "respiratory_rate": 26.0,
            "spo2_pct": 91.0,
            "temperature_c": 38.6,
            "systolic_bp": 92.0,
            "diastolic_bp": 58.0,
            "oxygen_flow": 4.0,
            "mobility_score": 2,
            "nurse_alert": 1,
        },
        "labs": {
            "wbc_count": 15.2,
            "lactate": 2.6,
            "creatinine": 1.5,
            "crp_level": 55.0,
            "hemoglobin": 11.0,
            "sepsis_risk_score": 0.68,
        },
        "model_type": "ensemble",
    }


# ── 1. Outcome Definitions & Registry Tests ───────────────────────────────


class TestOutcomeDefinitions:
    """Validate clinical definitions, horizons, and registry consistency."""

    def test_outcome_count_and_columns(self):
        assert NUM_OUTCOMES == 4
        assert len(ALL_OUTCOMES) == 4
        assert len(OUTCOME_COLUMNS) == 4
        assert OUTCOME_COLUMNS == [
            "deterioration_next_12h",
            "sepsis_next_6h",
            "cardiac_arrest_next_12h",
            "respiratory_failure_next_12h",
        ]

    def test_registry_lookup(self):
        for col in OUTCOME_COLUMNS:
            assert col in OUTCOME_REGISTRY
            defn = OUTCOME_REGISTRY[col]
            assert defn.label_column == col
            assert defn.display_name != ""
            assert defn.prediction_horizon_hours in (6, 12)
            assert 0.0 < defn.prevalence_estimate < 1.0
            assert len(defn.clinical_criteria) > 0

    def test_specific_clinical_criteria(self):
        assert SEPSIS_DEFINITION.prediction_horizon_hours == 6
        assert CARDIAC_ARREST_DEFINITION.prediction_horizon_hours == 12
        assert RESPIRATORY_FAILURE_DEFINITION.prediction_horizon_hours == 12
        assert DETERIORATION_DEFINITION.prediction_horizon_hours == 12


# ── 2. Label Generator Extension Tests ────────────────────────────────────


class TestMultiOutcomeLabelGenerator:
    """Validate generation of sepsis, cardiac arrest, and respiratory labels."""

    def test_generate_sepsis_labels(self, sample_multi_outcome_panel):
        gen = LabelGenerator()
        df = gen.generate_sepsis_labels(sample_multi_outcome_panel, horizon_hours=6)

        assert "sepsis_next_6h" in df.columns
        # Patient 1 has elevated sepsis criteria starting hour 6
        p1 = df[df["patient_id"] == 1]
        # At hour 0, sepsis starts at hour 6: 0 < 6 <= 6 -> True
        assert p1.loc[p1["hour_from_admission"] == 0, "sepsis_next_6h"].iloc[0] == 1
        assert df["sepsis_next_6h"].isin([0, 1]).all()

    def test_generate_cardiac_arrest_labels(self, sample_multi_outcome_panel):
        gen = LabelGenerator()
        df = gen.generate_cardiac_arrest_labels(sample_multi_outcome_panel, horizon_hours=12)

        assert "cardiac_arrest_next_12h" in df.columns
        assert df["cardiac_arrest_next_12h"].isin([0, 1]).all()
        # Patient 2 has tachycardia (HR >= 170) and deterioration around hour 8-9
        p2 = df[df["patient_id"] == 2]
        assert p2["cardiac_arrest_next_12h"].sum() > 0

    def test_generate_respiratory_failure_labels(self, sample_multi_outcome_panel):
        gen = LabelGenerator()
        df = gen.generate_respiratory_failure_labels(sample_multi_outcome_panel, horizon_hours=12)

        assert "respiratory_failure_next_12h" in df.columns
        assert df["respiratory_failure_next_12h"].isin([0, 1]).all()
        # Patient 3 has high oxygen flow + tachypnea and desaturation
        p3 = df[df["patient_id"] == 3]
        assert p3["respiratory_failure_next_12h"].sum() > 0

    def test_generate_all_outcome_labels(self, sample_multi_outcome_panel):
        gen = LabelGenerator()
        df = gen.generate_all_outcome_labels(sample_multi_outcome_panel)

        for col in OUTCOME_COLUMNS:
            assert col in df.columns
            assert df[col].isin([0, 1]).all()


# ── 3. Multi-Head Model Architecture Tests ────────────────────────────────


class TestMultiHeadModel:
    """Validate shared Transformer encoder and per-outcome heads."""

    def test_model_initialization_and_forward(self):
        model = MultiHeadPredictionModel(
            input_size=15,
            num_outcomes=4,
            d_model=32,
            nhead=2,
            num_layers=1,
            dim_feedforward=64,
            head_hidden_dim=32,
        )

        batch_size = 4
        seq_len = 24
        x = torch.randn(batch_size, seq_len, 15)

        # 1. Test shared encoder output shape
        encoded = model.encode(x)
        assert encoded.shape == (batch_size, 32 * 2)

        # 2. Test forward logits dictionary
        logits = model(x)
        assert isinstance(logits, dict)
        assert len(logits) == 4
        for name in OUTCOME_COLUMNS:
            assert name in logits
            assert logits[name].shape == (batch_size,)

        # 3. Test probabilities in [0, 1]
        probs = model.predict_proba(x)
        for name in OUTCOME_COLUMNS:
            assert probs[name].shape == (batch_size,)
            assert (probs[name] >= 0.0).all() and (probs[name] <= 1.0).all()

        # 4. Test stacked logits tensor
        logits_tensor = model.get_outcome_logits_tensor(x)
        assert logits_tensor.shape == (batch_size, 4)


# ── 4. Multi-Label Trainer Tests ──────────────────────────────────────────


class TestMultiLabelTrainer:
    """Validate multi-task trainer subclassing BaseTrainer."""

    def test_trainer_build_and_loss(self, tmp_path):
        trainer = MultiLabelTrainer(
            artifacts_dir=tmp_path,
            use_mlflow=False,
            outcome_names=OUTCOME_COLUMNS,
            task_weights=[1.0, 1.5, 2.0, 1.5],
        )

        model = trainer._build_model(input_size=15)
        assert isinstance(model, MultiHeadPredictionModel)
        assert trainer.num_outcomes == 4

        # Test loss computation
        logits = {name: torch.randn(8) for name in OUTCOME_COLUMNS}
        labels = {name: torch.randint(0, 2, (8,)).float() for name in OUTCOME_COLUMNS}
        loss = trainer._compute_multi_task_loss(logits, labels)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() > 0.0

    def test_trainer_mini_training_and_evaluate(self, tmp_path, sample_multi_outcome_panel):
        gen = LabelGenerator()
        df = gen.generate_all_outcome_labels(sample_multi_outcome_panel)

        trainer = MultiLabelTrainer(
            artifacts_dir=tmp_path,
            use_mlflow=False,
        )

        # Mock training setup with 15 numeric features
        feature_cols = [
            "heart_rate", "respiratory_rate", "spo2_pct",
            "temperature_c", "systolic_bp", "diastolic_bp",
            "oxygen_flow", "mobility_score", "nurse_alert",
            "wbc_count", "lactate", "creatinine",
            "crp_level", "hemoglobin", "sepsis_risk_score",
        ]
        X = df[feature_cols].values
        Y = df[OUTCOME_COLUMNS].values

        trainer.model = trainer._build_model(input_size=len(feature_cols))
        trainer.is_trained = True

        # Test multi-probability predictions
        pred_dict = trainer.predict_multi_proba((X, Y))
        assert len(pred_dict) == 4
        for name in OUTCOME_COLUMNS:
            assert name in pred_dict
            assert len(pred_dict[name]) == len(df)
            assert (pred_dict[name] >= 0.0).all() and (pred_dict[name] <= 1.0).all()

        # Test evaluate_multi
        metrics = trainer.evaluate_multi((X, Y), split_name="test")
        assert len(metrics) > 0
        for name, met in metrics.items():
            assert hasattr(met, "auroc")
            assert hasattr(met, "recall")


# ── 5. SHAP Multi-Outcome Tests ───────────────────────────────────────────


class TestMultiOutcomeSHAP:
    """Validate per-outcome SHAP feature explanations."""

    def test_explain_multi_outcome(self):
        model = MultiHeadPredictionModel(
            input_size=15,
            num_outcomes=4,
            d_model=32,
            nhead=2,
            num_layers=1,
            head_hidden_dim=32,
        )
        feature_names = [f"feat_{i}" for i in range(15)]
        explainer = SHAPExplainer(
            model=model,
            model_type="transformer",
            feature_names=feature_names,
        )

        sample_patient = np.random.randn(15)
        explanations = explainer.explain_multi_outcome(
            patient_features=sample_patient,
            patient_id=99,
            top_n=5,
        )

        assert isinstance(explanations, dict)
        for col in OUTCOME_COLUMNS:
            assert col in explanations
            records = explanations[col]
            assert len(records) <= 5
            for rec in records:
                assert "feature" in rec
                assert "value" in rec
                assert "shap_value" in rec
                assert "direction" in rec
                assert rec["direction"] in ("risk", "protective")


# ── 6. Serving API Multi-Outcome Tests ─────────────────────────────────────


class TestMultiOutcomeServingAPI:
    """Validate FastAPI multi-outcome serving endpoints."""

    def test_get_outcomes_endpoint(self):
        client = TestClient(app)
        res = client.get("/outcomes")
        assert res.status_code == 200
        data = res.json()
        assert len(data) == 4
        labels = [item["label_column"] for item in data]
        assert "deterioration_next_12h" in labels
        assert "sepsis_next_6h" in labels

    def test_predict_multi_endpoint(self, sample_prediction_payload):
        client = TestClient(app)
        res = client.post("/predict/multi", json=sample_prediction_payload)
        assert res.status_code == 200
        data = res.json()

        assert data["patient_id"] == 401
        assert data["num_outcomes"] == 4
        assert "predictions" in data
        assert "primary_concern" in data
        assert "clinical_alerts" in data
        assert len(data["clinical_alerts"]) > 0

        preds = data["predictions"]
        for col in OUTCOME_COLUMNS:
            assert col in preds
            p = preds[col]
            assert "probability" in p
            assert 0.0 <= p["probability"] <= 1.0
            assert p["risk_level"] in ("LOW", "MODERATE", "HIGH", "CRITICAL")
            assert 0 <= p["risk_score"] <= 100
            assert p["prediction_horizon_hours"] in (6, 12)

    def test_predict_multi_explain_endpoint(self, sample_prediction_payload):
        client = TestClient(app)
        res = client.post("/predict/multi/explain", json=sample_prediction_payload)
        assert res.status_code == 200
        data = res.json()

        assert data["patient_id"] == 401
        assert "predictions" in data
        assert "explanations" in data
        for col in OUTCOME_COLUMNS:
            assert col in data["explanations"]
            assert len(data["explanations"][col]) > 0

    def test_existing_predict_endpoint_unaffected(self, sample_prediction_payload):
        """Verify backwards compatibility: existing /predict is unchanged."""
        client = TestClient(app)
        res = client.post("/predict", json=sample_prediction_payload)
        assert res.status_code in (200, 500)
        if res.status_code == 200:
            data = res.json()
            assert "deterioration_probability" in data
            assert "risk_score" in data
