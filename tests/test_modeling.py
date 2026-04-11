"""
G_One_Sync AI — Modeling Unit Tests
======================================
Tests for loss functions, calibration, threshold optimization,
model forward passes, and ensemble predictions.
"""

import numpy as np
import pytest
import torch


# ── Focal Loss Tests ─────────────────────────────────────────────────


class TestFocalLoss:
    """Tests for the Focal Loss implementation."""

    def test_output_shape(self):
        """Focal loss should return a scalar."""
        from src.modeling.losses import FocalLoss
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        logits = torch.randn(32)
        targets = torch.randint(0, 2, (32,)).float()
        loss = criterion(logits, targets)
        assert loss.ndim == 0, "Loss should be a scalar"

    def test_positive_loss(self):
        """Focal loss should always be non-negative."""
        from src.modeling.losses import FocalLoss
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        for _ in range(10):
            logits = torch.randn(64)
            targets = torch.randint(0, 2, (64,)).float()
            loss = criterion(logits, targets)
            assert loss.item() >= 0, "Loss must be non-negative"

    def test_gamma_zero_equals_bce(self):
        """gamma=0 should approximate standard BCE."""
        from src.modeling.losses import FocalLoss
        focal = FocalLoss(alpha=0.5, gamma=0.0)
        bce = torch.nn.BCEWithLogitsLoss()

        logits = torch.randn(100)
        targets = torch.randint(0, 2, (100,)).float()

        focal_loss = focal(logits, targets).item()
        bce_loss = bce(logits, targets).item()

        # Should be in similar range (not exact due to alpha=0.5)
        assert abs(focal_loss - bce_loss * 0.5) < 0.5, \
            f"gamma=0 focal ({focal_loss}) should approximate 0.5*BCE ({bce_loss*0.5})"

    def test_hard_examples_higher_loss(self):
        """Hard-to-classify examples should have higher focal loss."""
        from src.modeling.losses import FocalLoss
        criterion = FocalLoss(alpha=0.5, gamma=2.0, reduction="none")

        # Easy example: high confidence correct prediction
        easy_logit = torch.tensor([5.0])  # sigmoid ≈ 0.99
        easy_target = torch.tensor([1.0])

        # Hard example: low confidence correct prediction
        hard_logit = torch.tensor([0.1])  # sigmoid ≈ 0.525
        hard_target = torch.tensor([1.0])

        easy_loss = criterion(easy_logit, easy_target).item()
        hard_loss = criterion(hard_logit, hard_target).item()

        assert hard_loss > easy_loss, "Hard examples should have higher focal loss"


class TestLabelSmoothingBCE:
    """Tests for label smoothing BCE."""

    def test_smoothed_targets(self):
        """Smoothing should move targets away from 0 and 1."""
        from src.modeling.losses import LabelSmoothingBCE
        criterion = LabelSmoothingBCE(smoothing=0.1)
        logits = torch.randn(32)
        targets = torch.ones(32)
        loss = criterion(logits, targets)
        assert loss.item() > 0


# ── Calibration Tests ────────────────────────────────────────────────


class TestProbabilityCalibrator:
    """Tests for probability calibration."""

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 500)
        y_prob = np.clip(y_true * 0.7 + np.random.normal(0, 0.2, 500), 0.01, 0.99)
        return y_prob, y_true

    def test_isotonic_fit_transform(self, sample_data):
        """Isotonic calibration should produce valid probabilities."""
        from src.modeling.calibration import ProbabilityCalibrator
        y_prob, y_true = sample_data
        cal = ProbabilityCalibrator(method="isotonic")
        cal.fit(y_prob, y_true)

        calibrated = cal.transform(y_prob)
        assert len(calibrated) == len(y_prob)
        assert np.all(calibrated >= 0) and np.all(calibrated <= 1)

    def test_platt_fit_transform(self, sample_data):
        """Platt scaling should produce valid probabilities."""
        from src.modeling.calibration import ProbabilityCalibrator
        y_prob, y_true = sample_data
        cal = ProbabilityCalibrator(method="platt")
        cal.fit(y_prob, y_true)

        calibrated = cal.transform(y_prob)
        assert len(calibrated) == len(y_prob)
        assert np.all(calibrated >= 0) and np.all(calibrated <= 1)

    def test_brier_score_improves(self, sample_data):
        """Calibration should improve (reduce) Brier score."""
        from src.modeling.calibration import ProbabilityCalibrator
        y_prob, y_true = sample_data
        cal = ProbabilityCalibrator(method="isotonic")
        cal.fit(y_prob, y_true)

        assert cal.calibration_stats["brier_improvement"] > 0, \
            "Calibration should improve Brier score"


class TestThresholdOptimizer:
    """Tests for threshold optimization."""

    def test_find_thresholds(self):
        """Should find valid thresholds for all strategies."""
        from src.modeling.calibration import ThresholdOptimizer
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 500)
        y_prob = np.clip(y_true * 0.6 + np.random.normal(0, 0.3, 500), 0.01, 0.99)

        optimizer = ThresholdOptimizer()
        results = optimizer.optimize(y_prob, y_true, model_name="test")

        assert "f1" in results
        assert "youden" in results
        assert "clinical" in results
        assert all(0 < t < 1 for t in results.values()), \
            "All thresholds should be between 0 and 1"


# ── Model Smoke Tests ────────────────────────────────────────────────


class TestModelForwardPass:
    """Smoke tests for model forward passes (CPU only)."""

    def test_bilstm_forward(self):
        """BiLSTM + Attention forward pass should produce logits + weights."""
        from src.modeling.lstm_trainer import BiLSTMAttentionModel
        model = BiLSTMAttentionModel(input_size=11, hidden_size=32, num_layers=1)
        x = torch.randn(4, 12, 11)  # (batch=4, seq=12, features=11)
        logits, attn = model(x)

        assert logits.shape == (4,), f"Expected (4,), got {logits.shape}"
        assert attn.shape == (4, 12), f"Expected (4, 12), got {attn.shape}"
        assert torch.allclose(attn.sum(dim=1), torch.ones(4), atol=1e-5), \
            "Attention weights should sum to 1"

    def test_transformer_forward(self):
        """Transformer forward pass should produce logits."""
        from src.modeling.transformer_trainer import TemporalTransformerModel
        model = TemporalTransformerModel(
            input_size=11, d_model=32, nhead=4, num_layers=1
        )
        x = torch.randn(4, 12, 11)
        logits = model(x)

        assert logits.shape == (4,), f"Expected (4,), got {logits.shape}"

    def test_ensemble_equal_weights(self):
        """Ensemble with equal weights should average predictions."""
        from src.modeling.ensemble import HybridEnsemble
        ensemble = HybridEnsemble(strategy="equal")

        val_preds = {
            "model_a": np.array([0.8, 0.2, 0.5]),
            "model_b": np.array([0.6, 0.4, 0.7]),
        }
        val_labels = np.array([1, 0, 1])

        ensemble.fit(val_preds, val_labels)
        result = ensemble.predict(val_preds)

        expected = np.array([0.7, 0.3, 0.6])
        np.testing.assert_allclose(result, expected, atol=1e-6)


# ── Monitoring Tests ─────────────────────────────────────────────────


class TestDriftDetector:
    """Tests for drift detection."""

    def test_no_drift_same_distribution(self):
        """PSI should be low for same distribution."""
        from src.monitoring.drift_detector import DriftDetector
        np.random.seed(42)
        data = np.random.randn(2000, 5)

        detector = DriftDetector(reference_data=data, n_bins=10)
        for row in data[:800]:
            detector.add_live_sample(row)

        psi = detector.compute_psi(0)
        assert psi < 0.2, f"PSI should be < 0.2 for same distribution, got {psi}"

    def test_drift_detected_shifted_distribution(self):
        """PSI should be high for shifted distribution."""
        from src.monitoring.drift_detector import DriftDetector
        np.random.seed(42)
        ref_data = np.random.randn(1000, 5)
        live_data = np.random.randn(500, 5) + 3  # Shift mean by 3

        detector = DriftDetector(reference_data=ref_data, n_bins=10)
        for row in live_data:
            detector.add_live_sample(row)

        psi = detector.compute_psi(0)
        assert psi > 0.25, f"PSI should be > 0.25 for shifted distribution, got {psi}"


class TestAlertingEngine:
    """Tests for the alerting engine."""

    def test_critical_alert_fires(self):
        """Critical risk alert should fire for high probability."""
        from src.monitoring.alerting import AlertingEngine
        engine = AlertingEngine()
        alerts = engine.evaluate_prediction(patient_id=42, probability=0.85)
        assert len(alerts) > 0
        assert alerts[0].severity.value == "CRITICAL"

    def test_cooldown_prevents_repeat(self):
        """Same alert should not fire within cooldown period."""
        from src.monitoring.alerting import AlertingEngine
        engine = AlertingEngine()
        alerts1 = engine.evaluate_prediction(patient_id=42, probability=0.85)
        alerts2 = engine.evaluate_prediction(patient_id=42, probability=0.85)
        assert len(alerts1) > 0
        assert len(alerts2) == 0, "Cooldown should prevent repeat alerts"
