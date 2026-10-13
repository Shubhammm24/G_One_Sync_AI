"""
G_One_Sync AI — Continuous Learning Tests
============================================
Test suite for Phase 7 continuous learning:
- Model versioning (create, promote, rollback, cleanup)
- Data accumulation (buffer, flush, stats)
- Champion/Challenger evaluation (promotion criteria)
- Shadow evaluation (paired predictions, metrics)
- Retraining orchestrator (lifecycle, drift triggers)
- Drift detector extensions (should_retrain, get_drift_severity)
- ContinuousSettings (defaults, env prefix)
"""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from config.settings import ContinuousSettings
from src.continuous.champion_challenger import (
    ChampionChallengerEvaluator,
    ComparisonResult,
)
from src.continuous.data_accumulator import DataAccumulator
from src.continuous.model_versioner import ModelVersion, ModelVersionManager
from src.continuous.shadow_evaluator import ShadowEvaluator, ShadowPrediction


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test artifacts."""
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def versioner(temp_dir):
    """Create a ModelVersionManager with temp directory."""
    return ModelVersionManager(versions_dir=temp_dir / "versions")


@pytest.fixture
def accumulator(temp_dir):
    """Create a DataAccumulator with temp directory."""
    return DataAccumulator(buffer_dir=temp_dir / "buffer", min_samples=10)


@pytest.fixture
def shadow():
    """Create a ShadowEvaluator with short evaluation period."""
    return ShadowEvaluator(evaluation_hours=0)  # Immediate completion for tests


@pytest.fixture
def evaluator():
    """Create a ChampionChallengerEvaluator with test thresholds."""
    return ChampionChallengerEvaluator(
        auroc_threshold=0.005,
        auprc_threshold=0.0,
        min_evaluation_samples=5,
    )


def _make_version(
    version_id: str = "test-v1",
    auroc: float = 0.85,
    auprc: float = 0.70,
    **kwargs,
) -> ModelVersion:
    """Helper to create a ModelVersion for testing."""
    return ModelVersion(
        version_id=version_id,
        model_type="ensemble",
        auroc=auroc,
        auprc=auprc,
        **kwargs,
    )


# ═════════════════════════════════════════════════════════════════════════
# MODEL VERSIONER TESTS
# ═════════════════════════════════════════════════════════════════════════


class TestModelVersionManager:
    """Tests for model version management."""

    def test_create_version(self, versioner, temp_dir):
        """Creating a version should persist metadata."""
        artifacts = temp_dir / "artifacts"
        artifacts.mkdir()
        (artifacts / "model.pt").write_text("dummy")

        version = versioner.create_version(
            model_type="bilstm",
            artifacts_source=artifacts,
            metrics={"auroc": 0.88, "auprc": 0.75},
            training_trigger="drift",
            training_samples=5000,
        )

        assert version.version_id.startswith("bilstm-")
        assert version.auroc == 0.88
        assert version.auprc == 0.75
        assert version.training_trigger == "drift"
        assert version.version_id in versioner.versions

    def test_set_champion(self, versioner, temp_dir):
        """Setting champion should designate the correct version."""
        artifacts = temp_dir / "artifacts"
        artifacts.mkdir()

        v = versioner.create_version(
            model_type="xgboost",
            artifacts_source=artifacts,
            metrics={"auroc": 0.90},
        )
        versioner.set_champion(v.version_id)

        assert versioner.champion is not None
        assert versioner.champion.version_id == v.version_id
        assert versioner.champion.is_champion is True

    def test_only_one_champion(self, versioner, temp_dir):
        """Only one version should be champion at a time."""
        artifacts = temp_dir / "artifacts"
        artifacts.mkdir()

        v1 = versioner.create_version(
            model_type="xgboost", artifacts_source=artifacts, metrics={"auroc": 0.85},
        )
        v2 = versioner.create_version(
            model_type="xgboost", artifacts_source=artifacts, metrics={"auroc": 0.90},
        )

        versioner.set_champion(v1.version_id)
        versioner.set_champion(v2.version_id)

        assert versioner.champion.version_id == v2.version_id
        assert versioner.versions[v1.version_id].is_champion is False

    def test_promote_challenger(self, versioner, temp_dir):
        """Promoting challenger should make it champion and retire old."""
        artifacts = temp_dir / "artifacts"
        artifacts.mkdir()

        v1 = versioner.create_version(
            model_type="xgboost", artifacts_source=artifacts, metrics={"auroc": 0.85},
        )
        v2 = versioner.create_version(
            model_type="xgboost", artifacts_source=artifacts, metrics={"auroc": 0.90},
        )

        versioner.set_champion(v1.version_id)
        versioner.set_challenger(v2.version_id)
        result = versioner.promote_challenger(promoted_by="test_user")

        assert result is not None
        assert result.is_champion is True
        assert result.promoted_by == "test_user"
        assert versioner.versions[v1.version_id].is_retired is True

    def test_rollback(self, versioner, temp_dir):
        """Rollback should restore a previous version as champion."""
        artifacts = temp_dir / "artifacts"
        artifacts.mkdir()

        v1 = versioner.create_version(
            model_type="xgboost", artifacts_source=artifacts, metrics={"auroc": 0.85},
        )
        v2 = versioner.create_version(
            model_type="xgboost", artifacts_source=artifacts, metrics={"auroc": 0.90},
        )

        versioner.set_champion(v1.version_id)
        versioner.set_champion(v2.version_id)

        versioner.rollback(v1.version_id)
        assert versioner.champion.version_id == v1.version_id

    def test_discard_challenger(self, versioner, temp_dir):
        """Discarding challenger should retire it."""
        artifacts = temp_dir / "artifacts"
        artifacts.mkdir()

        v = versioner.create_version(
            model_type="xgboost", artifacts_source=artifacts, metrics={},
        )
        versioner.set_challenger(v.version_id)
        versioner.discard_challenger(reason="Underperformed")

        assert versioner.versions[v.version_id].is_retired is True
        assert versioner.challenger is None

    def test_list_versions(self, versioner, temp_dir):
        """Listing should filter retired versions by default."""
        artifacts = temp_dir / "artifacts"
        artifacts.mkdir()

        v1 = versioner.create_version(
            model_type="xgboost", artifacts_source=artifacts, metrics={},
        )
        v2 = versioner.create_version(
            model_type="xgboost", artifacts_source=artifacts, metrics={},
        )

        v1.is_retired = True
        active = versioner.list_versions()
        assert len(active) == 1

        all_versions = versioner.list_versions(include_retired=True)
        assert len(all_versions) == 2

    def test_registry_persistence(self, temp_dir):
        """Version registry should persist across manager instances."""
        versions_dir = temp_dir / "versions"
        artifacts = temp_dir / "artifacts"
        artifacts.mkdir()

        # Create and save
        mgr1 = ModelVersionManager(versions_dir=versions_dir)
        mgr1.create_version(
            model_type="bilstm", artifacts_source=artifacts,
            metrics={"auroc": 0.85},
        )

        # Reload
        mgr2 = ModelVersionManager(versions_dir=versions_dir)
        assert len(mgr2.versions) == 1
        assert list(mgr2.versions.values())[0].auroc == 0.85

    def test_version_to_dict(self):
        """ModelVersion should serialize to dict."""
        v = _make_version()
        d = v.to_dict()
        assert d["version_id"] == "test-v1"
        assert d["auroc"] == 0.85

    def test_version_from_dict(self):
        """ModelVersion should deserialize from dict."""
        v = _make_version()
        d = v.to_dict()
        restored = ModelVersion.from_dict(d)
        assert restored.version_id == "test-v1"
        assert restored.auroc == 0.85

    def test_version_history(self, versioner, temp_dir):
        """Version history should be a timeline of all versions."""
        artifacts = temp_dir / "artifacts"
        artifacts.mkdir()

        versioner.create_version(
            model_type="xgboost", artifacts_source=artifacts, metrics={"auroc": 0.85},
        )
        versioner.create_version(
            model_type="xgboost", artifacts_source=artifacts, metrics={"auroc": 0.90},
        )

        history = versioner.get_version_history()
        assert len(history) == 2
        assert "version_id" in history[0]
        assert "auroc" in history[0]


# ═════════════════════════════════════════════════════════════════════════
# DATA ACCUMULATOR TESTS
# ═════════════════════════════════════════════════════════════════════════


class TestDataAccumulator:
    """Tests for live data accumulation."""

    def test_add_sample(self, accumulator):
        """Adding a sample should increment the count."""
        accumulator.add_sample(
            features={"hr": 80.0, "rr": 18.0},
            prediction=0.3,
            actual_outcome=0,
        )
        assert accumulator._total_accumulated == 1

    def test_flush_creates_parquet(self, accumulator):
        """Flushing should create a Parquet file on disk."""
        for i in range(5):
            accumulator.add_sample(
                features={"hr": 80.0 + i, "rr": 18.0},
                prediction=0.3,
            )

        path = accumulator.flush()
        assert path is not None
        assert path.suffix == ".parquet"
        assert path.exists()

    def test_get_buffer_data(self, accumulator):
        """get_buffer_data should return all accumulated samples."""
        for i in range(10):
            accumulator.add_sample(
                features={"hr": 80.0 + i},
                prediction=0.1 * i,
            )

        df = accumulator.get_buffer_data()
        assert len(df) == 10
        assert "hr" in df.columns
        assert "_prediction" in df.columns

    def test_has_enough_samples(self, accumulator):
        """has_enough_samples should reflect min_samples threshold."""
        assert not accumulator.has_enough_samples

        for i in range(10):
            accumulator.add_sample(features={"hr": 80.0}, prediction=0.5)

        accumulator.flush()
        assert accumulator.has_enough_samples

    def test_clear_buffer(self, accumulator):
        """Clearing should remove all data."""
        for i in range(5):
            accumulator.add_sample(features={"hr": 80.0}, prediction=0.5)
        accumulator.flush()

        count = accumulator.clear_buffer()
        assert count > 0
        assert accumulator.buffer_size == 0

    def test_add_batch(self, accumulator):
        """add_batch should add multiple samples at once."""
        df = pd.DataFrame({"hr": [80, 85, 90], "rr": [18, 20, 22]})
        predictions = np.array([0.2, 0.5, 0.8])
        actuals = np.array([0, 0, 1])

        accumulator.add_batch(df, predictions, actuals)
        assert accumulator._total_accumulated == 3

    def test_stats(self, accumulator):
        """Stats should include key metrics."""
        stats = accumulator.get_stats()
        assert "total_accumulated" in stats
        assert "has_enough_samples" in stats
        assert "min_samples_required" in stats


# ═════════════════════════════════════════════════════════════════════════
# CHAMPION/CHALLENGER TESTS
# ═════════════════════════════════════════════════════════════════════════


class TestChampionChallengerEvaluator:
    """Tests for champion/challenger comparison."""

    def test_promote_when_better(self, evaluator):
        """Challenger should be promoted when it exceeds thresholds."""
        champion = _make_version("champ", auroc=0.85, auprc=0.70)
        challenger = _make_version("chal", auroc=0.86, auprc=0.72)

        result = evaluator.compare(champion, challenger, evaluation_samples=100)
        assert result.recommendation == "PROMOTE"
        assert result.meets_auroc_criterion is True
        assert result.meets_auprc_criterion is True

    def test_reject_when_auroc_insufficient(self, evaluator):
        """Challenger should be rejected when AUROC improvement is too small."""
        champion = _make_version("champ", auroc=0.85, auprc=0.70)
        challenger = _make_version("chal", auroc=0.852, auprc=0.71)

        result = evaluator.compare(champion, challenger, evaluation_samples=100)
        assert result.recommendation == "REJECT"
        assert result.meets_auroc_criterion is False

    def test_reject_when_auprc_degrades(self, evaluator):
        """Challenger should be rejected when AUPRC degrades."""
        champion = _make_version("champ", auroc=0.85, auprc=0.70)
        challenger = _make_version("chal", auroc=0.87, auprc=0.68)

        result = evaluator.compare(champion, challenger, evaluation_samples=100)
        assert result.recommendation == "REJECT"
        assert result.meets_auprc_criterion is False

    def test_insufficient_data(self, evaluator):
        """Should flag insufficient data when not enough samples."""
        champion = _make_version("champ", auroc=0.85, auprc=0.70)
        challenger = _make_version("chal", auroc=0.90, auprc=0.80)

        result = evaluator.compare(champion, challenger, evaluation_samples=2)
        assert result.recommendation == "INSUFFICIENT_DATA"
        assert result.meets_sample_criterion is False

    def test_result_to_dict(self, evaluator):
        """ComparisonResult should serialize to dict."""
        champion = _make_version("champ", auroc=0.85, auprc=0.70)
        challenger = _make_version("chal", auroc=0.86, auprc=0.72)

        result = evaluator.compare(champion, challenger, evaluation_samples=100)
        d = result.to_dict()
        assert "recommendation" in d
        assert "auroc_delta" in d

    def test_auto_promote_disabled(self, evaluator):
        """should_auto_promote should return False when setting is off."""
        champion = _make_version("champ", auroc=0.85, auprc=0.70)
        challenger = _make_version("chal", auroc=0.90, auprc=0.80)

        result = evaluator.compare(champion, challenger, evaluation_samples=100)

        with patch("src.continuous.champion_challenger.continuous_settings") as mock:
            mock.auto_promote = False
            assert evaluator.should_auto_promote(result) is False


# ═════════════════════════════════════════════════════════════════════════
# SHADOW EVALUATOR TESTS
# ═════════════════════════════════════════════════════════════════════════


class TestShadowEvaluator:
    """Tests for shadow model evaluation."""

    def test_start_and_record(self, shadow):
        """Starting shadow mode and recording predictions should work."""
        shadow.start_shadow("v1", "v2")
        assert shadow.is_active is True

        shadow.record_prediction(patient_id=1, champion_prob=0.3, challenger_prob=0.4)
        shadow.record_prediction(patient_id=2, champion_prob=0.7, challenger_prob=0.8)

        assert len(shadow.records) == 2

    def test_record_outcome_backfill(self, shadow):
        """Recording outcomes should backfill predictions."""
        shadow.start_shadow("v1", "v2")
        shadow.record_prediction(patient_id=1, champion_prob=0.3, challenger_prob=0.4)

        updated = shadow.record_outcome(patient_id=1, actual_outcome=0)
        assert updated == 1
        assert shadow.records_with_outcomes[0].actual_outcome == 0

    def test_evaluation_complete(self, shadow):
        """Evaluation should complete when period elapses."""
        shadow.start_shadow("v1", "v2")
        # evaluation_hours=0 means immediately complete
        assert shadow.evaluation_complete is True

    def test_not_active_ignores_predictions(self, shadow):
        """Predictions should be ignored when not active."""
        shadow.record_prediction(patient_id=1, champion_prob=0.3, challenger_prob=0.4)
        assert len(shadow.records) == 0

    def test_compute_metrics_no_data(self, shadow):
        """Metrics computation with no labeled data should return status."""
        shadow.start_shadow("v1", "v2")
        metrics = shadow.compute_metrics()
        assert metrics["status"] == "no_labeled_data"

    def test_stop_shadow(self, shadow):
        """Stopping shadow should deactivate and return metrics."""
        shadow.start_shadow("v1", "v2")
        shadow.record_prediction(patient_id=1, champion_prob=0.3, challenger_prob=0.4)

        metrics = shadow.stop_shadow()
        assert shadow.is_active is False
        assert "total_records" in metrics

    def test_status(self, shadow):
        """Status should include key fields."""
        status = shadow.get_status()
        assert "is_active" in status
        assert "evaluation_complete" in status


# ═════════════════════════════════════════════════════════════════════════
# RETRAIN ORCHESTRATOR TESTS
# ═════════════════════════════════════════════════════════════════════════


class TestRetrainOrchestrator:
    """Tests for the retraining orchestrator."""

    def _make_orchestrator(self, temp_dir):
        """Helper to create an orchestrator with temp dirs."""
        from src.continuous.retrain_orchestrator import RetrainOrchestrator

        accumulator = DataAccumulator(
            buffer_dir=temp_dir / "buffer",
            min_samples=5,
        )
        versioner = ModelVersionManager(versions_dir=temp_dir / "versions")
        shadow = ShadowEvaluator(evaluation_hours=0)
        evaluator = ChampionChallengerEvaluator(
            auroc_threshold=0.005,
            min_evaluation_samples=1,
        )

        return RetrainOrchestrator(
            accumulator=accumulator,
            versioner=versioner,
            shadow_evaluator=shadow,
            evaluator=evaluator,
        )

    def test_no_drift_no_retrain(self, temp_dir):
        """No retraining without drift detection."""
        orch = self._make_orchestrator(temp_dir)
        assert orch.should_retrain() is False

    def test_drift_without_buffer_no_retrain(self, temp_dir):
        """Drift without enough buffer samples should not retrain."""
        orch = self._make_orchestrator(temp_dir)

        mock_detector = MagicMock()
        mock_detector.should_retrain.return_value = True
        orch.set_drift_detector(mock_detector)

        assert orch.should_retrain() is False

    def test_drift_with_buffer_triggers_retrain(self, temp_dir):
        """Drift with enough buffer samples should trigger retrain."""
        orch = self._make_orchestrator(temp_dir)

        mock_detector = MagicMock()
        mock_detector.should_retrain.return_value = True
        orch.set_drift_detector(mock_detector)

        # Fill buffer
        for i in range(10):
            orch.accumulator.add_sample(
                features={"hr": 80.0 + i},
                prediction=0.5,
            )
        orch.accumulator.flush()

        assert orch.should_retrain() is True

    def test_force_retrain(self, temp_dir):
        """Force retrain should bypass drift/buffer checks."""
        orch = self._make_orchestrator(temp_dir)

        version = orch.trigger_retrain(model_type="xgboost", force=True)
        assert version is not None
        assert version.version_id in orch.versioner.versions

    def test_orchestrator_status(self, temp_dir):
        """Status should include all key fields."""
        orch = self._make_orchestrator(temp_dir)
        status = orch.get_status()

        assert "continuous_enabled" in status
        assert "champion" in status
        assert "shadow_evaluation" in status
        assert "buffer" in status
        assert "recent_events" in status

    def test_event_log(self, temp_dir):
        """Events should be logged."""
        orch = self._make_orchestrator(temp_dir)
        orch.trigger_retrain(model_type="xgboost", force=True)

        events = orch.get_event_log()
        assert len(events) >= 2  # retrain_started + retrain_completed
        assert events[0]["event_type"] == "retrain_started"


# ═════════════════════════════════════════════════════════════════════════
# DRIFT DETECTOR EXTENSION TESTS
# ═════════════════════════════════════════════════════════════════════════


class TestDriftDetectorExtensions:
    """Tests for should_retrain and get_drift_severity additions."""

    def test_should_retrain_stable(self):
        """should_retrain should return False when stable."""
        from src.monitoring.drift_detector import DriftDetector

        detector = DriftDetector(feature_columns=["hr", "rr"])

        # Add matching reference and live data (no drift)
        ref = np.random.randn(100, 2)
        detector.set_reference(ref)
        for row in ref:
            detector.add_live_sample(row)

        assert detector.should_retrain() is False

    def test_get_drift_severity_format(self):
        """get_drift_severity should return well-formed dict."""
        from src.monitoring.drift_detector import DriftDetector

        detector = DriftDetector(feature_columns=["hr"])
        ref = np.random.randn(100, 1)
        detector.set_reference(ref)
        for row in ref:
            detector.add_live_sample(row)

        severity = detector.get_drift_severity()
        assert "status" in severity
        assert "retrain_recommended" in severity
        assert "drifted_features_count" in severity


# ═════════════════════════════════════════════════════════════════════════
# CONTINUOUS SETTINGS TESTS
# ═════════════════════════════════════════════════════════════════════════


class TestContinuousSettings:
    """Tests for ContinuousSettings configuration."""

    def test_defaults(self):
        """Default settings should have continuous disabled."""
        settings = ContinuousSettings()
        assert settings.continuous_enabled is False
        assert settings.auto_promote is False
        assert settings.drift_check_interval_s == 3600
        assert settings.min_retrain_samples == 5000
        assert settings.shadow_evaluation_hours == 24
        assert settings.champion_auroc_threshold == 0.005

    def test_env_prefix(self):
        """Settings should use JEEVAN_CL_ prefix."""
        assert ContinuousSettings.model_config["env_prefix"] == "JEEVAN_CL_"

    def test_auto_promote_default_off(self):
        """Auto-promote should default to False for clinical safety."""
        settings = ContinuousSettings()
        assert settings.auto_promote is False
