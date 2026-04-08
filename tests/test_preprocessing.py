"""
JeevanSync AI — Preprocessing Layer Tests
===========================================
Tests for schema validation, imputation, sliding windows,
feature extraction, and label generation.
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.schema_validator import SchemaValidator, ValidationReport
from src.preprocessing.imputation import ImputationEngine
from src.preprocessing.sliding_window import SlidingWindowBuilder
from src.preprocessing.feature_extractor import FeatureExtractor
from src.preprocessing.label_generator import LabelGenerator
from src.preprocessing.time_alignment import TimeAligner


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def sample_panel_df() -> pd.DataFrame:
    """Create a small sample panel for testing."""
    np.random.seed(42)
    records = []
    for pid in [1, 2, 3]:
        for hour in range(12):
            records.append({
                "patient_id": pid,
                "hour_from_admission": hour,
                "heart_rate": 70 + np.random.randn() * 10 + (pid * 5),
                "respiratory_rate": 15 + np.random.randn() * 3,
                "spo2_pct": 97 - np.random.rand() * 3,
                "temperature_c": 37.0 + np.random.randn() * 0.3,
                "systolic_bp": 120 + np.random.randn() * 15,
                "diastolic_bp": 75 + np.random.randn() * 10,
                "oxygen_device": "none",
                "oxygen_flow": 0.0,
                "mobility_score": np.random.randint(1, 5),
                "nurse_alert": int(np.random.rand() > 0.8),
                "wbc_count": 7 + np.random.randn() * 2,
                "lactate": 1.2 + np.random.rand() * 0.5,
                "creatinine": 0.9 + np.random.rand() * 0.3,
                "crp_level": 10 + np.random.rand() * 20,
                "hemoglobin": 13 + np.random.randn(),
                "sepsis_risk_score": 0.1 + np.random.rand() * 0.2,
                "age": 55 + pid * 5,
                "gender": "M" if pid % 2 == 0 else "F",
                "comorbidity_index": pid,
                "admission_type": "ED",
                "deterioration_hour": 8 if pid == 1 else -1,
                "deterioration_event": 1 if pid == 1 else 0,
            })
    return pd.DataFrame(records)


@pytest.fixture
def sample_vitals_df() -> pd.DataFrame:
    """Small vitals-only DataFrame."""
    return pd.DataFrame({
        "patient_id": [1, 1, 1, 2, 2],
        "hour_from_admission": [0, 1, 2, 0, 1],
        "heart_rate": [72.0, 75.0, 78.0, 80.0, 82.0],
        "respiratory_rate": [16.0, 17.0, 18.0, 20.0, 22.0],
        "spo2_pct": [98.0, 97.0, 96.0, 95.0, 93.0],
        "temperature_c": [37.0, 37.1, 37.3, 37.5, 38.0],
        "systolic_bp": [120.0, 118.0, 115.0, 110.0, 105.0],
        "diastolic_bp": [80.0, 78.0, 76.0, 72.0, 68.0],
        "oxygen_device": ["none"] * 5,
        "oxygen_flow": [0.0] * 5,
        "mobility_score": [3, 3, 2, 2, 1],
        "nurse_alert": [0, 0, 1, 0, 1],
    })


# ── Schema Validator Tests ───────────────────────────────────────────────


class TestSchemaValidator:
    """Tests for schema validation."""

    def test_valid_data_passes(self, sample_panel_df):
        validator = SchemaValidator()
        cleaned, report = validator.validate(sample_panel_df, mode="panel")
        assert report.valid_rows == len(sample_panel_df)
        assert report.flagged_rows == 0

    def test_out_of_range_flagged(self):
        df = pd.DataFrame({
            "patient_id": [1],
            "hour_from_admission": [0],
            "heart_rate": [500.0],  # Out of range
            "respiratory_rate": [16.0],
            "spo2_pct": [98.0],
            "temperature_c": [37.0],
            "systolic_bp": [120.0],
            "diastolic_bp": [80.0],
            "oxygen_device": ["none"],
            "oxygen_flow": [0.0],
            "mobility_score": [3],
            "nurse_alert": [0],
            "wbc_count": [7.5],
            "lactate": [1.2],
            "creatinine": [0.9],
            "crp_level": [15.0],
            "hemoglobin": [13.5],
            "sepsis_risk_score": [0.15],
            "age": [50],
            "gender": ["M"],
            "comorbidity_index": [2],
            "admission_type": ["ED"],
        })
        validator = SchemaValidator()
        cleaned, report = validator.validate(df, mode="panel")
        assert report.flagged_rows > 0
        assert "heart_rate" in report.range_violations

    def test_enforce_dtypes(self, sample_panel_df):
        validator = SchemaValidator()
        typed = validator.enforce_dtypes(sample_panel_df)
        assert typed["heart_rate"].dtype == np.float64
        assert typed["gender"].dtype.name == "category"

    def test_patient_continuity_check(self, sample_panel_df):
        validator = SchemaValidator()
        gaps = validator.validate_patient_continuity(sample_panel_df)
        assert len(gaps) == 0  # Should have no gaps in sample data


# ── Imputation Tests ─────────────────────────────────────────────────────


class TestImputationEngine:
    """Tests for missing value imputation."""

    def test_no_missing_returns_unchanged(self, sample_panel_df):
        engine = ImputationEngine()
        engine.fit(sample_panel_df)
        result = engine.transform(sample_panel_df)
        pd.testing.assert_frame_equal(result, sample_panel_df)

    def test_forward_fill(self):
        df = pd.DataFrame({
            "patient_id": [1, 1, 1],
            "hour_from_admission": [0, 1, 2],
            "heart_rate": [72.0, np.nan, 78.0],
        })
        engine = ImputationEngine()
        engine.fit(df)
        result = engine.transform(df, strategy="forward_fill")
        assert result["heart_rate"].iloc[1] == 72.0  # forward-filled

    def test_median_impute(self):
        train_df = pd.DataFrame({
            "patient_id": [1, 1, 1],
            "hour_from_admission": [0, 1, 2],
            "heart_rate": [70.0, 80.0, 90.0],  # Median = 80.0
        })
        test_df = pd.DataFrame({
            "patient_id": [2],
            "hour_from_admission": [0],
            "heart_rate": [np.nan],
        })
        engine = ImputationEngine()
        engine.fit(train_df)
        result = engine.transform(test_df, strategy="median")
        assert result["heart_rate"].iloc[0] == 80.0

    def test_missing_summary(self):
        df = pd.DataFrame({
            "a": [1, np.nan, 3],
            "b": [np.nan, np.nan, 3],
            "c": [1, 2, 3],
        })
        engine = ImputationEngine()
        summary = engine.get_missing_summary(df)
        assert "b" in summary.index
        assert summary.loc["b", "missing_count"] == 2


# ── Sliding Window Tests ────────────────────────────────────────────────


class TestSlidingWindowBuilder:
    """Tests for the sliding window mechanism."""

    def test_window_shape(self, sample_panel_df):
        builder = SlidingWindowBuilder(
            window_size=4,
            feature_columns=["heart_rate", "respiratory_rate", "spo2_pct"],
        )
        result = builder.build_windows(sample_panel_df, show_progress=False)

        assert result["windows"].ndim == 3
        assert result["windows"].shape[1] == 4  # window_size
        assert result["windows"].shape[2] == 3  # n_features

    def test_window_padding(self):
        """Windows at the start of a stay should be padded."""
        df = pd.DataFrame({
            "patient_id": [1, 1, 1],
            "hour_from_admission": [0, 1, 2],
            "heart_rate": [70.0, 72.0, 74.0],
        })
        builder = SlidingWindowBuilder(
            window_size=4,
            feature_columns=["heart_rate"],
        )
        result = builder.build_windows(df, show_progress=False)

        # First window should be padded (4 slots, only 1 real value)
        first_window = result["windows"][0]
        assert first_window.shape == (4, 1)
        # Padding should repeat the first value
        assert first_window[0, 0] == 70.0
        assert first_window[-1, 0] == 70.0

    def test_flatten(self):
        windows = np.random.randn(10, 4, 3)  # 10 samples, 4 timesteps, 3 features
        builder = SlidingWindowBuilder(window_size=4)
        flat = builder.flatten_windows(windows)
        assert flat.shape == (10, 12)  # 4 * 3

    def test_flattened_feature_names(self):
        builder = SlidingWindowBuilder(window_size=3)
        names = builder.get_flattened_feature_names(["hr", "rr"])
        assert names == ["hr_t-2", "rr_t-2", "hr_t-1", "rr_t-1", "hr_t0", "rr_t0"]


# ── Feature Extractor Tests ──────────────────────────────────────────────


class TestFeatureExtractor:
    """Tests for feature extraction from windows."""

    def test_statistical_features(self, sample_vitals_df):
        extractor = FeatureExtractor(
            window_size=3,
            vital_columns=["heart_rate", "respiratory_rate"],
            lab_columns=[],
        )
        window = sample_vitals_df[sample_vitals_df["patient_id"] == 1]
        features = extractor.extract_statistical_features(window)

        assert "heart_rate_mean" in features
        assert "heart_rate_std" in features
        assert "heart_rate_min" in features
        assert "heart_rate_max" in features
        assert features["heart_rate_mean"] == pytest.approx(75.0, abs=0.1)

    def test_trend_features(self, sample_vitals_df):
        extractor = FeatureExtractor(
            window_size=3,
            vital_columns=["heart_rate"],
            lab_columns=[],
        )
        window = sample_vitals_df[sample_vitals_df["patient_id"] == 1]
        features = extractor.extract_trend_features(window)

        assert "heart_rate_slope" in features
        assert "heart_rate_delta" in features
        assert features["heart_rate_delta"] == pytest.approx(6.0, abs=0.1)  # 78 - 72

    def test_variability_features(self, sample_vitals_df):
        extractor = FeatureExtractor(
            window_size=3,
            vital_columns=["heart_rate", "systolic_bp", "spo2_pct"],
            lab_columns=[],
        )
        window = sample_vitals_df[sample_vitals_df["patient_id"] == 1]
        features = extractor.extract_variability_features(window)

        assert "hrv_sdnn" in features
        assert "bpv_sbp_std" in features
        assert "spo2_var_range" in features
        assert features["hrv_sdnn"] > 0

    def test_clinical_scores(self, sample_vitals_df):
        extractor = FeatureExtractor(window_size=3)
        window = sample_vitals_df[sample_vitals_df["patient_id"] == 1]
        features = extractor.extract_clinical_scores(window)

        assert "map_current" in features
        assert "shock_index_current" in features
        assert "qsofa_score" in features

        # MAP ≈ DBP + (SBP - DBP) / 3
        expected_map = 76.0 + (115.0 - 76.0) / 3
        assert features["map_current"] == pytest.approx(expected_map, abs=0.5)

    def test_all_features_extraction(self, sample_vitals_df):
        extractor = FeatureExtractor(
            window_size=3,
            vital_columns=["heart_rate", "respiratory_rate", "spo2_pct",
                           "temperature_c", "systolic_bp", "diastolic_bp"],
            lab_columns=[],
        )
        window = sample_vitals_df[sample_vitals_df["patient_id"] == 1]
        features = extractor.extract_all_features(window)

        # Should have many features
        assert len(features) > 50


# ── Label Generator Tests ────────────────────────────────────────────────


class TestLabelGenerator:
    """Tests for target label generation."""

    def test_generate_12h_label(self, sample_panel_df):
        gen = LabelGenerator()
        result = gen.generate_labels(sample_panel_df, horizons=[12])

        # Patient 1 has deterioration at hour 8
        # So hours 0-7 should have label=1 if hour < 8 && 8 <= hour + 12
        p1 = result[result["patient_id"] == 1]
        assert p1[p1["hour_from_admission"] == 0]["deterioration_next_12h"].iloc[0] == 1
        assert p1[p1["hour_from_admission"] == 7]["deterioration_next_12h"].iloc[0] == 1

        # Patient 2 has no deterioration
        p2 = result[result["patient_id"] == 2]
        assert p2["deterioration_next_12h"].sum() == 0

    def test_generate_6h_label(self, sample_panel_df):
        gen = LabelGenerator()
        result = gen.generate_labels(sample_panel_df, horizons=[6])

        # Patient 1: deterioration at hour 8
        # 6h label at hour 0: 0 < 8 <= 6 → False (8 > 6)
        # 6h label at hour 2: 2 < 8 <= 8 → True
        # 6h label at hour 3: 3 < 8 <= 9 → True
        p1 = result[result["patient_id"] == 1]
        assert p1[p1["hour_from_admission"] == 2]["deterioration_next_6h"].iloc[0] == 1
        assert p1[p1["hour_from_admission"] == 3]["deterioration_next_6h"].iloc[0] == 1

    def test_class_distribution(self, sample_panel_df):
        gen = LabelGenerator()
        result = gen.generate_labels(sample_panel_df, horizons=[12])
        analysis = gen.analyze_class_distribution(result, "deterioration_next_12h")

        assert "total_samples" in analysis
        assert "positive_rate" in analysis
        assert "recommendation" in analysis

    def test_sample_weights(self):
        gen = LabelGenerator()
        labels = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])  # 80/20 imbalance
        weights = gen.generate_sample_weights(labels)

        assert len(weights) == 10
        assert weights[8] > weights[0]  # Positive samples should have higher weight


# ── Time Aligner Tests ──────────────────────────────────────────────────


class TestTimeAligner:
    """Tests for time alignment."""

    def test_merge_vitals_and_labs(self):
        aligner = TimeAligner()

        vitals = pd.DataFrame({
            "patient_id": [1, 1],
            "hour_from_admission": [0, 1],
            "heart_rate": [72.0, 75.0],
        })
        labs = pd.DataFrame({
            "patient_id": [1, 1],
            "hour_from_admission": [0, 1],
            "lactate": [1.2, 1.5],
        })

        merged = aligner.merge_vitals_and_labs(vitals, labs)
        assert "heart_rate" in merged.columns
        assert "lactate" in merged.columns
        assert len(merged) == 2

    def test_continuous_grid(self):
        aligner = TimeAligner()
        df = pd.DataFrame({
            "patient_id": [1, 1, 1],
            "hour_from_admission": [0, 2, 4],  # Gaps at hours 1, 3
            "heart_rate": [70.0, 74.0, 78.0],
        })
        result = aligner.ensure_continuous_grid(df)
        assert len(result) == 5  # Hours 0, 1, 2, 3, 4
        assert 1 in result["hour_from_admission"].values
        assert 3 in result["hour_from_admission"].values

    def test_clip_stay_length(self):
        aligner = TimeAligner()
        df = pd.DataFrame({
            "patient_id": [1] * 100,
            "hour_from_admission": list(range(100)),
        })
        result = aligner.clip_stay_length(df, max_hours=72)
        assert result["hour_from_admission"].max() == 71
