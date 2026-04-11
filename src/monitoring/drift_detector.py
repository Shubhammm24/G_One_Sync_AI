"""
G_One_Sync AI — Data & Prediction Drift Detector
====================================================
Monitors for distribution shift between training data and live
predictions. Uses PSI (feature drift) and KS-test (statistical drift).

Clinical significance:
    Drift in ICU data often means patient demographics, treatment
    protocols, or seasonal illness patterns have changed. If the model
    was trained on pre-COVID data and post-COVID patients arrive,
    feature distributions shift and model performance degrades silently.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger
from scipy import stats


class DriftDetector:
    """
    Real-time drift detection for ICU prediction models.

    Methods:
        1. PSI (Population Stability Index) — feature-level drift
        2. KS Test (Kolmogorov-Smirnov) — distribution comparison
        3. Prediction drift — shift in mean predicted probability
    """

    # PSI thresholds (industry standard)
    PSI_NO_DRIFT = 0.1
    PSI_MODERATE = 0.2
    PSI_SIGNIFICANT = 0.25

    def __init__(
        self,
        reference_data: Optional[np.ndarray] = None,
        feature_names: Optional[list[str]] = None,
        window_size: int = 5000,
        n_bins: int = 10,
    ):
        """
        Args:
            reference_data: Training/validation feature matrix (N, F) as baseline
            feature_names: Names for each feature column
            window_size: Max live samples to keep in sliding window
            n_bins: Number of bins for PSI calculation
        """
        self.feature_names = feature_names or []
        self.n_bins = n_bins
        self.window_size = window_size

        # Reference distributions (from training data)
        self.reference_histograms: dict[int, np.ndarray] = {}
        self.reference_edges: dict[int, np.ndarray] = {}
        self.reference_stats: dict[int, dict] = {}

        # Live data sliding window
        self.live_buffer: deque = deque(maxlen=window_size)

        # Prediction drift tracking
        self.reference_pred_mean: Optional[float] = None
        self.live_predictions: deque = deque(maxlen=window_size)

        # Drift history
        self.drift_history: list[dict] = []

        if reference_data is not None:
            self.set_reference(reference_data)

    def set_reference(self, data: np.ndarray) -> None:
        """
        Set the reference distribution from training data.

        Args:
            data: (N, F) feature matrix from training/validation set
        """
        logger.info("Setting reference distribution from {} samples, {} features",
                     data.shape[0], data.shape[1])

        for f_idx in range(data.shape[1]):
            col = data[:, f_idx]
            col = col[~np.isnan(col)]

            # Compute histogram for PSI
            hist, edges = np.histogram(col, bins=self.n_bins, density=False)
            hist = hist / hist.sum()  # Normalize to proportions
            hist = np.clip(hist, 1e-8, None)  # Avoid log(0)

            self.reference_histograms[f_idx] = hist
            self.reference_edges[f_idx] = edges

            # Store stats for KS test
            self.reference_stats[f_idx] = {
                "mean": float(np.mean(col)),
                "std": float(np.std(col)),
                "median": float(np.median(col)),
                "q25": float(np.percentile(col, 25)),
                "q75": float(np.percentile(col, 75)),
            }

        logger.info("Reference distributions set for {} features", data.shape[1])

    def set_reference_predictions(self, predictions: np.ndarray) -> None:
        """Set baseline prediction distribution."""
        self.reference_pred_mean = float(np.mean(predictions))
        logger.info("Reference prediction mean: {:.4f}", self.reference_pred_mean)

    # ── Live Data Ingestion ──────────────────────────────────────

    def add_live_sample(self, features: np.ndarray) -> None:
        """Add live feature vector(s) to the sliding window."""
        if features.ndim == 1:
            self.live_buffer.append(features)
        else:
            for row in features:
                self.live_buffer.append(row)

    def add_live_prediction(self, probability: float) -> None:
        """Add a live prediction to the tracking buffer."""
        self.live_predictions.append(probability)

    # ── PSI Calculation ──────────────────────────────────────────

    def compute_psi(
        self,
        feature_idx: int,
        live_data: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute Population Stability Index for a single feature.

        PSI = Σ (live_pct - ref_pct) × ln(live_pct / ref_pct)

        Values:
            < 0.10: No significant drift
            0.10 - 0.25: Moderate drift — monitor closely
            > 0.25: Significant drift — retrain recommended
        """
        if feature_idx not in self.reference_histograms:
            return 0.0

        if live_data is None:
            if len(self.live_buffer) < 100:
                return 0.0
            live_data = np.array(self.live_buffer)[:, feature_idx]

        live_data = live_data[~np.isnan(live_data)]
        if len(live_data) < 50:
            return 0.0

        edges = self.reference_edges[feature_idx]
        ref_hist = self.reference_histograms[feature_idx]

        # Compute live histogram using reference bin edges
        live_hist, _ = np.histogram(live_data, bins=edges, density=False)
        live_hist = live_hist / live_hist.sum()
        live_hist = np.clip(live_hist, 1e-8, None)

        # PSI formula
        psi = float(np.sum((live_hist - ref_hist) * np.log(live_hist / ref_hist)))

        return psi

    # ── KS Test ──────────────────────────────────────────────────

    def compute_ks_test(
        self,
        feature_idx: int,
        live_data: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Kolmogorov-Smirnov test comparing live vs reference distribution.

        Returns:
            Dict with 'statistic', 'p_value', and 'is_drifted'
        """
        if feature_idx not in self.reference_stats:
            return {"statistic": 0.0, "p_value": 1.0, "is_drifted": False}

        if live_data is None:
            if len(self.live_buffer) < 100:
                return {"statistic": 0.0, "p_value": 1.0, "is_drifted": False}
            live_data = np.array(self.live_buffer)[:, feature_idx]

        live_data = live_data[~np.isnan(live_data)]

        # Generate reference samples from stored stats
        ref_stats = self.reference_stats[feature_idx]
        ref_samples = np.random.normal(
            ref_stats["mean"], max(ref_stats["std"], 1e-8), size=len(live_data)
        )

        # Two-sample KS test
        ks_stat, p_value = stats.ks_2samp(ref_samples, live_data)

        return {
            "statistic": float(ks_stat),
            "p_value": float(p_value),
            "is_drifted": p_value < 0.01,  # 1% significance level
        }

    # ── Prediction Drift ─────────────────────────────────────────

    def compute_prediction_drift(self) -> dict:
        """Check if mean predicted probability has shifted."""
        if self.reference_pred_mean is None or len(self.live_predictions) < 50:
            return {"drift": 0.0, "is_drifted": False}

        live_mean = float(np.mean(self.live_predictions))
        drift = abs(live_mean - self.reference_pred_mean)
        relative_drift = drift / max(self.reference_pred_mean, 1e-8)

        return {
            "reference_mean": self.reference_pred_mean,
            "live_mean": live_mean,
            "absolute_drift": drift,
            "relative_drift": relative_drift,
            "is_drifted": relative_drift > 0.20,  # 20% drift threshold
        }

    # ── Full Drift Report ────────────────────────────────────────

    def generate_drift_report(self) -> dict:
        """
        Generate a comprehensive drift report across all features.

        Returns:
            Dict with per-feature PSI, KS results, and overall status
        """
        if len(self.live_buffer) < 100:
            return {
                "status": "insufficient_data",
                "samples_collected": len(self.live_buffer),
                "minimum_required": 100,
            }

        live_matrix = np.array(self.live_buffer)
        n_features = live_matrix.shape[1]

        feature_reports = []
        drifted_features = []

        for f_idx in range(min(n_features, len(self.reference_histograms))):
            psi = self.compute_psi(f_idx, live_matrix[:, f_idx])
            ks = self.compute_ks_test(f_idx, live_matrix[:, f_idx])

            name = self.feature_names[f_idx] if f_idx < len(self.feature_names) else f"feature_{f_idx}"

            # Determine drift severity
            if psi > self.PSI_SIGNIFICANT:
                severity = "SIGNIFICANT"
                drifted_features.append(name)
            elif psi > self.PSI_MODERATE:
                severity = "MODERATE"
                drifted_features.append(name)
            elif psi > self.PSI_NO_DRIFT:
                severity = "MINOR"
            else:
                severity = "NONE"

            feature_reports.append({
                "feature": name,
                "psi": psi,
                "ks_statistic": ks["statistic"],
                "ks_p_value": ks["p_value"],
                "drift_severity": severity,
            })

        # Sort by PSI descending
        feature_reports.sort(key=lambda x: x["psi"], reverse=True)

        # Prediction drift
        pred_drift = self.compute_prediction_drift()

        # Overall status
        overall_psi = np.mean([r["psi"] for r in feature_reports]) if feature_reports else 0
        if len(drifted_features) > n_features * 0.3:
            overall_status = "RETRAIN_RECOMMENDED"
        elif len(drifted_features) > 0:
            overall_status = "MONITOR"
        else:
            overall_status = "STABLE"

        report = {
            "status": overall_status,
            "samples_analyzed": len(self.live_buffer),
            "overall_mean_psi": float(overall_psi),
            "drifted_features_count": len(drifted_features),
            "drifted_features": drifted_features,
            "prediction_drift": pred_drift,
            "feature_details": feature_reports[:20],  # Top 20 by PSI
        }

        # Log summary
        logger.info("═══ Drift Report ═══")
        logger.info("  Status: {} | Mean PSI: {:.4f} | Drifted: {}/{}",
                     overall_status, overall_psi, len(drifted_features), n_features)
        for r in feature_reports[:5]:
            logger.info("  {} PSI={:.4f} [{}]", r["feature"], r["psi"], r["drift_severity"])

        self.drift_history.append(report)

        return report

    def save_report(self, path: Path) -> None:
        """Save drift report to JSON."""
        report = self.generate_drift_report()
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("Drift report saved → {}", path)
