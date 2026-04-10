"""
G_One_Sync AI — Probability Calibration
==========================================
Post-training calibration ensures predicted probabilities match true
outcome rates. Critical for clinical decision-making — a predicted
70% risk should mean ~70% of such patients actually deteriorate.

Methods:
    1. Platt Scaling — logistic regression on model outputs
    2. Isotonic Regression — non-parametric monotonic calibration
    3. Temperature Scaling — single-parameter learned temperature

Also includes reliability diagrams and threshold optimization.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
)

from config.settings import model_settings


class ProbabilityCalibrator:
    """
    Post-training probability calibration for clinical deterioration models.

    Calibration ensures that when the model says "70% risk", approximately
    70% of those patients actually deteriorate. This is essential for
    clinical trust and actionability.
    """

    def __init__(self, method: str = "isotonic"):
        """
        Args:
            method: 'platt' (logistic), 'isotonic', or 'temperature'
        """
        self.method = method
        self.calibrator = None
        self.is_fitted = False
        self.calibration_stats: dict = {}

    def fit(
        self,
        y_prob: np.ndarray,
        y_true: np.ndarray,
    ) -> "ProbabilityCalibrator":
        """
        Fit the calibrator on validation set predictions.

        Args:
            y_prob: Uncalibrated predicted probabilities (N,)
            y_true: True binary labels (N,)
        """
        logger.info("Fitting {} calibrator on {} samples...", self.method, len(y_prob))

        if self.method == "platt":
            # Platt scaling: logistic regression on logit(prob)
            self.calibrator = LogisticRegression(C=1.0, max_iter=1000)
            logits = np.log(np.clip(y_prob, 1e-7, 1 - 1e-7) / (1 - np.clip(y_prob, 1e-7, 1 - 1e-7)))
            self.calibrator.fit(logits.reshape(-1, 1), y_true)

        elif self.method == "isotonic":
            # Isotonic regression: non-parametric, monotonic fit
            self.calibrator = IsotonicRegression(
                out_of_bounds="clip", y_min=0.0, y_max=1.0
            )
            self.calibrator.fit(y_prob, y_true)

        elif self.method == "temperature":
            # Temperature scaling: find T that minimizes NLL
            best_t, best_nll = 1.0, float("inf")
            for t in np.linspace(0.1, 5.0, 490):
                logits = np.log(np.clip(y_prob, 1e-7, 1 - 1e-7) / (1 - np.clip(y_prob, 1e-7, 1 - 1e-7)))
                scaled = 1 / (1 + np.exp(-logits / t))
                nll = -np.mean(y_true * np.log(scaled + 1e-7) + (1 - y_true) * np.log(1 - scaled + 1e-7))
                if nll < best_nll:
                    best_nll, best_t = nll, t
            self.calibrator = best_t
            logger.info("Temperature scaling: T={:.3f}", best_t)

        self.is_fitted = True

        # Compute calibration stats
        calibrated = self.transform(y_prob)
        self.calibration_stats = self._compute_stats(y_prob, calibrated, y_true)

        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        """Transform uncalibrated probabilities to calibrated ones."""
        if not self.is_fitted:
            raise RuntimeError("Calibrator must be fitted first")

        if self.method == "platt":
            logits = np.log(np.clip(y_prob, 1e-7, 1 - 1e-7) / (1 - np.clip(y_prob, 1e-7, 1 - 1e-7)))
            return self.calibrator.predict_proba(logits.reshape(-1, 1))[:, 1]

        elif self.method == "isotonic":
            return self.calibrator.transform(y_prob)

        elif self.method == "temperature":
            logits = np.log(np.clip(y_prob, 1e-7, 1 - 1e-7) / (1 - np.clip(y_prob, 1e-7, 1 - 1e-7)))
            return 1 / (1 + np.exp(-logits / self.calibrator))

        return y_prob

    def _compute_stats(
        self,
        y_prob_raw: np.ndarray,
        y_prob_cal: np.ndarray,
        y_true: np.ndarray,
    ) -> dict:
        """Compute before/after calibration metrics."""
        brier_before = brier_score_loss(y_true, y_prob_raw)
        brier_after = brier_score_loss(y_true, y_prob_cal)

        # ECE (Expected Calibration Error) — 10 bins
        ece_before = self._ece(y_prob_raw, y_true)
        ece_after = self._ece(y_prob_cal, y_true)

        stats = {
            "brier_before": float(brier_before),
            "brier_after": float(brier_after),
            "brier_improvement": float((brier_before - brier_after) / brier_before * 100),
            "ece_before": float(ece_before),
            "ece_after": float(ece_after),
            "ece_improvement": float((ece_before - ece_after) / max(ece_before, 1e-8) * 100),
            "auroc_preserved": float(roc_auc_score(y_true, y_prob_cal)),
            "method": self.method,
        }

        logger.info("═══ Calibration Results ({}) ═══", self.method)
        logger.info("  Brier Score: {:.4f} → {:.4f} ({:.1f}% improvement)",
                     brier_before, brier_after, stats["brier_improvement"])
        logger.info("  ECE: {:.4f} → {:.4f} ({:.1f}% improvement)",
                     ece_before, ece_after, stats["ece_improvement"])
        logger.info("  AUROC preserved: {:.4f}", stats["auroc_preserved"])

        return stats

    @staticmethod
    def _ece(y_prob: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
        """Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
            if mask.sum() == 0:
                continue
            bin_acc = y_true[mask].mean()
            bin_conf = y_prob[mask].mean()
            ece += mask.sum() / len(y_prob) * abs(bin_acc - bin_conf)
        return ece


class ThresholdOptimizer:
    """
    Find the optimal classification threshold for each model
    based on clinical objectives.
    """

    def __init__(self):
        self.optimal_thresholds: dict[str, dict] = {}

    def optimize(
        self,
        y_prob: np.ndarray,
        y_true: np.ndarray,
        model_name: str = "model",
        strategies: list[str] | None = None,
    ) -> dict[str, float]:
        """
        Find optimal thresholds using multiple strategies.

        Strategies:
            - f1: Maximize F1 score
            - youden: Maximize Youden's J (sensitivity + specificity - 1)
            - precision_90: Minimum threshold for 90% precision
            - recall_90: Maximum threshold maintaining 90% recall
            - clinical: Balance for clinical use (high recall, acceptable precision)
        """
        strategies = strategies or ["f1", "youden", "precision_90", "recall_90", "clinical"]
        thresholds = np.linspace(0.05, 0.95, 181)

        results = {}
        best_metrics = {}

        for strategy in strategies:
            if strategy == "f1":
                best_t, best_v = 0.5, 0.0
                for t in thresholds:
                    preds = (y_prob >= t).astype(int)
                    v = f1_score(y_true, preds, zero_division=0)
                    if v > best_v:
                        best_v, best_t = v, t
                results["f1"] = float(best_t)
                best_metrics["f1"] = float(best_v)

            elif strategy == "youden":
                from sklearn.metrics import roc_curve
                fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
                j_scores = tpr - fpr
                best_idx = np.argmax(j_scores)
                results["youden"] = float(roc_thresholds[best_idx])
                best_metrics["youden_j"] = float(j_scores[best_idx])

            elif strategy == "precision_90":
                best_t = 0.5
                for t in thresholds:
                    preds = (y_prob >= t).astype(int)
                    p = precision_score(y_true, preds, zero_division=0)
                    if p >= 0.90:
                        best_t = t
                        break
                results["precision_90"] = float(best_t)

            elif strategy == "recall_90":
                best_t = 0.5
                for t in reversed(thresholds):
                    preds = (y_prob >= t).astype(int)
                    r = recall_score(y_true, preds, zero_division=0)
                    if r >= 0.90:
                        best_t = t
                        break
                results["recall_90"] = float(best_t)

            elif strategy == "clinical":
                # Clinical: maximize F1 with a recall floor of 75%
                best_t, best_v = 0.5, 0.0
                for t in thresholds:
                    preds = (y_prob >= t).astype(int)
                    r = recall_score(y_true, preds, zero_division=0)
                    if r >= 0.75:
                        v = f1_score(y_true, preds, zero_division=0)
                        if v > best_v:
                            best_v, best_t = v, t
                results["clinical"] = float(best_t)
                best_metrics["clinical_f1"] = float(best_v)

        self.optimal_thresholds[model_name] = results

        # Log results
        logger.info("═══ Threshold Optimization: {} ═══", model_name)
        for strategy, threshold in results.items():
            preds = (y_prob >= threshold).astype(int)
            prec = precision_score(y_true, preds, zero_division=0)
            rec = recall_score(y_true, preds, zero_division=0)
            f1 = f1_score(y_true, preds, zero_division=0)
            logger.info(
                "  {}: threshold={:.3f} → P={:.3f} R={:.3f} F1={:.3f}",
                strategy.ljust(14), threshold, prec, rec, f1,
            )

        return results

    def save(self, path: Path) -> None:
        """Save optimal thresholds to JSON."""
        with open(path, "w") as f:
            json.dump(self.optimal_thresholds, f, indent=2)
        logger.info("Thresholds saved → {}", path)

    def load(self, path: Path) -> None:
        """Load optimal thresholds from JSON."""
        with open(path) as f:
            self.optimal_thresholds = json.load(f)
