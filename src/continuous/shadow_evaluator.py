"""
G_One_Sync AI — Shadow Model Evaluator
==========================================
Evaluates a challenger model in shadow mode: the challenger
scores every live prediction request alongside the champion,
but its output is logged only — never served to clinicians.

After the shadow evaluation period (default 24h), the evaluator
compares accumulated shadow predictions against actual outcomes
to decide if the challenger should be promoted.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from loguru import logger

from config.settings import continuous_settings


@dataclass
class ShadowPrediction:
    """A single shadow prediction record."""

    patient_id: int
    champion_prob: float
    challenger_prob: float
    actual_outcome: Optional[int] = None
    timestamp: float = field(default_factory=time.time)


class ShadowEvaluator:
    """
    Runs a challenger model in shadow mode alongside the champion.

    Collects paired (champion, challenger) predictions on the same
    inputs, then evaluates both against actual outcomes once they
    are available.

    The shadow period is configurable (default 24 hours). At the end
    of the period, metrics are computed for both models and a
    comparison is produced.
    """

    def __init__(
        self,
        evaluation_hours: int | None = None,
        max_records: int = 100_000,
    ):
        self.evaluation_hours = (
            evaluation_hours or continuous_settings.shadow_evaluation_hours
        )
        self.max_records = max_records

        # Shadow prediction buffer
        self.records: deque[ShadowPrediction] = deque(maxlen=max_records)

        # Tracking
        self.started_at: Optional[float] = None
        self.champion_version: str = ""
        self.challenger_version: str = ""
        self._is_active: bool = False

        logger.info(
            "ShadowEvaluator initialized: evaluation_period={}h | max_records={}",
            self.evaluation_hours,
            self.max_records,
        )

    def start_shadow(
        self,
        champion_version: str,
        challenger_version: str,
    ) -> None:
        """Start a shadow evaluation period.

        Args:
            champion_version: Version ID of the current champion.
            challenger_version: Version ID of the challenger.
        """
        self.champion_version = champion_version
        self.challenger_version = challenger_version
        self.started_at = time.time()
        self._is_active = True
        self.records.clear()

        logger.info(
            "Shadow evaluation started: {} vs {} | period={}h",
            champion_version,
            challenger_version,
            self.evaluation_hours,
        )

    def record_prediction(
        self,
        patient_id: int,
        champion_prob: float,
        challenger_prob: float,
    ) -> None:
        """Record a paired shadow prediction.

        Args:
            patient_id: Patient identifier.
            champion_prob: Champion model's predicted probability.
            challenger_prob: Challenger model's predicted probability.
        """
        if not self._is_active:
            return

        self.records.append(ShadowPrediction(
            patient_id=patient_id,
            champion_prob=champion_prob,
            challenger_prob=challenger_prob,
        ))

    def record_outcome(self, patient_id: int, actual_outcome: int) -> int:
        """Backfill actual outcome for a patient's predictions.

        Args:
            patient_id: Patient identifier.
            actual_outcome: True label (0 or 1).

        Returns:
            Number of records updated.
        """
        updated = 0
        for record in self.records:
            if record.patient_id == patient_id and record.actual_outcome is None:
                record.actual_outcome = actual_outcome
                updated += 1
        return updated

    @property
    def is_active(self) -> bool:
        """Check if shadow evaluation is currently running."""
        return self._is_active

    @property
    def evaluation_complete(self) -> bool:
        """Check if the shadow evaluation period has elapsed."""
        if not self._is_active or self.started_at is None:
            return False
        elapsed_hours = (time.time() - self.started_at) / 3600
        return elapsed_hours >= self.evaluation_hours

    @property
    def records_with_outcomes(self) -> list[ShadowPrediction]:
        """Get records that have actual outcomes filled in."""
        return [r for r in self.records if r.actual_outcome is not None]

    def compute_metrics(self) -> dict:
        """Compute evaluation metrics for both champion and challenger.

        Returns:
            Dict with champion_metrics, challenger_metrics, and comparison.
        """
        labeled = self.records_with_outcomes
        if not labeled:
            return {
                "status": "no_labeled_data",
                "total_records": len(self.records),
                "labeled_records": 0,
            }

        actuals = np.array([r.actual_outcome for r in labeled])
        champion_probs = np.array([r.champion_prob for r in labeled])
        challenger_probs = np.array([r.challenger_prob for r in labeled])

        # Avoid import-time dependency on sklearn
        from sklearn.metrics import (
            roc_auc_score,
            average_precision_score,
        )

        champion_metrics = {}
        challenger_metrics = {}

        try:
            champion_metrics["auroc"] = float(roc_auc_score(actuals, champion_probs))
            champion_metrics["auprc"] = float(average_precision_score(actuals, champion_probs))
        except ValueError:
            champion_metrics["auroc"] = 0.0
            champion_metrics["auprc"] = 0.0

        try:
            challenger_metrics["auroc"] = float(roc_auc_score(actuals, challenger_probs))
            challenger_metrics["auprc"] = float(average_precision_score(actuals, challenger_probs))
        except ValueError:
            challenger_metrics["auroc"] = 0.0
            challenger_metrics["auprc"] = 0.0

        return {
            "status": "evaluated",
            "total_records": len(self.records),
            "labeled_records": len(labeled),
            "champion_version": self.champion_version,
            "challenger_version": self.challenger_version,
            "champion_metrics": champion_metrics,
            "challenger_metrics": challenger_metrics,
            "auroc_delta": challenger_metrics["auroc"] - champion_metrics["auroc"],
            "auprc_delta": challenger_metrics["auprc"] - champion_metrics["auprc"],
            "evaluation_hours_elapsed": (
                (time.time() - self.started_at) / 3600 if self.started_at else 0
            ),
        }

    def stop_shadow(self) -> dict:
        """Stop the shadow evaluation and return final metrics.

        Returns:
            Final evaluation metrics.
        """
        metrics = self.compute_metrics()
        self._is_active = False

        logger.info(
            "Shadow evaluation stopped: {} records ({} labeled) | "
            "champion AUROC={:.4f} | challenger AUROC={:.4f}",
            len(self.records),
            len(self.records_with_outcomes),
            metrics.get("champion_metrics", {}).get("auroc", 0),
            metrics.get("challenger_metrics", {}).get("auroc", 0),
        )

        return metrics

    def get_status(self) -> dict:
        """Get current shadow evaluation status."""
        elapsed_hours = 0.0
        if self.started_at:
            elapsed_hours = (time.time() - self.started_at) / 3600

        return {
            "is_active": self._is_active,
            "champion_version": self.champion_version,
            "challenger_version": self.challenger_version,
            "total_records": len(self.records),
            "labeled_records": len(self.records_with_outcomes),
            "elapsed_hours": elapsed_hours,
            "evaluation_hours": self.evaluation_hours,
            "evaluation_complete": self.evaluation_complete,
        }
