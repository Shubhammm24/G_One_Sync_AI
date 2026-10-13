"""
G_One_Sync AI — Champion/Challenger Evaluator
=================================================
Compares a newly trained challenger model against the current
champion using configurable promotion criteria. In a clinical
setting, this ensures that no model goes live without rigorous
comparison against the production baseline.

Promotion rules:
    1. Challenger AUROC > Champion AUROC by at least threshold (default 0.5%)
    2. Challenger AUPRC >= Champion AUPRC (must not degrade)
    3. Challenger must be evaluated on at least min_evaluation_samples
    4. If auto_promote is False, human approval is required

Clinical rationale:
    In a life-safety system, a model that trades AUPRC (precision-recall)
    for AUROC (discrimination) could increase false positives in an
    already alert-fatigued ICU, doing more harm than good.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from loguru import logger

from config.settings import continuous_settings
from src.continuous.model_versioner import ModelVersion


@dataclass
class ComparisonResult:
    """Result of comparing challenger vs champion model."""

    champion_version: str
    challenger_version: str
    champion_auroc: float
    challenger_auroc: float
    champion_auprc: float
    challenger_auprc: float
    auroc_delta: float
    auprc_delta: float
    auroc_threshold: float
    auprc_threshold: float
    evaluation_samples: int
    min_samples_required: int
    meets_auroc_criterion: bool
    meets_auprc_criterion: bool
    meets_sample_criterion: bool
    recommendation: str  # "PROMOTE", "REJECT", "INSUFFICIENT_DATA"
    reason: str
    timestamp: float

    def to_dict(self) -> dict:
        return {
            "champion_version": self.champion_version,
            "challenger_version": self.challenger_version,
            "champion_auroc": self.champion_auroc,
            "challenger_auroc": self.challenger_auroc,
            "champion_auprc": self.champion_auprc,
            "challenger_auprc": self.challenger_auprc,
            "auroc_delta": self.auroc_delta,
            "auprc_delta": self.auprc_delta,
            "meets_auroc_criterion": self.meets_auroc_criterion,
            "meets_auprc_criterion": self.meets_auprc_criterion,
            "meets_sample_criterion": self.meets_sample_criterion,
            "recommendation": self.recommendation,
            "reason": self.reason,
            "timestamp": self.timestamp,
        }


class ChampionChallengerEvaluator:
    """
    Evaluates whether a challenger model should replace the champion.

    Uses configurable thresholds from ContinuousSettings to make
    promotion recommendations. The final decision can be automatic
    or require human approval.
    """

    def __init__(
        self,
        auroc_threshold: float | None = None,
        auprc_threshold: float | None = None,
        min_evaluation_samples: int | None = None,
    ):
        self.auroc_threshold = (
            auroc_threshold if auroc_threshold is not None
            else continuous_settings.champion_auroc_threshold
        )
        self.auprc_threshold = (
            auprc_threshold if auprc_threshold is not None
            else continuous_settings.champion_auprc_threshold
        )
        self.min_evaluation_samples = (
            min_evaluation_samples if min_evaluation_samples is not None
            else continuous_settings.min_evaluation_samples
        )

        logger.info(
            "ChampionChallengerEvaluator initialized: "
            "AUROC threshold={:.4f} | AUPRC threshold={:.4f} | min_samples={}",
            self.auroc_threshold,
            self.auprc_threshold,
            self.min_evaluation_samples,
        )

    def compare(
        self,
        champion: ModelVersion,
        challenger: ModelVersion,
        evaluation_samples: int = 0,
    ) -> ComparisonResult:
        """Compare challenger against champion and produce a recommendation.

        Args:
            champion: Current production model version.
            challenger: Newly trained candidate model version.
            evaluation_samples: Number of samples used for evaluation.

        Returns:
            ComparisonResult with a recommendation.
        """
        auroc_delta = challenger.auroc - champion.auroc
        auprc_delta = challenger.auprc - champion.auprc

        meets_auroc = auroc_delta >= self.auroc_threshold
        meets_auprc = auprc_delta >= self.auprc_threshold
        meets_samples = evaluation_samples >= self.min_evaluation_samples

        # Determine recommendation
        if not meets_samples:
            recommendation = "INSUFFICIENT_DATA"
            reason = (
                f"Only {evaluation_samples} evaluation samples "
                f"(need {self.min_evaluation_samples})"
            )
        elif meets_auroc and meets_auprc:
            recommendation = "PROMOTE"
            reason = (
                f"Challenger improves AUROC by {auroc_delta:+.4f} "
                f"(threshold: {self.auroc_threshold:.4f}) "
                f"and AUPRC by {auprc_delta:+.4f}"
            )
        elif not meets_auroc:
            recommendation = "REJECT"
            reason = (
                f"AUROC improvement {auroc_delta:+.4f} below threshold "
                f"{self.auroc_threshold:.4f}"
            )
        else:
            recommendation = "REJECT"
            reason = f"AUPRC degraded by {auprc_delta:+.4f} (must be >= 0)"

        result = ComparisonResult(
            champion_version=champion.version_id,
            challenger_version=challenger.version_id,
            champion_auroc=champion.auroc,
            challenger_auroc=challenger.auroc,
            champion_auprc=champion.auprc,
            challenger_auprc=challenger.auprc,
            auroc_delta=auroc_delta,
            auprc_delta=auprc_delta,
            auroc_threshold=self.auroc_threshold,
            auprc_threshold=self.auprc_threshold,
            evaluation_samples=evaluation_samples,
            min_samples_required=self.min_evaluation_samples,
            meets_auroc_criterion=meets_auroc,
            meets_auprc_criterion=meets_auprc,
            meets_sample_criterion=meets_samples,
            recommendation=recommendation,
            reason=reason,
            timestamp=time.time(),
        )

        logger.info(
            "Champion vs Challenger: {} | "
            "AUROC {:.4f} → {:.4f} (Δ={:+.4f}) | "
            "AUPRC {:.4f} → {:.4f} (Δ={:+.4f}) | "
            "samples={} | → {}",
            f"{champion.version_id} vs {challenger.version_id}",
            champion.auroc,
            challenger.auroc,
            auroc_delta,
            champion.auprc,
            challenger.auprc,
            auprc_delta,
            evaluation_samples,
            recommendation,
        )

        return result

    def should_auto_promote(self, result: ComparisonResult) -> bool:
        """Check if a challenger should be automatically promoted.

        Requires:
            1. Result recommendation is PROMOTE
            2. auto_promote is enabled in settings

        Args:
            result: Comparison result from compare().

        Returns:
            True if automatic promotion is allowed and recommended.
        """
        if not continuous_settings.auto_promote:
            logger.info(
                "Auto-promote disabled — promotion of {} requires human approval",
                result.challenger_version,
            )
            return False

        return result.recommendation == "PROMOTE"
