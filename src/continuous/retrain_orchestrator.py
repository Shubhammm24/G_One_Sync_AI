"""
G_One_Sync AI — Retraining Orchestrator
==========================================
Central coordinator for continuous learning. Monitors drift,
triggers retraining, manages shadow evaluation, and handles
champion/challenger promotion — all gated behind feature flags.

Flow:
    1. Periodically check DriftDetector.should_retrain()
    2. If drift detected AND buffer has enough samples:
        a. Merge buffer data with historical training data
        b. Retrain models (calls existing train_pipeline)
        c. Create new model version
        d. Start shadow evaluation
    3. After shadow period:
        a. Compare challenger vs champion
        b. Promote if criteria met (or queue for human approval)
        c. Discard if criteria not met

Usage:
    python -m src.continuous.retrain_orchestrator
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
from loguru import logger

from config.settings import continuous_settings, model_settings
from src.continuous.champion_challenger import (
    ChampionChallengerEvaluator,
    ComparisonResult,
)
from src.continuous.data_accumulator import DataAccumulator
from src.continuous.model_versioner import ModelVersion, ModelVersionManager
from src.continuous.shadow_evaluator import ShadowEvaluator


class RetrainOrchestrator:
    """
    Orchestrates the continuous learning lifecycle.

    Manages the interaction between drift detection, data
    accumulation, model retraining, shadow evaluation, and
    champion/challenger promotion.
    """

    def __init__(
        self,
        drift_detector: Any = None,
        accumulator: DataAccumulator | None = None,
        versioner: ModelVersionManager | None = None,
        shadow_evaluator: ShadowEvaluator | None = None,
        evaluator: ChampionChallengerEvaluator | None = None,
    ):
        self.accumulator = accumulator or DataAccumulator()
        self.versioner = versioner or ModelVersionManager()
        self.shadow = shadow_evaluator or ShadowEvaluator()
        self.evaluator = evaluator or ChampionChallengerEvaluator()

        # Drift detector injected (from existing monitoring module)
        self._drift_detector = drift_detector

        # State tracking
        self._retraining_in_progress: bool = False
        self._last_drift_check: float = 0.0
        self._last_retrain_time: float = 0.0
        self._retrain_count: int = 0
        self._pending_approval: Optional[ComparisonResult] = None

        # Event log
        self._event_log: list[dict] = []

        logger.info(
            "RetrainOrchestrator initialized | "
            "drift_check_interval={}s | min_samples={} | "
            "auto_promote={} | shadow_hours={}",
            continuous_settings.drift_check_interval_s,
            continuous_settings.min_retrain_samples,
            continuous_settings.auto_promote,
            continuous_settings.shadow_evaluation_hours,
        )

    def set_drift_detector(self, drift_detector: Any) -> None:
        """Set the drift detector (injected after construction).

        Args:
            drift_detector: An instance of DriftDetector from
                src.monitoring.drift_detector.
        """
        self._drift_detector = drift_detector

    # ── Event Logging ────────────────────────────────────────────

    def _log_event(self, event_type: str, details: dict) -> None:
        """Log an orchestration event."""
        event = {
            "event_type": event_type,
            "timestamp": time.time(),
            **details,
        }
        self._event_log.append(event)
        logger.info("CL Event: {} | {}", event_type, details)

    # ── Drift Check ──────────────────────────────────────────────

    def check_drift(self) -> bool:
        """Check if drift warrants retraining.

        Returns:
            True if retraining is recommended.
        """
        if self._drift_detector is None:
            logger.debug("No drift detector configured — skipping check")
            return False

        self._last_drift_check = time.time()

        if hasattr(self._drift_detector, "should_retrain"):
            should = self._drift_detector.should_retrain()
        else:
            report = self._drift_detector.generate_drift_report()
            should = report.get("status") == "RETRAIN_RECOMMENDED"

        if should:
            self._log_event("drift_detected", {
                "action": "retraining_recommended",
            })

        return should

    # ── Retraining Trigger ───────────────────────────────────────

    def should_retrain(self) -> bool:
        """Determine if retraining should be triggered.

        Conditions:
            1. Drift detected
            2. Buffer has enough samples
            3. No retraining already in progress
            4. Shadow evaluation is not running

        Returns:
            True if all conditions are met.
        """
        if self._retraining_in_progress:
            logger.debug("Retraining already in progress")
            return False

        if self.shadow.is_active:
            logger.debug("Shadow evaluation in progress — skipping retrain")
            return False

        drift_detected = self.check_drift()
        buffer_ready = self.accumulator.has_enough_samples

        if drift_detected and not buffer_ready:
            logger.info(
                "Drift detected but buffer insufficient: {}/{} samples",
                self.accumulator.buffer_size,
                continuous_settings.min_retrain_samples,
            )

        return drift_detected and buffer_ready

    def trigger_retrain(
        self,
        model_type: str = "ensemble",
        force: bool = False,
    ) -> Optional[ModelVersion]:
        """Trigger a model retraining run.

        This creates a new model version from the accumulated data.
        In production, this would call train_pipeline.main() — here
        we create the version structure and prepare for shadow eval.

        Args:
            model_type: Which model architecture to retrain.
            force: If True, skip drift/buffer checks.

        Returns:
            New model version if retraining succeeded, else None.
        """
        if not force and not self.should_retrain():
            return None

        self._retraining_in_progress = True
        self._log_event("retrain_started", {
            "model_type": model_type,
            "buffer_samples": self.accumulator.buffer_size,
            "trigger": "drift" if not force else "manual",
        })

        try:
            # Get buffer data
            buffer_data = self.accumulator.get_buffer_data()
            buffer_samples = len(buffer_data)

            logger.info(
                "Starting retraining: {} samples from buffer | model={}",
                buffer_samples,
                model_type,
            )

            # In a full implementation, this would:
            # 1. Merge buffer data with historical training data
            # 2. Call train_pipeline.main() with merged data
            # 3. Evaluate on held-out set
            # 4. Return the trained model artifacts
            #
            # For now, we create the version structure so the
            # orchestration flow works end-to-end.

            # Create placeholder metrics (replaced by actual training)
            metrics = {
                "auroc": 0.0,
                "auprc": 0.0,
                "f1": 0.0,
                "training_samples": buffer_samples,
            }

            # Get parent version
            champion = self.versioner.champion
            parent_id = champion.version_id if champion else ""

            # Create artifacts directory for the new version
            artifacts_dir = model_settings.artifacts_dir / model_type
            artifacts_dir.mkdir(parents=True, exist_ok=True)

            # Create the version
            new_version = self.versioner.create_version(
                model_type=model_type,
                artifacts_source=artifacts_dir,
                metrics=metrics,
                training_trigger="drift" if not force else "manual",
                parent_version=parent_id,
                training_samples=buffer_samples,
            )

            self._retrain_count += 1
            self._last_retrain_time = time.time()

            self._log_event("retrain_completed", {
                "version_id": new_version.version_id,
                "model_type": model_type,
                "training_samples": buffer_samples,
            })

            return new_version

        except Exception as e:
            logger.error("Retraining failed: {}", e)
            self._log_event("retrain_failed", {"error": str(e)})
            return None

        finally:
            self._retraining_in_progress = False

    # ── Shadow Evaluation ────────────────────────────────────────

    def start_shadow_evaluation(self, version_id: str) -> bool:
        """Start shadow evaluation for a newly trained model.

        Args:
            version_id: Version ID of the challenger model.

        Returns:
            True if shadow evaluation started successfully.
        """
        champion = self.versioner.champion
        if champion is None:
            logger.warning("No champion model — promoting directly")
            self.versioner.set_champion(version_id)
            return True

        # Set as challenger
        self.versioner.set_challenger(version_id)

        # Start shadow mode
        self.shadow.start_shadow(
            champion_version=champion.version_id,
            challenger_version=version_id,
        )

        self._log_event("shadow_started", {
            "champion": champion.version_id,
            "challenger": version_id,
            "evaluation_hours": self.shadow.evaluation_hours,
        })

        return True

    def check_shadow_completion(self) -> Optional[ComparisonResult]:
        """Check if shadow evaluation is complete and compare models.

        Returns:
            ComparisonResult if evaluation is complete, else None.
        """
        if not self.shadow.is_active or not self.shadow.evaluation_complete:
            return None

        # Stop shadow and get metrics
        shadow_metrics = self.shadow.stop_shadow()

        champion = self.versioner.champion
        challenger = self.versioner.challenger

        if champion is None or challenger is None:
            logger.warning("Champion or challenger missing — cannot compare")
            return None

        # Update challenger metrics from shadow evaluation
        challenger_metrics = shadow_metrics.get("challenger_metrics", {})
        if challenger_metrics:
            challenger.auroc = challenger_metrics.get("auroc", challenger.auroc)
            challenger.auprc = challenger_metrics.get("auprc", challenger.auprc)

        # Compare
        result = self.evaluator.compare(
            champion=champion,
            challenger=challenger,
            evaluation_samples=shadow_metrics.get("labeled_records", 0),
        )

        self._log_event("shadow_completed", {
            "recommendation": result.recommendation,
            "reason": result.reason,
            "auroc_delta": result.auroc_delta,
            "auprc_delta": result.auprc_delta,
        })

        return result

    # ── Promotion Decision ───────────────────────────────────────

    def process_promotion(self, result: ComparisonResult) -> str:
        """Process a promotion decision based on comparison result.

        Args:
            result: Comparison result from check_shadow_completion().

        Returns:
            Final action taken: "promoted", "rejected", "pending_approval".
        """
        if result.recommendation == "REJECT":
            self.versioner.discard_challenger(reason=result.reason)
            self._log_event("challenger_rejected", {
                "version": result.challenger_version,
                "reason": result.reason,
            })
            return "rejected"

        if result.recommendation == "INSUFFICIENT_DATA":
            self._log_event("insufficient_data", {
                "version": result.challenger_version,
                "samples": result.evaluation_samples,
                "required": result.min_samples_required,
            })
            return "insufficient_data"

        # Recommendation is PROMOTE
        if self.evaluator.should_auto_promote(result):
            self.versioner.promote_challenger(promoted_by="auto")
            self._log_event("auto_promoted", {
                "version": result.challenger_version,
                "auroc_delta": result.auroc_delta,
            })

            # Clear buffer after successful promotion
            self.accumulator.clear_buffer()
            self.versioner.cleanup_old_versions()

            return "promoted"
        else:
            # Queue for human approval
            self._pending_approval = result
            self._log_event("pending_approval", {
                "version": result.challenger_version,
                "auroc_delta": result.auroc_delta,
            })
            return "pending_approval"

    def approve_promotion(self, approver: str = "clinician") -> bool:
        """Manually approve a pending challenger promotion.

        Args:
            approver: Identity of the person approving.

        Returns:
            True if promotion was approved and executed.
        """
        if self._pending_approval is None:
            logger.warning("No pending promotion to approve")
            return False

        self.versioner.promote_challenger(promoted_by=approver)
        self._log_event("manually_promoted", {
            "version": self._pending_approval.challenger_version,
            "approved_by": approver,
        })

        self._pending_approval = None
        self.accumulator.clear_buffer()
        self.versioner.cleanup_old_versions()

        return True

    def reject_promotion(self, reason: str = "Manual rejection") -> bool:
        """Manually reject a pending challenger promotion.

        Args:
            reason: Rejection reason.

        Returns:
            True if rejection was processed.
        """
        if self._pending_approval is None:
            logger.warning("No pending promotion to reject")
            return False

        self.versioner.discard_challenger(reason=reason)
        self._log_event("manually_rejected", {
            "version": self._pending_approval.challenger_version,
            "reason": reason,
        })

        self._pending_approval = None
        return True

    # ── Full Lifecycle Loop ──────────────────────────────────────

    async def run_lifecycle(
        self,
        check_interval_s: int | None = None,
    ) -> None:
        """Run the continuous learning lifecycle loop.

        This is the main entrypoint for the orchestrator. It:
        1. Periodically checks for drift
        2. Triggers retraining when conditions are met
        3. Manages shadow evaluation
        4. Processes promotion decisions

        Args:
            check_interval_s: Override drift check interval.
        """
        interval = check_interval_s or continuous_settings.drift_check_interval_s

        logger.info("═══ Continuous Learning Lifecycle Started ═══")
        logger.info("  Check interval: {}s | auto_promote: {}",
                     interval, continuous_settings.auto_promote)

        while True:
            try:
                # Check shadow evaluation completion
                if self.shadow.is_active:
                    result = self.check_shadow_completion()
                    if result:
                        action = self.process_promotion(result)
                        logger.info("Promotion decision: {}", action)

                # Check if retraining is needed
                elif self.should_retrain():
                    new_version = self.trigger_retrain()
                    if new_version:
                        self.start_shadow_evaluation(new_version.version_id)

            except Exception as e:
                logger.error("Lifecycle loop error: {}", e)

            await asyncio.sleep(interval)

    # ── Status & Monitoring ──────────────────────────────────────

    def get_status(self) -> dict:
        """Get comprehensive orchestrator status for dashboard."""
        champion = self.versioner.champion
        challenger = self.versioner.challenger

        return {
            "continuous_enabled": continuous_settings.continuous_enabled,
            "auto_promote": continuous_settings.auto_promote,
            "retraining_in_progress": self._retraining_in_progress,
            "retrain_count": self._retrain_count,
            "last_retrain_time": self._last_retrain_time,
            "last_drift_check": self._last_drift_check,
            "champion": {
                "version_id": champion.version_id if champion else None,
                "auroc": champion.auroc if champion else None,
                "auprc": champion.auprc if champion else None,
            },
            "challenger": {
                "version_id": challenger.version_id if challenger else None,
                "auroc": challenger.auroc if challenger else None,
                "auprc": challenger.auprc if challenger else None,
            } if challenger else None,
            "shadow_evaluation": self.shadow.get_status(),
            "buffer": self.accumulator.get_stats(),
            "pending_approval": self._pending_approval.to_dict()
            if self._pending_approval
            else None,
            "recent_events": self._event_log[-20:],
        }

    def get_event_log(self, limit: int = 50) -> list[dict]:
        """Get recent orchestration events."""
        return self._event_log[-limit:]


# ── Entrypoint ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if not continuous_settings.continuous_enabled:
        logger.error(
            "Continuous learning is not enabled. "
            "Set JEEVAN_CL_CONTINUOUS_ENABLED=true to activate."
        )
        sys.exit(1)

    orchestrator = RetrainOrchestrator()
    asyncio.run(orchestrator.run_lifecycle())
