"""
G_One_Sync AI — Model Version Manager
=========================================
Manages model versions with timestamped snapshots, metrics
history, and safe rollback. Every trained model is stored
as an immutable version — rollback is always one config change.

Clinical rationale:
    In a medical device context, every model version must be traceable.
    The versioner creates an audit trail linking each version to its
    training data, hyperparameters, evaluation metrics, and promotion
    decision (human-approved or auto-promoted).
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from loguru import logger

from config.settings import continuous_settings, model_settings


@dataclass
class ModelVersion:
    """Immutable snapshot of a trained model version."""

    version_id: str
    model_type: str  # "xgboost", "bilstm", "transformer", "ensemble"
    created_at: float = field(default_factory=time.time)
    artifacts_path: str = ""
    training_data_hash: str = ""
    training_samples: int = 0

    # Evaluation metrics
    auroc: float = 0.0
    auprc: float = 0.0
    f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    specificity: float = 0.0
    calibration_ece: float = 0.0

    # Lineage
    parent_version: str = ""  # Version this was fine-tuned from
    training_trigger: str = ""  # "initial", "drift", "scheduled", "manual"
    promoted_at: Optional[float] = None
    promoted_by: str = ""  # "auto" or approver identity
    is_champion: bool = False
    is_challenger: bool = False
    is_retired: bool = False
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ModelVersion:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class ModelVersionManager:
    """
    Manages the lifecycle of model versions.

    Responsibilities:
        - Create new versions with metadata and artifacts
        - Track champion/challenger designations
        - Promote challengers to champion (with audit trail)
        - Rollback to previous versions
        - Enforce max version retention
        - Persist version registry to disk
    """

    REGISTRY_FILE = "version_registry.json"

    def __init__(self, versions_dir: Path | None = None):
        self.versions_dir = versions_dir or continuous_settings.model_versions_dir
        self.versions_dir.mkdir(parents=True, exist_ok=True)

        self.versions: dict[str, ModelVersion] = {}
        self._load_registry()

        logger.info(
            "ModelVersionManager initialized: {} versions, dir={}",
            len(self.versions),
            self.versions_dir,
        )

    def _load_registry(self) -> None:
        """Load version registry from disk."""
        registry_path = self.versions_dir / self.REGISTRY_FILE
        if registry_path.exists():
            with open(registry_path) as f:
                data = json.load(f)
            for v_data in data.get("versions", []):
                version = ModelVersion.from_dict(v_data)
                self.versions[version.version_id] = version
            logger.info("Loaded {} versions from registry", len(self.versions))

    def _save_registry(self) -> None:
        """Persist version registry to disk."""
        registry_path = self.versions_dir / self.REGISTRY_FILE
        data = {
            "versions": [v.to_dict() for v in self.versions.values()],
            "updated_at": time.time(),
        }
        with open(registry_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    # ── Version Creation ─────────────────────────────────────────

    def create_version(
        self,
        model_type: str,
        artifacts_source: Path,
        metrics: dict[str, float],
        training_trigger: str = "manual",
        parent_version: str = "",
        training_samples: int = 0,
        training_data_hash: str = "",
        notes: str = "",
    ) -> ModelVersion:
        """Create a new model version from trained artifacts.

        Args:
            model_type: Model architecture name.
            artifacts_source: Path to the trained model artifacts.
            metrics: Evaluation metrics dict (auroc, auprc, f1, etc.).
            training_trigger: What triggered this training run.
            parent_version: Version ID this was fine-tuned from.
            training_samples: Number of training samples used.
            training_data_hash: Hash of training data for reproducibility.
            notes: Human-readable notes.

        Returns:
            The created ModelVersion.
        """
        # Use high-resolution timestamp to avoid collisions in rapid calls
        version_id = f"{model_type}-{int(time.time() * 1_000_000)}"

        # Create version directory and copy artifacts
        version_dir = self.versions_dir / version_id
        version_dir.mkdir(parents=True, exist_ok=True)

        if artifacts_source.is_dir():
            shutil.copytree(artifacts_source, version_dir / "artifacts", dirs_exist_ok=True)
        elif artifacts_source.is_file():
            shutil.copy2(artifacts_source, version_dir / artifacts_source.name)

        version = ModelVersion(
            version_id=version_id,
            model_type=model_type,
            artifacts_path=str(version_dir),
            training_data_hash=training_data_hash,
            training_samples=training_samples,
            auroc=metrics.get("auroc", 0.0),
            auprc=metrics.get("auprc", 0.0),
            f1=metrics.get("f1", 0.0),
            precision=metrics.get("precision", 0.0),
            recall=metrics.get("recall", 0.0),
            specificity=metrics.get("specificity", 0.0),
            calibration_ece=metrics.get("calibration_ece", 0.0),
            parent_version=parent_version,
            training_trigger=training_trigger,
            notes=notes,
        )

        # Save metadata alongside artifacts
        meta_path = version_dir / "version_meta.json"
        with open(meta_path, "w") as f:
            json.dump(version.to_dict(), f, indent=2, default=str)

        self.versions[version_id] = version
        self._save_registry()

        logger.info(
            "Model version created: {} | AUROC={:.4f} AUPRC={:.4f} | trigger={}",
            version_id,
            version.auroc,
            version.auprc,
            training_trigger,
        )

        return version

    # ── Champion / Challenger Management ──────────────────────────

    @property
    def champion(self) -> Optional[ModelVersion]:
        """Get the current champion model version."""
        for v in self.versions.values():
            if v.is_champion and not v.is_retired:
                return v
        return None

    @property
    def challenger(self) -> Optional[ModelVersion]:
        """Get the current challenger model version (if any)."""
        for v in self.versions.values():
            if v.is_challenger and not v.is_retired:
                return v
        return None

    def set_champion(self, version_id: str) -> ModelVersion:
        """Designate a version as the champion (production) model.

        Args:
            version_id: Version to promote.

        Returns:
            The promoted version.
        """
        if version_id not in self.versions:
            raise ValueError(f"Version '{version_id}' not found")

        # Demote current champion
        for v in self.versions.values():
            if v.is_champion:
                v.is_champion = False

        version = self.versions[version_id]
        version.is_champion = True
        version.is_challenger = False
        version.promoted_at = time.time()
        self._save_registry()

        logger.info("Champion model set: {}", version_id)
        return version

    def set_challenger(self, version_id: str) -> ModelVersion:
        """Designate a version as the challenger (shadow evaluation).

        Args:
            version_id: Version to evaluate as challenger.

        Returns:
            The challenger version.
        """
        if version_id not in self.versions:
            raise ValueError(f"Version '{version_id}' not found")

        # Remove existing challenger
        for v in self.versions.values():
            if v.is_challenger:
                v.is_challenger = False

        version = self.versions[version_id]
        version.is_challenger = True
        self._save_registry()

        logger.info("Challenger model set: {}", version_id)
        return version

    def promote_challenger(self, promoted_by: str = "auto") -> Optional[ModelVersion]:
        """Promote the current challenger to champion.

        The old champion is retired. The challenger becomes the new champion.

        Args:
            promoted_by: Identifier of who approved the promotion.

        Returns:
            The newly promoted champion, or None if no challenger exists.
        """
        current_challenger = self.challenger
        if current_challenger is None:
            logger.warning("No challenger to promote")
            return None

        current_champion = self.champion
        if current_champion:
            current_champion.is_champion = False
            current_champion.is_retired = True
            logger.info("Previous champion retired: {}", current_champion.version_id)

        current_challenger.is_champion = True
        current_challenger.is_challenger = False
        current_challenger.promoted_at = time.time()
        current_challenger.promoted_by = promoted_by
        self._save_registry()

        logger.info(
            "Challenger promoted to champion: {} (by {})",
            current_challenger.version_id,
            promoted_by,
        )
        return current_challenger

    def discard_challenger(self, reason: str = "") -> None:
        """Discard the current challenger (failed evaluation)."""
        current_challenger = self.challenger
        if current_challenger:
            current_challenger.is_challenger = False
            current_challenger.is_retired = True
            current_challenger.notes += f" Discarded: {reason}"
            self._save_registry()
            logger.info(
                "Challenger discarded: {} | reason: {}",
                current_challenger.version_id,
                reason,
            )

    # ── Rollback ─────────────────────────────────────────────────

    def rollback(self, target_version_id: str) -> ModelVersion:
        """Rollback to a previous model version.

        Args:
            target_version_id: Version to rollback to.

        Returns:
            The restored champion version.
        """
        if target_version_id not in self.versions:
            raise ValueError(f"Version '{target_version_id}' not found")

        target = self.versions[target_version_id]
        if target.is_retired:
            target.is_retired = False  # Un-retire for rollback

        return self.set_champion(target_version_id)

    # ── Version Listing & Cleanup ────────────────────────────────

    def list_versions(
        self,
        model_type: str | None = None,
        include_retired: bool = False,
    ) -> list[ModelVersion]:
        """List all model versions, optionally filtered."""
        versions = list(self.versions.values())
        if model_type:
            versions = [v for v in versions if v.model_type == model_type]
        if not include_retired:
            versions = [v for v in versions if not v.is_retired]
        return sorted(versions, key=lambda v: v.created_at, reverse=True)

    def cleanup_old_versions(self) -> int:
        """Remove old retired versions beyond max_model_versions."""
        max_versions = continuous_settings.max_model_versions
        retired = [
            v for v in self.versions.values()
            if v.is_retired and not v.is_champion
        ]
        retired.sort(key=lambda v: v.created_at)

        removed = 0
        while len(self.versions) > max_versions and retired:
            old = retired.pop(0)
            # Remove artifacts
            version_dir = Path(old.artifacts_path)
            if version_dir.exists():
                shutil.rmtree(version_dir, ignore_errors=True)
            del self.versions[old.version_id]
            removed += 1

        if removed:
            self._save_registry()
            logger.info("Cleaned up {} old model versions", removed)

        return removed

    def get_version_history(self) -> list[dict]:
        """Get a summary timeline of all versions for dashboard display."""
        return [
            {
                "version_id": v.version_id,
                "model_type": v.model_type,
                "created_at": v.created_at,
                "auroc": v.auroc,
                "auprc": v.auprc,
                "is_champion": v.is_champion,
                "is_challenger": v.is_challenger,
                "is_retired": v.is_retired,
                "training_trigger": v.training_trigger,
                "promoted_at": v.promoted_at,
            }
            for v in sorted(self.versions.values(), key=lambda x: x.created_at)
        ]
