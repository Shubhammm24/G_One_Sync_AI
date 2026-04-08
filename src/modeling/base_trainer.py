"""
G_One_Sync AI — Base Trainer
==============================
Abstract base class for all model trainers.
Provides common interface for training, evaluation, saving, and MLflow logging.
"""

from __future__ import annotations

import abc
import json
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)

from config.settings import model_settings
from src.modeling.experiment_tracker import ExperimentTracker


class EvaluationMetrics:
    """Container for model evaluation metrics."""

    def __init__(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: float = 0.5,
    ):
        self.y_true = y_true
        self.y_prob = y_prob
        self.threshold = threshold
        self.y_pred = (y_prob >= threshold).astype(int)

        # Core metrics
        self.auroc = float(roc_auc_score(y_true, y_prob))
        self.auprc = float(average_precision_score(y_true, y_prob))
        self.accuracy = float(accuracy_score(y_true, self.y_pred))
        self.precision = float(precision_score(y_true, self.y_pred, zero_division=0))
        self.recall = float(recall_score(y_true, self.y_pred, zero_division=0))
        self.f1 = float(f1_score(y_true, self.y_pred, zero_division=0))
        self.specificity = self._compute_specificity()

        # Curves
        self.fpr, self.tpr, self.roc_thresholds = roc_curve(y_true, y_prob)
        self.pr_precision, self.pr_recall, self.pr_thresholds = precision_recall_curve(y_true, y_prob)

        # Confusion matrix
        self.cm = confusion_matrix(y_true, self.y_pred)

    def _compute_specificity(self) -> float:
        tn = int(((self.y_true == 0) & (self.y_pred == 0)).sum())
        fp = int(((self.y_true == 0) & (self.y_pred == 1)).sum())
        return tn / max(tn + fp, 1)

    def to_dict(self) -> dict[str, float]:
        return {
            "auroc": self.auroc,
            "auprc": self.auprc,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "specificity": self.specificity,
            "threshold": self.threshold,
        }

    def print_report(self, prefix: str = "") -> None:
        logger.info("{}AUROC: {:.4f} | AUPRC: {:.4f}", prefix, self.auroc, self.auprc)
        logger.info("{}Precision: {:.4f} | Recall: {:.4f} | F1: {:.4f}",
                     prefix, self.precision, self.recall, self.f1)
        logger.info("{}Specificity: {:.4f} | Accuracy: {:.4f}",
                     prefix, self.specificity, self.accuracy)
        logger.info("{}Confusion Matrix:\n{}", prefix, self.cm)

    def find_optimal_threshold(self, strategy: str = "f1") -> float:
        """Find optimal classification threshold."""
        if strategy == "f1":
            thresholds = np.linspace(0.1, 0.9, 81)
            best_f1, best_t = 0.0, 0.5
            for t in thresholds:
                preds = (self.y_prob >= t).astype(int)
                f = f1_score(self.y_true, preds, zero_division=0)
                if f > best_f1:
                    best_f1, best_t = f, t
            return float(best_t)
        elif strategy == "youden":
            # Youden's J = sensitivity + specificity - 1
            j_scores = self.tpr - self.fpr
            return float(self.roc_thresholds[np.argmax(j_scores)])
        return 0.5


class BaseTrainer(abc.ABC):
    """
    Abstract base trainer for all G_One_Sync models.
    Subclasses implement: _build_model, _train_impl, _predict_proba_impl.
    """

    def __init__(
        self,
        model_name: str,
        artifacts_dir: Optional[Path] = None,
        use_mlflow: bool = True,
    ):
        self.model_name = model_name
        self.artifacts_dir = artifacts_dir or model_settings.artifacts_dir / model_name
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.use_mlflow = use_mlflow

        # Device detection
        self.device = torch.device(
            model_settings.gpu_device
            if model_settings.use_gpu and torch.cuda.is_available()
            else "cpu"
        )

        # State
        self.model = None
        self.is_trained = False
        self.train_metrics: Optional[EvaluationMetrics] = None
        self.val_metrics: Optional[EvaluationMetrics] = None
        self.test_metrics: Optional[EvaluationMetrics] = None
        self.training_time: float = 0.0
        self.hyperparams: dict[str, Any] = {}

        # MLflow tracker
        self.tracker = ExperimentTracker() if use_mlflow else None

        logger.info(
            "Trainer '{}' initialized | device={} | artifacts={}",
            model_name, self.device, self.artifacts_dir,
        )

    @abc.abstractmethod
    def _build_model(self, **kwargs) -> Any:
        """Build the model architecture. Implemented by subclasses."""
        ...

    @abc.abstractmethod
    def _train_impl(self, train_data, val_data, **kwargs) -> dict[str, list[float]]:
        """Train the model. Returns training history dict."""
        ...

    @abc.abstractmethod
    def _predict_proba_impl(self, data) -> np.ndarray:
        """Predict probabilities. Returns (N,) array of P(deterioration=1)."""
        ...

    def train(
        self,
        train_data,
        val_data,
        run_name: Optional[str] = None,
        **kwargs,
    ) -> EvaluationMetrics:
        """
        Full training loop with logging and evaluation.
        """
        run_name = run_name or f"{self.model_name}-{int(time.time())}"

        # Start MLflow run
        if self.tracker:
            self.tracker.start_run(
                run_name=run_name,
                tags={"model_type": self.model_name, "device": str(self.device)},
            )
            self.tracker.log_params(self.hyperparams)

        try:
            logger.info("╔═══ Training {} ═══╗", self.model_name)
            start = time.time()

            # Build model
            self.model = self._build_model(**kwargs)

            # Train
            history = self._train_impl(train_data, val_data, **kwargs)
            self.training_time = time.time() - start
            self.is_trained = True

            logger.info("Training completed in {:.1f}s", self.training_time)

            # Log training history to MLflow
            if self.tracker and history:
                for epoch, metrics in enumerate(
                    zip(*[history[k] for k in history])
                ):
                    step_metrics = {k: v for k, v in zip(history.keys(), metrics)}
                    self.tracker.log_metrics(step_metrics, step=epoch)

            # Evaluate on validation set
            self.val_metrics = self.evaluate(val_data, split_name="val")

            if self.tracker:
                self.tracker.log_metrics(
                    {f"val_{k}": v for k, v in self.val_metrics.to_dict().items()}
                )
                self.tracker.log_metric("training_time_s", self.training_time)

            return self.val_metrics

        except Exception as e:
            logger.error("Training failed: {}", e)
            if self.tracker:
                self.tracker.end_run(status="FAILED")
            raise

    def evaluate(
        self,
        data,
        split_name: str = "test",
    ) -> EvaluationMetrics:
        """Evaluate model on a dataset split."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before evaluation")

        y_prob = self._predict_proba_impl(data)

        # Extract true labels
        if isinstance(data, pd.DataFrame):
            y_true = data["deterioration_next_12h"].values
        elif isinstance(data, tuple):
            y_true = data[1] if isinstance(data[1], np.ndarray) else data[1].numpy()
        elif hasattr(data, "dataset"):
            y_true = data.dataset.labels.numpy()
        else:
            raise ValueError(f"Cannot extract labels from {type(data)}")

        metrics = EvaluationMetrics(y_true, y_prob)
        logger.info("— {} Evaluation —", split_name.upper())
        metrics.print_report(prefix=f"  [{split_name}] ")

        return metrics

    def save_model(self, path: Optional[Path] = None) -> Path:
        """Save the trained model to disk."""
        path = path or self.artifacts_dir / f"{self.model_name}_model"
        path.parent.mkdir(parents=True, exist_ok=True)
        self._save_impl(path)
        logger.info("Model saved → {}", path)
        return path

    def _save_impl(self, path: Path) -> None:
        """Default save implementation. Override for custom formats."""
        import joblib
        joblib.dump(self.model, path.with_suffix(".joblib"))

    def load_model(self, path: Path) -> None:
        """Load a trained model from disk."""
        import joblib
        self.model = joblib.load(path.with_suffix(".joblib"))
        self.is_trained = True
        logger.info("Model loaded ← {}", path)

    def finalize_run(self) -> None:
        """Save model, log to MLflow, and end run."""
        if not self.is_trained:
            return

        # Save model locally
        model_path = self.save_model()

        # Save metrics
        if self.val_metrics:
            metrics_path = self.artifacts_dir / "val_metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(self.val_metrics.to_dict(), f, indent=2)

        # Log to MLflow — find actual saved files (with correct extensions)
        if self.tracker:
            # Log all model files matching the base name
            base_name = model_path.stem  # e.g. "xgboost_model"
            for f in self.artifacts_dir.iterdir():
                if f.is_file() and f.stem.startswith(base_name.split(".")[0]):
                    try:
                        self.tracker.log_artifact(str(f))
                    except Exception as e:
                        logger.warning("Failed to log artifact {}: {}", f.name, e)

            if self.val_metrics:
                try:
                    self.tracker.log_artifact(str(metrics_path))
                except Exception as e:
                    logger.warning("Failed to log metrics artifact: {}", e)

            self.tracker.end_run()

        logger.info("✅ Run finalized for '{}'", self.model_name)

