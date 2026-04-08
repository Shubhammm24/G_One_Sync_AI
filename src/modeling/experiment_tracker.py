"""
G_One_Sync AI — MLflow Experiment Tracker
===========================================
Centralized experiment tracking with MLflow for all model types.
Logs params, metrics, artifacts, and model registry.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.pytorch
from loguru import logger

from config.settings import model_settings


class ExperimentTracker:
    """
    MLflow-based experiment tracking wrapper.
    Provides a consistent interface for logging across all model types.
    """

    def __init__(
        self,
        experiment_name: str = "g-one-sync-deterioration",
        tracking_uri: Optional[str] = None,
    ):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or model_settings.mlflow_tracking_uri

        # Setup MLflow
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

        logger.info(
            "MLflow tracking: experiment='{}', uri='{}'",
            self.experiment_name, self.tracking_uri,
        )

    def start_run(
        self,
        run_name: str,
        tags: Optional[dict[str, str]] = None,
    ) -> mlflow.ActiveRun:
        """Start a new MLflow run."""
        run = mlflow.start_run(run_name=run_name, tags=tags or {})
        logger.info("MLflow run started: {} ({})", run_name, run.info.run_id[:8])
        return run

    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters."""
        # MLflow only accepts strings/numbers, flatten nested dicts
        flat = self._flatten_dict(params)
        mlflow.log_params(flat)
        logger.debug("Logged {} params", len(flat))

    def log_metrics(self, metrics: dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics (optionally at a specific step for epoch-level tracking)."""
        mlflow.log_metrics(metrics, step=step)

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log a single metric."""
        mlflow.log_metric(key, value, step=step)

    def log_artifact(self, filepath: str | Path) -> None:
        """Log a file artifact (model, plot, config, etc.)."""
        mlflow.log_artifact(str(filepath))

    def log_dict_artifact(self, data: dict, filename: str) -> None:
        """Log a dict as a JSON artifact."""
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f, indent=2, default=str)
            f.flush()
            mlflow.log_artifact(f.name, artifact_path="metadata")

    def log_model_xgboost(self, model, artifact_path: str = "model") -> None:
        """Log an XGBoost model."""
        mlflow.xgboost.log_model(model, artifact_path=artifact_path)

    def log_model_pytorch(self, model, artifact_path: str = "model") -> None:
        """Log a PyTorch model."""
        mlflow.pytorch.log_model(model, artifact_path=artifact_path)

    def log_figure(self, figure, artifact_file: str) -> None:
        """Log a matplotlib/plotly figure."""
        mlflow.log_figure(figure, artifact_file)

    def end_run(self, status: str = "FINISHED") -> None:
        """End the current MLflow run."""
        mlflow.end_run(status=status)
        logger.info("MLflow run ended: {}", status)

    def get_best_run(
        self,
        metric: str = "val_auroc",
        ascending: bool = False,
    ) -> Optional[mlflow.entities.Run]:
        """Get the best run by a metric."""
        runs = mlflow.search_runs(
            experiment_names=[self.experiment_name],
            order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
            max_results=1,
        )
        if len(runs) > 0:
            return runs.iloc[0]
        return None

    @staticmethod
    def _flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
        """Flatten nested dict for MLflow param logging."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(ExperimentTracker._flatten_dict(v, new_key, sep).items())
            else:
                # MLflow params must be strings of length <= 500
                items.append((new_key, str(v)[:500]))
        return dict(items)
