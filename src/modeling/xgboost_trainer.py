"""
G_One_Sync AI — XGBoost GPU Trainer
=====================================
Gradient-boosted tree classifier with full CUDA acceleration.
Uses device="cuda" for GPU-accelerated histogram-based tree construction.
Optimized for RTX 3050 4GB VRAM.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from loguru import logger
from sklearn.utils.class_weight import compute_sample_weight

from config.settings import model_settings
from src.modeling.base_trainer import BaseTrainer, EvaluationMetrics


class XGBoostTrainer(BaseTrainer):
    """
    XGBoost classifier with GPU acceleration.
    Uses `device="cuda"` and `tree_method="hist"` for fast GPU training.
    """

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 8,
        learning_rate: float = 0.05,
        min_child_weight: int = 5,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        gamma: float = 0.1,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        scale_pos_weight: Optional[float] = None,
        early_stopping_rounds: int = 50,
        artifacts_dir: Optional[Path] = None,
        use_mlflow: bool = True,
    ):
        super().__init__(
            model_name="xgboost",
            artifacts_dir=artifacts_dir,
            use_mlflow=use_mlflow,
        )

        self.hyperparams = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "min_child_weight": min_child_weight,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "gamma": gamma,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "scale_pos_weight": scale_pos_weight,
            "early_stopping_rounds": early_stopping_rounds,
            "device": str(self.device),
            "tree_method": "hist",
        }

        self.feature_names: list[str] = []
        self.feature_importances_: Optional[np.ndarray] = None

    def _build_model(self, **kwargs) -> xgb.XGBClassifier:
        """Build XGBoost classifier with GPU support."""
        params = {
            "n_estimators": self.hyperparams["n_estimators"],
            "max_depth": self.hyperparams["max_depth"],
            "learning_rate": self.hyperparams["learning_rate"],
            "min_child_weight": self.hyperparams["min_child_weight"],
            "subsample": self.hyperparams["subsample"],
            "colsample_bytree": self.hyperparams["colsample_bytree"],
            "gamma": self.hyperparams["gamma"],
            "reg_alpha": self.hyperparams["reg_alpha"],
            "reg_lambda": self.hyperparams["reg_lambda"],
            "objective": "binary:logistic",
            "eval_metric": ["logloss", "auc", "aucpr"],
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 1,
        }

        # GPU acceleration
        if self.device.type == "cuda":
            params["device"] = "cuda"
            logger.info("🔥 XGBoost using CUDA GPU acceleration")
        else:
            params["device"] = "cpu"

        # Handle class imbalance
        if self.hyperparams["scale_pos_weight"]:
            params["scale_pos_weight"] = self.hyperparams["scale_pos_weight"]

        model = xgb.XGBClassifier(**params)
        logger.info("XGBoost model built: {} estimators, depth={}, lr={}",
                     params["n_estimators"], params["max_depth"], params["learning_rate"])

        return model

    def _prepare_data(
        self,
        data: pd.DataFrame,
        target_column: str = "deterioration_next_12h",
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Extract features and labels from DataFrame."""
        exclude = {"patient_id", "hour_from_admission", target_column}
        feature_cols = [c for c in data.columns if c not in exclude]
        X = data[feature_cols].values.astype(np.float32)
        y = data[target_column].values.astype(int)
        return X, y, feature_cols

    def _train_impl(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        **kwargs,
    ) -> dict[str, list[float]]:
        """Train XGBoost with early stopping on validation set."""

        target_col = kwargs.get("target_column", "deterioration_next_12h")

        X_train, y_train, feature_cols = self._prepare_data(train_data, target_col)
        X_val, y_val, _ = self._prepare_data(val_data, target_col)
        self.feature_names = feature_cols

        # Auto-compute scale_pos_weight if not set
        if not self.hyperparams["scale_pos_weight"]:
            neg = (y_train == 0).sum()
            pos = max((y_train == 1).sum(), 1)
            self.model.set_params(scale_pos_weight=neg / pos)
            logger.info("Auto scale_pos_weight: {:.2f} (neg/pos = {}/{})", neg / pos, neg, pos)

        # Compute sample weights for imbalanced data
        sample_weights = compute_sample_weight("balanced", y_train)

        logger.info(
            "Training XGBoost: {} train, {} val, {} features",
            len(X_train), len(X_val), len(feature_cols),
        )

        # Train with eval set for early stopping
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            sample_weight=sample_weights,
            verbose=50,  # Print every 50 rounds
        )

        # Extract training history
        results = self.model.evals_result()
        history = {}
        if "validation_0" in results:
            history["train_logloss"] = results["validation_0"]["logloss"]
            history["train_auc"] = results["validation_0"]["auc"]
        if "validation_1" in results:
            history["val_logloss"] = results["validation_1"]["logloss"]
            history["val_auc"] = results["validation_1"]["auc"]

        # Feature importance
        self.feature_importances_ = self.model.feature_importances_

        # Log best iteration
        best_iter = self.model.best_iteration if hasattr(self.model, "best_iteration") else self.hyperparams["n_estimators"]
        logger.info("Best iteration: {} / {}", best_iter, self.hyperparams["n_estimators"])

        return history

    def _predict_proba_impl(self, data) -> np.ndarray:
        """Predict deterioration probabilities."""
        if isinstance(data, pd.DataFrame):
            X, _, _ = self._prepare_data(data)
        elif isinstance(data, np.ndarray):
            X = data
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self, top_n: int = 30) -> pd.DataFrame:
        """Get feature importance ranking."""
        if self.feature_importances_ is None:
            raise RuntimeError("Model must be trained first")

        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": self.feature_importances_,
        }).sort_values("importance", ascending=False)

        logger.info("Top {} features:", top_n)
        for _, row in importance_df.head(top_n).iterrows():
            logger.info("  {:.4f} — {}", row["importance"], row["feature"])

        return importance_df

    def _save_impl(self, path: Path) -> None:
        """Save XGBoost model in native format."""
        self.model.save_model(str(path.with_suffix(".json")))
        # Also save feature names
        import json
        with open(path.with_suffix(".features.json"), "w") as f:
            json.dump(self.feature_names, f)

    def load_model(self, path: Path) -> None:
        """Load XGBoost model from native format."""
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(path.with_suffix(".json")))
        self.is_trained = True
        # Load feature names
        import json
        feat_path = path.with_suffix(".features.json")
        if feat_path.exists():
            with open(feat_path) as f:
                self.feature_names = json.load(f)
        logger.info("XGBoost model loaded ← {}", path)
