"""
G_One_Sync AI — Hybrid Ensemble
==================================
Combines XGBoost + BiLSTM + Transformer predictions via learned
or fixed weighting for maximum predictive performance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from config.settings import model_settings
from src.modeling.base_trainer import EvaluationMetrics


class HybridEnsemble:
    """
    Weighted ensemble of XGBoost + BiLSTM + Transformer.

    Supports:
    - Fixed equal/custom weights
    - Learned stacking via logistic regression meta-learner
    - Rank-based averaging
    """

    def __init__(
        self,
        strategy: str = "learned",
        model_names: list[str] = None,
    ):
        """
        Args:
            strategy: 'equal', 'custom', 'learned', or 'rank'
            model_names: Names for each model in the ensemble
        """
        self.strategy = strategy
        self.model_names = model_names or ["xgboost", "bilstm", "transformer"]
        self.weights: Optional[np.ndarray] = None
        self.meta_learner: Optional[LogisticRegression] = None
        self.is_fitted = False

    def fit(
        self,
        val_predictions: dict[str, np.ndarray],
        val_labels: np.ndarray,
        custom_weights: Optional[dict[str, float]] = None,
    ) -> dict[str, float]:
        """
        Fit the ensemble weights on validation predictions.

        Args:
            val_predictions: {model_name: (N,) probabilities}
            val_labels: (N,) true labels
            custom_weights: Optional custom weights for 'custom' strategy

        Returns:
            Dict of model weights
        """
        names = list(val_predictions.keys())
        preds_matrix = np.column_stack([val_predictions[n] for n in names])
        self.model_names = names

        # Per-model AUROC
        for name in names:
            auroc = roc_auc_score(val_labels, val_predictions[name])
            logger.info("  {} val AUROC: {:.4f}", name, auroc)

        if self.strategy == "equal":
            self.weights = np.ones(len(names)) / len(names)

        elif self.strategy == "custom":
            if custom_weights is None:
                raise ValueError("custom_weights required for 'custom' strategy")
            w = np.array([custom_weights.get(n, 1.0) for n in names])
            self.weights = w / w.sum()

        elif self.strategy == "learned":
            # Stacking with logistic regression
            self.meta_learner = LogisticRegression(
                penalty="l2", C=1.0, max_iter=1000, random_state=42
            )
            self.meta_learner.fit(preds_matrix, val_labels)
            self.weights = None  # Meta-learner handles weighting
            logger.info("Meta-learner coefficients: {}",
                        dict(zip(names, self.meta_learner.coef_[0].round(3))))

        elif self.strategy == "rank":
            # Weight by validation AUROC
            aurocs = np.array([roc_auc_score(val_labels, val_predictions[n]) for n in names])
            self.weights = aurocs / aurocs.sum()

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        self.is_fitted = True

        # Compute ensemble AUROC
        ensemble_probs = self.predict(val_predictions)
        ensemble_auroc = roc_auc_score(val_labels, ensemble_probs)
        logger.info("Ensemble ({}) val AUROC: {:.4f}", self.strategy, ensemble_auroc)

        weight_dict = dict(zip(names, self.weights)) if self.weights is not None else {"strategy": "learned"}
        return weight_dict

    def predict(self, predictions: dict[str, np.ndarray]) -> np.ndarray:
        """Combine model predictions into ensemble probability."""
        if not self.is_fitted:
            raise RuntimeError("Ensemble must be fit first")

        names = list(predictions.keys())
        preds_matrix = np.column_stack([predictions[n] for n in names])

        if self.strategy == "learned" and self.meta_learner is not None:
            return self.meta_learner.predict_proba(preds_matrix)[:, 1]
        else:
            return preds_matrix @ self.weights

    def evaluate(
        self,
        predictions: dict[str, np.ndarray],
        labels: np.ndarray,
    ) -> EvaluationMetrics:
        """Evaluate ensemble on a test set."""
        ensemble_probs = self.predict(predictions)
        metrics = EvaluationMetrics(labels, ensemble_probs)

        logger.info("═══ Ensemble Evaluation ═══")
        metrics.print_report(prefix="  [ensemble] ")

        return metrics
