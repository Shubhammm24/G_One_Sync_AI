"""
G_One_Sync AI — Optuna Hyperparameter Optimization
=====================================================
Automated HPO for XGBoost, BiLSTM, and Transformer models.
Uses Optuna with pruning for efficient search on GPU.
"""

from __future__ import annotations

from typing import Optional, Callable

import numpy as np
import optuna
from loguru import logger
from sklearn.metrics import roc_auc_score

from config.settings import model_settings


class HPOptimizer:
    """
    Optuna-based hyperparameter optimization for all model types.
    Supports GPU-accelerated training during search.
    """

    def __init__(
        self,
        n_trials: int = 50,
        direction: str = "maximize",
        metric: str = "val_auroc",
        study_name: str = "g-one-sync-hpo",
        pruner: Optional[optuna.pruners.BasePruner] = None,
    ):
        self.n_trials = n_trials
        self.direction = direction
        self.metric = metric

        self.pruner = pruner or optuna.pruners.MedianPruner(
            n_warmup_steps=5, n_startup_trials=5
        )

        self.study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            pruner=self.pruner,
        )

        logger.info(
            "HPO initialized: {} trials, metric='{}', direction='{}'",
            n_trials, metric, direction,
        )

    def optimize_xgboost(
        self,
        train_data: tuple[np.ndarray, np.ndarray],
        val_data: tuple[np.ndarray, np.ndarray],
    ) -> dict:
        """Optimize XGBoost hyperparameters."""

        def objective(trial: optuna.Trial) -> float:
            from src.modeling.xgboost_trainer import XGBoostTrainer

            trainer = XGBoostTrainer(
                n_estimators=trial.suggest_int("n_estimators", 100, 1000),
                max_depth=trial.suggest_int("max_depth", 3, 12),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                gamma=trial.suggest_float("gamma", 0.0, 1.0),
                reg_alpha=trial.suggest_float("reg_alpha", 0.0, 2.0),
                reg_lambda=trial.suggest_float("reg_lambda", 0.5, 5.0),
                use_mlflow=False,
            )

            metrics = trainer.train(train_data, val_data)
            return metrics.auroc

        self.study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        return self._get_results("xgboost")

    def optimize_bilstm(
        self,
        train_data: tuple[np.ndarray, np.ndarray],
        val_data: tuple[np.ndarray, np.ndarray],
        input_size: int = 15,
    ) -> dict:
        """Optimize BiLSTM hyperparameters."""

        def objective(trial: optuna.Trial) -> float:
            from src.modeling.lstm_trainer import BiLSTMTrainer

            trainer = BiLSTMTrainer(
                input_size=input_size,
                hidden_size=trial.suggest_categorical("hidden_size", [64, 128, 256]),
                num_layers=trial.suggest_int("num_layers", 1, 3),
                dropout=trial.suggest_float("dropout", 0.1, 0.5),
                learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
                weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
                batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]),
                epochs=30,  # Shorter for HPO
                patience=7,
                use_mlflow=False,
            )

            metrics = trainer.train(train_data, val_data)
            return metrics.auroc

        self.study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        return self._get_results("bilstm")

    def optimize_transformer(
        self,
        train_data: tuple[np.ndarray, np.ndarray],
        val_data: tuple[np.ndarray, np.ndarray],
        input_size: int = 15,
    ) -> dict:
        """Optimize Transformer hyperparameters."""

        def objective(trial: optuna.Trial) -> float:
            from src.modeling.transformer_trainer import TransformerTrainer

            d_model = trial.suggest_categorical("d_model", [32, 64, 128])
            nhead = trial.suggest_categorical("nhead", [2, 4, 8])
            # Ensure d_model is divisible by nhead
            while d_model % nhead != 0:
                nhead = nhead // 2

            trainer = TransformerTrainer(
                input_size=input_size,
                d_model=d_model,
                nhead=nhead,
                num_layers=trial.suggest_int("num_layers", 1, 4),
                dim_feedforward=trial.suggest_categorical("dim_feedforward", [64, 128, 256]),
                dropout=trial.suggest_float("dropout", 0.1, 0.4),
                learning_rate=trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
                weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
                batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]),
                epochs=30,
                patience=7,
                use_mlflow=False,
            )

            metrics = trainer.train(train_data, val_data)
            return metrics.auroc

        self.study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        return self._get_results("transformer")

    def _get_results(self, model_name: str) -> dict:
        """Extract optimization results."""
        best = self.study.best_trial
        results = {
            "model": model_name,
            "best_value": best.value,
            "best_params": best.params,
            "n_trials": len(self.study.trials),
            "n_pruned": len([t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        }

        logger.info("═══ HPO Results: {} ═══", model_name)
        logger.info("  Best {}: {:.4f}", self.metric, best.value)
        logger.info("  Best params: {}", best.params)
        logger.info("  Trials: {} total, {} pruned", results["n_trials"], results["n_pruned"])

        return results
