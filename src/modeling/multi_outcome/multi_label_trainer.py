"""
G_One_Sync AI — Multi-Label Trainer
=======================================
BaseTrainer subclass for multi-task learning across
multiple clinical outcome prediction heads.

Implements:
    - Weighted multi-task BCE loss
    - Per-outcome evaluation metrics
    - Checkpoint saving with hyperparams
    - Mixed precision training (RTX 3050 compatible)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset

from config.settings import model_settings, multi_outcome_settings
from src.modeling.base_trainer import BaseTrainer, EvaluationMetrics
from src.modeling.multi_outcome.label_definitions import (
    NUM_OUTCOMES,
    OUTCOME_COLUMNS,
    OUTCOME_REGISTRY,
)
from src.modeling.multi_outcome.multi_head_model import MultiHeadPredictionModel


class MultiLabelTrainer(BaseTrainer):
    """
    Multi-task trainer for simultaneous prediction of multiple
    clinical outcomes using a shared Transformer encoder.

    Inherits from BaseTrainer and implements all abstract methods.
    """

    def __init__(
        self,
        artifacts_dir: Optional[Path] = None,
        use_mlflow: bool = True,
        outcome_names: Optional[list[str]] = None,
        task_weights: Optional[list[float]] = None,
    ):
        super().__init__(
            model_name="multi_outcome",
            artifacts_dir=artifacts_dir or multi_outcome_settings.multi_artifacts_dir,
            use_mlflow=use_mlflow,
        )

        self.outcome_names = outcome_names or OUTCOME_COLUMNS[:NUM_OUTCOMES]
        self.task_weights = task_weights or multi_outcome_settings.task_weights
        self.num_outcomes = len(self.outcome_names)

        # Hyperparams (logged to MLflow)
        self.hyperparams = {
            "model_type": "multi_head_transformer",
            "num_outcomes": self.num_outcomes,
            "d_model": multi_outcome_settings.shared_d_model,
            "nhead": multi_outcome_settings.shared_nhead,
            "num_layers": multi_outcome_settings.shared_num_layers,
            "head_hidden_dim": multi_outcome_settings.head_hidden_dim,
            "head_dropout": multi_outcome_settings.head_dropout,
            "epochs": multi_outcome_settings.multi_epochs,
            "learning_rate": multi_outcome_settings.multi_learning_rate,
            "batch_size": multi_outcome_settings.multi_batch_size,
            "task_weights": str(self.task_weights),
            "outcomes": str(self.outcome_names),
        }

        # Per-outcome evaluation storage
        self.outcome_metrics: dict[str, EvaluationMetrics] = {}

        logger.info(
            "MultiLabelTrainer: {} outcomes, weights={}",
            self.num_outcomes, self.task_weights,
        )

    def _build_model(self, input_size: int = 11, **kwargs) -> MultiHeadPredictionModel:
        """Build the multi-head prediction model."""
        model = MultiHeadPredictionModel(
            input_size=input_size,
            num_outcomes=self.num_outcomes,
            d_model=multi_outcome_settings.shared_d_model,
            nhead=multi_outcome_settings.shared_nhead,
            num_layers=multi_outcome_settings.shared_num_layers,
            head_hidden_dim=multi_outcome_settings.head_hidden_dim,
            dropout=0.2,
            head_dropout=multi_outcome_settings.head_dropout,
            outcome_names=self.outcome_names,
        )
        model = model.to(self.device)
        self.hyperparams["input_size"] = input_size
        return model

    def _compute_multi_task_loss(
        self,
        logits: dict[str, torch.Tensor],
        labels: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Weighted multi-task BCE loss.

        loss = sum_i( alpha_i * BCE_weighted(logits_i, labels_i) )

        Each outcome uses class-weighted BCE to handle imbalance.
        """
        total_loss = torch.tensor(0.0, device=self.device)

        for i, name in enumerate(self.outcome_names):
            if name not in logits or name not in labels:
                continue

            outcome_labels = labels[name]
            outcome_logits = logits[name]

            # Compute class weight for this outcome
            n_pos = outcome_labels.sum().clamp(min=1.0)
            n_neg = (outcome_labels.numel() - n_pos).clamp(min=1.0)
            pos_weight = (n_neg / n_pos).clamp(max=20.0)

            criterion = nn.BCEWithLogitsLoss(
                pos_weight=pos_weight.unsqueeze(0),
            )

            task_loss = criterion(outcome_logits, outcome_labels)
            task_weight = self.task_weights[i] if i < len(self.task_weights) else 1.0
            total_loss = total_loss + task_weight * task_loss

        return total_loss

    def _prepare_dataloaders(
        self,
        data: tuple | pd.DataFrame,
        batch_size: int | None = None,
        shuffle: bool = True,
    ) -> DataLoader:
        """Prepare DataLoader from training data."""
        batch_size = batch_size or multi_outcome_settings.multi_batch_size

        if isinstance(data, tuple):
            X, Y = data[0], data[1]
            if isinstance(X, pd.DataFrame):
                X = X.values
            if isinstance(Y, pd.DataFrame):
                Y = Y.values
            elif isinstance(Y, np.ndarray) and Y.ndim == 1:
                # Single label — expand to multi-outcome with zeros
                Y_multi = np.zeros((len(Y), self.num_outcomes), dtype=np.float32)
                Y_multi[:, 0] = Y  # First outcome = existing deterioration label
                Y = Y_multi
        elif isinstance(data, pd.DataFrame):
            # Extract feature columns and label columns
            label_cols = [c for c in self.outcome_names if c in data.columns]
            feature_cols = [c for c in data.columns if c not in label_cols]
            X = data[feature_cols].values
            Y = data[label_cols].values if label_cols else np.zeros(
                (len(data), self.num_outcomes), dtype=np.float32
            )
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        X_tensor = torch.tensor(X, dtype=torch.float32)
        Y_tensor = torch.tensor(Y, dtype=torch.float32)

        # Ensure sequence dimension: (B, F) -> (B, 1, F) repeated to (B, T, F)
        if X_tensor.ndim == 2:
            X_tensor = X_tensor.unsqueeze(1).repeat(1, 24, 1)

        dataset = TensorDataset(X_tensor, Y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def _train_impl(self, train_data, val_data, **kwargs) -> dict[str, list[float]]:
        """Multi-task training loop with mixed precision."""
        epochs = kwargs.get("epochs", multi_outcome_settings.multi_epochs)
        lr = kwargs.get("lr", multi_outcome_settings.multi_learning_rate)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        scaler = torch.amp.GradScaler("cuda") if self.device.type == "cuda" else None

        train_loader = self._prepare_dataloaders(train_data, shuffle=True)
        val_loader = self._prepare_dataloaders(val_data, shuffle=False)

        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        patience_counter = 0
        patience = 10

        for epoch in range(epochs):
            # ── Training ─────────────────────────────────────────────
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            for X_batch, Y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                Y_batch = Y_batch.to(self.device)

                # Build label dict
                labels = {
                    self.outcome_names[i]: Y_batch[:, i]
                    for i in range(min(Y_batch.shape[1], self.num_outcomes))
                }

                optimizer.zero_grad()

                if scaler:
                    with torch.amp.autocast("cuda"):
                        logits = self.model(X_batch)
                        loss = self._compute_multi_task_loss(logits, labels)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = self.model(X_batch)
                    loss = self._compute_multi_task_loss(logits, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_train_loss = epoch_loss / max(n_batches, 1)
            history["train_loss"].append(avg_train_loss)

            # ── Validation ───────────────────────────────────────────
            self.model.eval()
            val_loss = 0.0
            val_batches = 0

            with torch.no_grad():
                for X_batch, Y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    Y_batch = Y_batch.to(self.device)
                    labels = {
                        self.outcome_names[i]: Y_batch[:, i]
                        for i in range(min(Y_batch.shape[1], self.num_outcomes))
                    }
                    logits = self.model(X_batch)
                    loss = self._compute_multi_task_loss(logits, labels)
                    val_loss += loss.item()
                    val_batches += 1

            avg_val_loss = val_loss / max(val_batches, 1)
            history["val_loss"].append(avg_val_loss)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(
                    "Epoch {}/{}: train_loss={:.4f}, val_loss={:.4f}, lr={:.2e}",
                    epoch + 1, epochs, avg_train_loss, avg_val_loss,
                    scheduler.get_last_lr()[0],
                )

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                self._save_checkpoint(epoch, best_val_loss)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping at epoch {}", epoch + 1)
                    self._load_best_checkpoint()
                    break

        return history

    def _predict_proba_impl(self, data) -> np.ndarray:
        """
        Predict probabilities for the PRIMARY outcome (deterioration).
        Required by BaseTrainer for compatibility with existing evaluation.

        Returns: (N,) array of P(deterioration=1)
        """
        all_probs = self.predict_multi_proba(data)
        # Return first outcome (deterioration) for BaseTrainer compatibility
        return all_probs[self.outcome_names[0]]

    def predict_multi_proba(self, data) -> dict[str, np.ndarray]:
        """
        Predict probabilities for ALL outcomes.

        Returns: dict[outcome_name, (N,) array of probabilities]
        """
        self.model.eval()

        if isinstance(data, tuple):
            X = data[0]
            if isinstance(X, pd.DataFrame):
                X = X.values
        elif isinstance(data, pd.DataFrame):
            label_cols = [c for c in self.outcome_names if c in data.columns]
            feature_cols = [c for c in data.columns if c not in label_cols]
            X = data[feature_cols].values
        elif isinstance(data, np.ndarray):
            X = data
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        X_tensor = torch.tensor(X, dtype=torch.float32)
        if X_tensor.ndim == 2:
            X_tensor = X_tensor.unsqueeze(1).repeat(1, 24, 1)
        X_tensor = X_tensor.to(self.device)

        with torch.no_grad():
            prob_dict = self.model.predict_proba(X_tensor)

        return {
            name: probs.cpu().numpy()
            for name, probs in prob_dict.items()
        }

    def evaluate_multi(
        self,
        data,
        split_name: str = "test",
    ) -> dict[str, EvaluationMetrics]:
        """
        Evaluate model on all outcomes separately.

        Returns: dict[outcome_name, EvaluationMetrics]
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before evaluation")

        # Get predictions for all outcomes
        all_probs = self.predict_multi_proba(data)

        # Extract true labels
        if isinstance(data, tuple):
            Y = data[1]
            if isinstance(Y, pd.DataFrame):
                Y = Y.values
        elif isinstance(data, pd.DataFrame):
            label_cols = [c for c in self.outcome_names if c in data.columns]
            Y = data[label_cols].values
        else:
            raise ValueError(f"Cannot extract labels from {type(data)}")

        results: dict[str, EvaluationMetrics] = {}

        for i, name in enumerate(self.outcome_names):
            if name not in all_probs:
                continue
            if i >= Y.shape[1] if Y.ndim > 1 else i > 0:
                continue

            y_true = Y[:, i] if Y.ndim > 1 else Y
            y_prob = all_probs[name]

            # Skip if no positive samples
            if y_true.sum() == 0:
                logger.warning(
                    "[{}] No positive samples for '{}' -- skipping metrics",
                    split_name, name,
                )
                continue

            metrics = EvaluationMetrics(y_true, y_prob)
            results[name] = metrics

            display = OUTCOME_REGISTRY.get(name)
            display_name = display.display_name if display else name
            logger.info(
                "[{}] {} -- AUROC={:.4f}, AUPRC={:.4f}, Sens={:.4f}, Spec={:.4f}",
                split_name, display_name,
                metrics.auroc, metrics.auprc, metrics.recall, metrics.specificity,
            )

        self.outcome_metrics = results
        return results

    def _save_checkpoint(self, epoch: int, val_loss: float) -> None:
        """Save best model checkpoint."""
        checkpoint_path = self.artifacts_dir / "multi_outcome_best.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "val_loss": val_loss,
            "hyperparams": self.hyperparams,
            "outcome_names": self.outcome_names,
        }, checkpoint_path)

    def _load_best_checkpoint(self) -> None:
        """Load best checkpoint from training."""
        checkpoint_path = self.artifacts_dir / "multi_outcome_best.pt"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            logger.info("Loaded best checkpoint from epoch {}", checkpoint["epoch"])

    def _save_impl(self, path: Path) -> None:
        """Save model with architecture hyperparams for serving."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "hyperparams": self.hyperparams,
            "outcome_names": self.outcome_names,
            "task_weights": self.task_weights,
        }, path.with_suffix(".pt"))

    def load_model(self, path: Path) -> None:
        """Load trained multi-outcome model."""
        checkpoint = torch.load(path.with_suffix(".pt"), map_location=self.device, weights_only=False)
        hp = checkpoint["hyperparams"]
        self.outcome_names = checkpoint.get("outcome_names", self.outcome_names)
        self.num_outcomes = len(self.outcome_names)

        self.model = MultiHeadPredictionModel(
            input_size=hp.get("input_size", 11),
            num_outcomes=self.num_outcomes,
            d_model=hp.get("d_model", 64),
            nhead=hp.get("nhead", 4),
            num_layers=hp.get("num_layers", 2),
            head_hidden_dim=hp.get("head_hidden_dim", 64),
            outcome_names=self.outcome_names,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()
        self.is_trained = True
        logger.info(
            "Multi-outcome model loaded: {} outcomes from {}",
            self.num_outcomes, path,
        )
