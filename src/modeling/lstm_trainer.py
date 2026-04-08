"""
G_One_Sync AI — BiLSTM + Attention Trainer
=============================================
Bidirectional LSTM with temporal attention mechanism for ICU
time-series deterioration prediction. Fully CUDA-accelerated.

Architecture:
    Input (W, F) → BiLSTM → Attention → FC → Sigmoid
    Optimized for RTX 3050 4GB VRAM.
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
from torch.utils.data import DataLoader

from config.settings import model_settings
from src.modeling.base_trainer import BaseTrainer, EvaluationMetrics
from src.modeling.data_loader import ICUSequenceDataset


# ── Model Architecture ──────────────────────────────────────────────────


class TemporalAttention(nn.Module):
    """Additive attention over LSTM timesteps."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, lstm_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_output: (batch, seq_len, hidden_size)

        Returns:
            context: (batch, hidden_size) — weighted sum
            weights: (batch, seq_len) — attention weights
        """
        scores = self.attention(lstm_output).squeeze(-1)  # (B, T)
        weights = torch.softmax(scores, dim=1)             # (B, T)
        context = torch.bmm(
            weights.unsqueeze(1), lstm_output
        ).squeeze(1)                                        # (B, H)
        return context, weights


class BiLSTMAttentionModel(nn.Module):
    """
    Bidirectional LSTM with Temporal Attention for ICU deterioration prediction.

    Architecture:
        Input → BatchNorm → BiLSTM(2 layers) → Attention → Dropout → FC → Sigmoid
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        direction_factor = 2 if bidirectional else 1

        # Input normalization
        self.input_norm = nn.BatchNorm1d(input_size)

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # Attention over time steps
        self.attention = TemporalAttention(hidden_size * direction_factor)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * direction_factor, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, input_size)

        Returns:
            logits: (batch,)
            attention_weights: (batch, seq_len)
        """
        # Normalize per-feature across the sequence
        B, T, F = x.shape
        x_norm = self.input_norm(x.reshape(-1, F)).reshape(B, T, F)

        # BiLSTM
        lstm_out, _ = self.lstm(x_norm)  # (B, T, H*2)

        # Attention
        context, attn_weights = self.attention(lstm_out)  # (B, H*2)

        # Classify
        logits = self.classifier(context).squeeze(-1)  # (B,)

        return logits, attn_weights


# ── Trainer ──────────────────────────────────────────────────────────────


class BiLSTMTrainer(BaseTrainer):
    """
    Trainer for BiLSTM + Attention model.
    Fully CUDA-accelerated with mixed precision for RTX 3050 4GB.
    """

    def __init__(
        self,
        input_size: int = 15,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        epochs: int = 50,
        patience: int = 10,
        use_mixed_precision: bool = True,
        artifacts_dir: Optional[Path] = None,
        use_mlflow: bool = True,
    ):
        super().__init__(
            model_name="bilstm",
            artifacts_dir=artifacts_dir,
            use_mlflow=use_mlflow,
        )

        self.hyperparams = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "epochs": epochs,
            "patience": patience,
            "use_mixed_precision": use_mixed_precision,
            "device": str(self.device),
        }

        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.use_amp = use_mixed_precision and self.device.type == "cuda"

    def _build_model(self, **kwargs) -> BiLSTMAttentionModel:
        """Build BiLSTM + Attention model."""
        model = BiLSTMAttentionModel(
            input_size=self.hyperparams["input_size"],
            hidden_size=self.hyperparams["hidden_size"],
            num_layers=self.hyperparams["num_layers"],
            dropout=self.hyperparams["dropout"],
        ).to(self.device)

        n_params = sum(p.numel() for p in model.parameters())
        logger.info(
            "BiLSTM model: {:.1f}K params, device={}",
            n_params / 1000, self.device,
        )

        return model

    def _train_impl(
        self,
        train_data: tuple[np.ndarray, np.ndarray],
        val_data: tuple[np.ndarray, np.ndarray],
        **kwargs,
    ) -> dict[str, list[float]]:
        """Train BiLSTM with early stopping and mixed precision."""

        # Create datasets and loaders
        train_ds = ICUSequenceDataset(train_data[0], train_data[1])
        val_ds = ICUSequenceDataset(val_data[0], val_data[1])

        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True, pin_memory=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.batch_size, shuffle=False, pin_memory=True
        )

        # Optimizer + scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hyperparams["learning_rate"],
            weight_decay=self.hyperparams["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=5, factor=0.5, verbose=True,
        )

        # Class-weighted loss
        pos_count = train_data[1].sum()
        neg_count = len(train_data[1]) - pos_count
        pos_weight = torch.tensor([neg_count / max(pos_count, 1)], device=self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Mixed precision scaler
        scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        # Training loop
        history = {
            "train_loss": [], "val_loss": [],
            "train_auroc": [], "val_auroc": [],
        }
        best_val_auroc = 0.0
        patience_counter = 0
        best_state = None

        for epoch in range(self.epochs):
            # ── Train epoch ──
            self.model.train()
            train_losses = []
            train_probs = []
            train_labels = []

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device, non_blocking=True)
                y_batch = y_batch.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                if self.use_amp:
                    with torch.amp.autocast("cuda"):
                        logits, _ = self.model(X_batch)
                        loss = criterion(logits, y_batch)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits, _ = self.model(X_batch)
                    loss = criterion(logits, y_batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()

                train_losses.append(loss.item())
                train_probs.extend(torch.sigmoid(logits).detach().cpu().numpy())
                train_labels.extend(y_batch.cpu().numpy())

            # ── Validate epoch ──
            self.model.eval()
            val_losses = []
            val_probs = []
            val_labels = []

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device, non_blocking=True)
                    y_batch = y_batch.to(self.device, non_blocking=True)

                    if self.use_amp:
                        with torch.amp.autocast("cuda"):
                            logits, _ = self.model(X_batch)
                            loss = criterion(logits, y_batch)
                    else:
                        logits, _ = self.model(X_batch)
                        loss = criterion(logits, y_batch)

                    val_losses.append(loss.item())
                    val_probs.extend(torch.sigmoid(logits).cpu().numpy())
                    val_labels.extend(y_batch.cpu().numpy())

            # Compute epoch metrics
            from sklearn.metrics import roc_auc_score
            train_auroc = roc_auc_score(train_labels, train_probs)
            val_auroc = roc_auc_score(val_labels, val_probs)

            history["train_loss"].append(np.mean(train_losses))
            history["val_loss"].append(np.mean(val_losses))
            history["train_auroc"].append(train_auroc)
            history["val_auroc"].append(val_auroc)

            scheduler.step(val_auroc)

            # Log
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(
                    "Epoch {}/{} | train_loss={:.4f} train_auroc={:.4f} | "
                    "val_loss={:.4f} val_auroc={:.4f} | LR={:.6f}",
                    epoch + 1, self.epochs,
                    np.mean(train_losses), train_auroc,
                    np.mean(val_losses), val_auroc,
                    optimizer.param_groups[0]["lr"],
                )

            # GPU memory
            if self.device.type == "cuda" and epoch == 0:
                allocated = torch.cuda.memory_allocated(self.device) / 1e9
                logger.info("GPU memory after epoch 1: {:.2f} GB / 4.0 GB", allocated)

            # Early stopping
            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                logger.info("Early stopping at epoch {} (best AUROC={:.4f})", epoch + 1, best_val_auroc)
                break

        # Restore best model
        if best_state:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})

        logger.info("Best val AUROC: {:.4f}", best_val_auroc)
        return history

    def _predict_proba_impl(self, data) -> np.ndarray:
        """Predict deterioration probabilities."""
        self.model.eval()

        if isinstance(data, tuple):
            windows = data[0] if isinstance(data[0], np.ndarray) else data[0].numpy()
        elif isinstance(data, np.ndarray):
            windows = data
        elif isinstance(data, DataLoader):
            all_probs = []
            with torch.no_grad():
                for X_batch, _ in data:
                    X_batch = X_batch.to(self.device, non_blocking=True)
                    if self.use_amp:
                        with torch.amp.autocast("cuda"):
                            logits, _ = self.model(X_batch)
                    else:
                        logits, _ = self.model(X_batch)
                    all_probs.extend(torch.sigmoid(logits).cpu().numpy())
            return np.array(all_probs)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        # Batch predict from numpy array
        ds = ICUSequenceDataset(windows, np.zeros(len(windows)))
        loader = DataLoader(ds, batch_size=self.batch_size, pin_memory=True)

        all_probs = []
        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(self.device, non_blocking=True)
                if self.use_amp:
                    with torch.amp.autocast("cuda"):
                        logits, _ = self.model(X_batch)
                else:
                    logits, _ = self.model(X_batch)
                all_probs.extend(torch.sigmoid(logits).cpu().numpy())

        return np.array(all_probs)

    def get_attention_weights(self, windows: np.ndarray) -> np.ndarray:
        """Extract attention weights for interpretability."""
        self.model.eval()
        ds = ICUSequenceDataset(windows, np.zeros(len(windows)))
        loader = DataLoader(ds, batch_size=self.batch_size, pin_memory=True)

        all_weights = []
        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(self.device, non_blocking=True)
                _, attn_w = self.model(X_batch)
                all_weights.extend(attn_w.cpu().numpy())

        return np.array(all_weights)

    def _save_impl(self, path: Path) -> None:
        """Save PyTorch model state dict."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "hyperparams": self.hyperparams,
        }, path.with_suffix(".pt"))

    def load_model(self, path: Path) -> None:
        """Load PyTorch model."""
        checkpoint = torch.load(path.with_suffix(".pt"), map_location=self.device, weights_only=False)
        self.hyperparams = checkpoint["hyperparams"]
        self.model = self._build_model()
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.is_trained = True
        logger.info("BiLSTM model loaded ← {}", path)
