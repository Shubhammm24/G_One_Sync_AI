"""
G_One_Sync AI — Temporal Transformer Trainer
===============================================
Transformer encoder for ICU time-series deterioration prediction.
Uses positional encoding + multi-head self-attention over temporal windows.

Architecture:
    Input (W, F) → Linear Proj → PosEncoding → TransformerEncoder → Pool → FC → Sigmoid
    Optimized for RTX 3050 4GB VRAM with mixed precision.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader

from config.settings import model_settings
from src.modeling.base_trainer import BaseTrainer
from src.modeling.data_loader import ICUSequenceDataset


# ── Model Architecture ──────────────────────────────────────────────────


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences."""

    def __init__(self, d_model: int, max_len: int = 200, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TemporalTransformerModel(nn.Module):
    """
    Transformer Encoder for ICU time-series classification.

    Architecture:
        Input → Linear Projection → Positional Encoding
        → TransformerEncoder (N layers, H heads)
        → Temporal Pooling (mean + last) → FC → Sigmoid
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
        )

        # Classification head (mean pool + last token + attention-weighted)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(d_model, 1),
        )

        # Learnable query for attention pooling
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            logits: (batch,)
        """
        B = x.size(0)

        # Project input features to d_model
        x = self.input_proj(x)  # (B, T, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encode
        encoded = self.transformer(x)  # (B, T, d_model)

        # Pooling: concatenate mean and last timestep
        mean_pool = encoded.mean(dim=1)    # (B, d_model)
        last_token = encoded[:, -1, :]      # (B, d_model)
        pooled = torch.cat([mean_pool, last_token], dim=1)  # (B, d_model*2)

        # Classify
        logits = self.classifier(pooled).squeeze(-1)  # (B,)

        return logits


# ── Trainer ──────────────────────────────────────────────────────────────


class TransformerTrainer(BaseTrainer):
    """
    Trainer for Temporal Transformer model.
    CUDA-accelerated with mixed precision for RTX 3050 4GB.
    """

    def __init__(
        self,
        input_size: int = 15,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.2,
        learning_rate: float = 5e-4,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        epochs: int = 50,
        patience: int = 10,
        warmup_epochs: int = 5,
        use_mixed_precision: bool = True,
        loss_type: str = "focal",
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        artifacts_dir: Optional[Path] = None,
        use_mlflow: bool = True,
    ):
        super().__init__(
            model_name="transformer",
            artifacts_dir=artifacts_dir,
            use_mlflow=use_mlflow,
        )

        self.hyperparams = {
            "input_size": input_size,
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "dim_feedforward": dim_feedforward,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "epochs": epochs,
            "patience": patience,
            "warmup_epochs": warmup_epochs,
            "loss_type": loss_type,
            "focal_alpha": focal_alpha,
            "focal_gamma": focal_gamma,
            "device": str(self.device),
        }

        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.warmup_epochs = warmup_epochs
        self.use_amp = use_mixed_precision and self.device.type == "cuda"
        self.loss_type = loss_type
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def _build_model(self, **kwargs) -> TemporalTransformerModel:
        """Build Temporal Transformer model."""
        model = TemporalTransformerModel(
            input_size=self.hyperparams["input_size"],
            d_model=self.hyperparams["d_model"],
            nhead=self.hyperparams["nhead"],
            num_layers=self.hyperparams["num_layers"],
            dim_feedforward=self.hyperparams["dim_feedforward"],
            dropout=self.hyperparams["dropout"],
        ).to(self.device)

        n_params = sum(p.numel() for p in model.parameters())
        logger.info(
            "Transformer model: {:.1f}K params, d_model={}, heads={}, layers={}",
            n_params / 1000, self.hyperparams["d_model"],
            self.hyperparams["nhead"], self.hyperparams["num_layers"],
        )

        return model

    def _get_lr_scheduler(self, optimizer, total_steps: int):
        """Cosine annealing with linear warmup."""
        warmup_steps = self.warmup_epochs * (total_steps // self.epochs)

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def _train_impl(
        self,
        train_data: tuple[np.ndarray, np.ndarray],
        val_data: tuple[np.ndarray, np.ndarray],
        **kwargs,
    ) -> dict[str, list[float]]:
        """Train Transformer with warmup, cosine decay, and early stopping."""

        train_ds = ICUSequenceDataset(train_data[0], train_data[1])
        val_ds = ICUSequenceDataset(val_data[0], val_data[1])

        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True, pin_memory=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.batch_size, shuffle=False, pin_memory=True
        )

        # Optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hyperparams["learning_rate"],
            weight_decay=self.hyperparams["weight_decay"],
            betas=(0.9, 0.98),
        )

        total_steps = len(train_loader) * self.epochs
        scheduler = self._get_lr_scheduler(optimizer, total_steps)

        # Loss function — Focal Loss or weighted BCE
        pos_count = train_data[1].sum()
        neg_count = len(train_data[1]) - pos_count
        pos_weight_val = float(neg_count / max(pos_count, 1))

        if self.loss_type == "focal":
            from src.modeling.losses import FocalLoss
            criterion = FocalLoss(
                alpha=self.focal_alpha,
                gamma=self.focal_gamma,
                pos_weight=pos_weight_val,
            )
            logger.info("Using Focal Loss (alpha={}, gamma={}, pos_weight={:.1f})",
                        self.focal_alpha, self.focal_gamma, pos_weight_val)
        else:
            pos_weight = torch.tensor([pos_weight_val], device=self.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        history = {"train_loss": [], "val_loss": [], "train_auroc": [], "val_auroc": []}
        best_val_auroc = 0.0
        patience_counter = 0
        best_state = None

        for epoch in range(self.epochs):
            # ── Train ──
            self.model.train()
            train_losses, train_probs, train_labels = [], [], []

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device, non_blocking=True)
                y_batch = y_batch.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                if self.use_amp:
                    with torch.amp.autocast("cuda"):
                        logits = self.model(X_batch)
                        loss = criterion(logits, y_batch)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = self.model(X_batch)
                    loss = criterion(logits, y_batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()

                scheduler.step()
                train_losses.append(loss.item())
                train_probs.extend(torch.sigmoid(logits).detach().cpu().numpy())
                train_labels.extend(y_batch.cpu().numpy())

            # ── Validate ──
            self.model.eval()
            val_losses, val_probs, val_labels = [], [], []

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device, non_blocking=True)
                    y_batch = y_batch.to(self.device, non_blocking=True)

                    if self.use_amp:
                        with torch.amp.autocast("cuda"):
                            logits = self.model(X_batch)
                            loss = criterion(logits, y_batch)
                    else:
                        logits = self.model(X_batch)
                        loss = criterion(logits, y_batch)

                    val_losses.append(loss.item())
                    val_probs.extend(torch.sigmoid(logits).cpu().numpy())
                    val_labels.extend(y_batch.cpu().numpy())

            from sklearn.metrics import roc_auc_score
            train_auroc = roc_auc_score(train_labels, train_probs)
            val_auroc = roc_auc_score(val_labels, val_probs)

            history["train_loss"].append(np.mean(train_losses))
            history["val_loss"].append(np.mean(val_losses))
            history["train_auroc"].append(train_auroc)
            history["val_auroc"].append(val_auroc)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(
                    "Epoch {}/{} | loss={:.4f}/{:.4f} | auroc={:.4f}/{:.4f} | lr={:.6f}",
                    epoch + 1, self.epochs,
                    np.mean(train_losses), np.mean(val_losses),
                    train_auroc, val_auroc,
                    optimizer.param_groups[0]["lr"],
                )

            if self.device.type == "cuda" and epoch == 0:
                allocated = torch.cuda.memory_allocated(self.device) / 1e9
                logger.info("GPU memory after epoch 1: {:.2f} GB / 4.0 GB", allocated)

            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                logger.info("Early stopping at epoch {} (best AUROC={:.4f})", epoch + 1, best_val_auroc)
                break

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
                            logits = self.model(X_batch)
                    else:
                        logits = self.model(X_batch)
                    all_probs.extend(torch.sigmoid(logits).cpu().numpy())
            return np.array(all_probs)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        ds = ICUSequenceDataset(windows, np.zeros(len(windows)))
        loader = DataLoader(ds, batch_size=self.batch_size, pin_memory=True)

        all_probs = []
        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(self.device, non_blocking=True)
                if self.use_amp:
                    with torch.amp.autocast("cuda"):
                        logits = self.model(X_batch)
                else:
                    logits = self.model(X_batch)
                all_probs.extend(torch.sigmoid(logits).cpu().numpy())

        return np.array(all_probs)

    def _save_impl(self, path: Path) -> None:
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "hyperparams": self.hyperparams,
        }, path.with_suffix(".pt"))

    def load_model(self, path: Path) -> None:
        checkpoint = torch.load(path.with_suffix(".pt"), map_location=self.device, weights_only=False)
        self.hyperparams = checkpoint["hyperparams"]
        self.model = self._build_model()
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.is_trained = True
        logger.info("Transformer model loaded ← {}", path)
