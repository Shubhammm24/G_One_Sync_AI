"""
G_One_Sync AI — Multi-Head Prediction Model
===============================================
Shared Transformer encoder with per-outcome classification heads.

Architecture:
    Input (B, T, F) --> Linear Proj --> PosEncoding --> TransformerEncoder
                     --> Shared Representation (B, D*2)
                     --> Head_1(deterioration) --> P(deterioration)
                     --> Head_2(sepsis)        --> P(sepsis)
                     --> Head_3(cardiac)       --> P(cardiac_arrest)
                     --> Head_4(resp_failure)  --> P(resp_failure)

Design Decision:
    The shared encoder is architecturally identical to
    TemporalTransformerModel from transformer_trainer.py but is a
    NEW class (not a modification). This avoids any risk to the
    existing binary prediction pipeline.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from loguru import logger

from src.modeling.multi_outcome.label_definitions import NUM_OUTCOMES, OUTCOME_COLUMNS

# ── Positional Encoding (identical to transformer_trainer.py) ────────────


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
            pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ── Classification Head ──────────────────────────────────────────────────


class ClassificationHead(nn.Module):
    """Per-outcome binary classification head (2-layer MLP)."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) — shared encoder representation
        Returns:
            logits: (B,) — raw logits for this outcome
        """
        return self.head(x).squeeze(-1)


# ── Multi-Head Model ─────────────────────────────────────────────────────


class MultiHeadPredictionModel(nn.Module):
    """
    Shared Transformer encoder with multiple independent classification heads.

    Each head predicts a different clinical outcome. The shared encoder
    learns cross-outcome representations while heads specialize.
    """

    def __init__(
        self,
        input_size: int,
        num_outcomes: int = NUM_OUTCOMES,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        head_hidden_dim: int = 64,
        dropout: float = 0.2,
        head_dropout: float = 0.3,
        outcome_names: Optional[list[str]] = None,
    ):
        super().__init__()

        self.input_size = input_size
        self.num_outcomes = num_outcomes
        self.d_model = d_model
        self.outcome_names = outcome_names or OUTCOME_COLUMNS[:num_outcomes]

        # ── Shared encoder ───────────────────────────────────────────
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
        )

        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

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

        # Shared representation dimension: mean_pool + last_token = d_model * 2
        shared_dim = d_model * 2

        # ── Per-outcome classification heads ─────────────────────────
        self.heads = nn.ModuleDict({
            name: ClassificationHead(shared_dim, head_hidden_dim, head_dropout)
            for name in self.outcome_names
        })

        # Log architecture
        total_params = sum(p.numel() for p in self.parameters())
        encoder_params = sum(
            p.numel() for n, p in self.named_parameters() if "heads" not in n
        )
        head_params = total_params - encoder_params
        logger.info(
            "MultiHeadModel: {} outcomes, {} params (encoder={}, heads={})",
            num_outcomes, total_params, encoder_params, head_params,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shared encoder forward pass.

        Args:
            x: (B, T, input_size) — temporal input sequence
        Returns:
            representation: (B, d_model*2) — pooled encoder output
        """
        x = self.input_proj(x)       # (B, T, d_model)
        x = self.pos_encoder(x)
        encoded = self.transformer(x)  # (B, T, d_model)

        # Pooling: mean + last token (same strategy as TemporalTransformerModel)
        mean_pool = encoded.mean(dim=1)    # (B, d_model)
        last_token = encoded[:, -1, :]      # (B, d_model)
        pooled = torch.cat([mean_pool, last_token], dim=1)  # (B, d_model*2)

        return pooled

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Full forward pass: encode + all heads.

        Args:
            x: (B, T, input_size)
        Returns:
            logits: dict[outcome_name, (B,)] — raw logits per outcome
        """
        representation = self.encode(x)

        logits = {}
        for name, head in self.heads.items():
            logits[name] = head(representation)

        return logits

    def predict_proba(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Predict calibrated probabilities for all outcomes.

        Args:
            x: (B, T, input_size)
        Returns:
            probabilities: dict[outcome_name, (B,)] — sigmoid probabilities
        """
        logits = self.forward(x)
        return {name: torch.sigmoid(logit) for name, logit in logits.items()}

    def get_outcome_logits_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return logits as a stacked tensor for loss computation.

        Args:
            x: (B, T, input_size)
        Returns:
            logits: (B, num_outcomes) — ordered by self.outcome_names
        """
        logit_dict = self.forward(x)
        return torch.stack(
            [logit_dict[name] for name in self.outcome_names], dim=1
        )
