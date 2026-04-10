"""
G_One_Sync AI — Custom Loss Functions
========================================
Focal Loss for handling class imbalance in ICU deterioration prediction.
Reduces the loss contribution from easy-to-classify (stable) patients,
focusing training on hard-to-classify (borderline) cases.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification with extreme class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Where:
        - gamma > 0 reduces loss for well-classified examples ("easy negatives")
        - alpha balances positive vs negative class contribution
        - gamma=2 is the sweet spot for most imbalanced problems

    For ICU deterioration (5-6% positive rate), this dramatically
    reduces false alarm pressure compared to weighted BCE.

    Reference: Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017)
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        pos_weight: float | None = None,
        reduction: str = "mean",
    ):
        """
        Args:
            alpha: Weighting factor for the positive class [0, 1].
                   Set higher to emphasize recall, lower to emphasize precision.
            gamma: Focusing parameter. Higher = more focus on hard examples.
                   gamma=0 → standard BCE. gamma=2 → standard focal loss.
            pos_weight: Optional override for positive class weight.
            reduction: 'mean', 'sum', or 'none'.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits: Raw model output (before sigmoid), shape (N,)
            targets: Binary labels, shape (N,)

        Returns:
            Focal loss scalar
        """
        # Compute base BCE (without reduction)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        # Compute p_t (probability of correct class)
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Focal modulating factor: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha weighting: alpha for positives, (1-alpha) for negatives
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Optional: additional positive class weight
        if self.pos_weight is not None:
            class_weight = torch.where(
                targets == 1,
                torch.tensor(self.pos_weight, device=logits.device),
                torch.tensor(1.0, device=logits.device),
            )
            focal_weight = focal_weight * class_weight

        # Combined focal loss
        loss = alpha_t * focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class LabelSmoothingBCE(nn.Module):
    """
    BCE with label smoothing to prevent overconfident predictions.
    Smooths labels: 0 → epsilon, 1 → 1-epsilon.
    Helps with calibration.
    """

    def __init__(self, smoothing: float = 0.05, pos_weight: float | None = None):
        super().__init__()
        self.smoothing = smoothing
        self.pos_weight = (
            torch.tensor([pos_weight]) if pos_weight else None
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        smooth_targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(
            logits, smooth_targets,
            pos_weight=self.pos_weight.to(logits.device) if self.pos_weight is not None else None,
        )
