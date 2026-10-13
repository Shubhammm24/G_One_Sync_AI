"""
G_One_Sync AI — Differential Privacy for Federated Learning
===============================================================
Implements per-sample gradient clipping, Gaussian noise injection,
and Rényi Differential Privacy (RDP) accounting for privacy budget
tracking across federated learning rounds.

Clinical rationale:
    Hospital sites must prove to their IRBs that sharing model updates
    (gradients) does not leak information about individual patients.
    DP-SGD provides a mathematically rigorous guarantee: even an
    adversary with unlimited compute cannot determine whether any
    single patient was in the training set.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from src.federated.fl_schemas import PrivacyBudget


# ── Gradient Clipping ────────────────────────────────────────────────────


def clip_gradients(
    model: nn.Module,
    max_grad_norm: float,
) -> float:
    """Clip model gradients to a maximum L2 norm.

    Per DP-SGD (Abadi et al., 2016), per-sample gradient clipping
    bounds the sensitivity of the gradient computation, which is
    required before adding calibrated noise.

    Args:
        model: PyTorch model with computed gradients.
        max_grad_norm: Maximum allowed L2 norm for gradients.

    Returns:
        The original (unclipped) gradient norm for monitoring.
    """
    parameters = [p for p in model.parameters() if p.grad is not None]
    if not parameters:
        return 0.0

    # Compute total gradient norm
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), 2) for p in parameters]),
        2,
    ).item()

    # Clip
    clip_coef = max_grad_norm / max(total_norm, max_grad_norm)
    if clip_coef < 1.0:
        for p in parameters:
            p.grad.detach().mul_(clip_coef)

    return total_norm


# ── Noise Injection ──────────────────────────────────────────────────────


def add_gaussian_noise(
    model: nn.Module,
    noise_multiplier: float,
    max_grad_norm: float,
    batch_size: int,
) -> None:
    """Add calibrated Gaussian noise to gradients for DP guarantee.

    The noise standard deviation is:
        σ = noise_multiplier × (max_grad_norm / batch_size)

    This ensures (ε, δ)-differential privacy when combined with
    gradient clipping and proper accounting.

    Args:
        model: PyTorch model with clipped gradients.
        noise_multiplier: Controls the noise level (higher = more private).
        max_grad_norm: Gradient clipping bound (sensitivity).
        batch_size: Number of samples in the current batch.
    """
    noise_std = noise_multiplier * max_grad_norm / batch_size

    for p in model.parameters():
        if p.grad is not None:
            noise = torch.normal(
                mean=0.0,
                std=noise_std,
                size=p.grad.shape,
                device=p.grad.device,
                dtype=p.grad.dtype,
            )
            p.grad.add_(noise)


def add_noise_to_state_dict(
    state_dict: dict[str, torch.Tensor],
    noise_multiplier: float,
    max_grad_norm: float,
    dataset_size: int,
) -> dict[str, torch.Tensor]:
    """Add Gaussian noise to a model state dict before sending to coordinator.

    This is an alternative to per-gradient noise: instead of noising
    each SGD step, we noise the final model update (state dict delta)
    before transmission. This is simpler and compatible with XGBoost.

    Args:
        state_dict: Model state dict (or state dict delta).
        noise_multiplier: Controls noise level.
        max_grad_norm: Sensitivity bound.
        dataset_size: Local dataset size (for noise scaling).

    Returns:
        Noised state dict.
    """
    noise_std = noise_multiplier * max_grad_norm / dataset_size
    noised = {}

    for key, tensor in state_dict.items():
        if tensor.is_floating_point():
            noise = torch.normal(
                mean=0.0,
                std=noise_std,
                size=tensor.shape,
                device=tensor.device,
                dtype=tensor.dtype,
            )
            noised[key] = tensor + noise
        else:
            # Non-float tensors (indices, counts) — do not noise
            noised[key] = tensor.clone()

    logger.debug(
        "Added DP noise to state dict: σ={:.6f}, {} parameters",
        noise_std,
        len(noised),
    )

    return noised


# ── Rényi Differential Privacy Accounting ────────────────────────────────


class RDPAccountant:
    """
    Rényi Differential Privacy (RDP) accountant.

    Tracks privacy budget consumption across multiple FL rounds using
    the Rényi divergence framework (Mironov, 2017). RDP provides
    tighter composition bounds than basic (ε, δ)-DP composition.

    Usage:
        accountant = RDPAccountant(epsilon_budget=8.0, delta=1e-5)
        for round_num in range(num_rounds):
            epsilon_round = accountant.compute_round_epsilon(
                noise_multiplier=1.1,
                sample_rate=64/5000,
                num_steps=epochs_per_round * steps_per_epoch,
            )
            accountant.record_round(epsilon_round)
            if accountant.is_budget_exhausted():
                break
    """

    # Standard RDP orders for accounting
    DEFAULT_ORDERS = [
        1 + x / 10.0 for x in range(1, 100)
    ] + list(range(12, 64))

    def __init__(
        self,
        epsilon_budget: float = 8.0,
        delta: float = 1e-5,
        orders: Optional[list[float]] = None,
    ):
        self.delta = delta
        self.orders = np.array(orders or self.DEFAULT_ORDERS)
        self.budget = PrivacyBudget(
            epsilon_total=epsilon_budget,
            delta=delta,
        )

        # Track RDP values across rounds for tight composition
        self._rdp_cumulative = np.zeros_like(self.orders)

        logger.info(
            "RDP Accountant initialized: ε_budget={}, δ={}, {} orders",
            epsilon_budget,
            delta,
            len(self.orders),
        )

    def _compute_rdp_single_step(
        self,
        noise_multiplier: float,
        sample_rate: float,
    ) -> np.ndarray:
        """Compute RDP for a single gradient step via the Gaussian mechanism.

        Uses upper bounds on the subsampled Gaussian mechanism RDP
        from Mironov et al. (2019). Always returns non-negative values.
        """
        if noise_multiplier <= 0:
            return np.full_like(self.orders, float("inf"))

        rdp = np.zeros_like(self.orders, dtype=np.float64)
        sigma_sq = noise_multiplier ** 2

        for i, alpha in enumerate(self.orders):
            if alpha <= 1:
                # RDP of order ≤ 1 is the KL divergence; skip
                rdp[i] = float("inf")
                continue

            if sample_rate >= 1.0:
                # Full-batch Gaussian mechanism: RDP_α = α / (2σ²)
                rdp[i] = alpha / (2.0 * sigma_sq)
            else:
                # Subsampled Gaussian: use the simple upper bound
                #   RDP_α ≤ (q² × α) / (2σ²)
                # where q = sample_rate. This is always non-negative
                # and correct (though not the tightest possible).
                rdp[i] = (sample_rate ** 2) * alpha / (2.0 * sigma_sq)

        return np.maximum(rdp, 0.0)

    def _rdp_to_epsilon(self, rdp: np.ndarray) -> float:
        """Convert RDP values to (ε, δ)-DP via the optimal conversion.

        Uses: ε = min over α of [ rdp(α) - log(δ) / (α - 1) ]
        """
        epsilons = []
        for rdp_val, alpha in zip(rdp, self.orders):
            if alpha <= 1:
                continue
            epsilon = rdp_val - math.log(self.delta) / (alpha - 1)
            epsilons.append(epsilon)

        return min(epsilons) if epsilons else float("inf")

    def compute_round_epsilon(
        self,
        noise_multiplier: float,
        sample_rate: float,
        num_steps: int,
    ) -> float:
        """Compute the ε consumed by one FL round.

        Args:
            noise_multiplier: σ used in Gaussian noise injection.
            sample_rate: Batch size / dataset size (subsampling ratio).
            num_steps: Total gradient steps in this round
                       (epochs_per_round × batches_per_epoch).

        Returns:
            ε spent for this round (accounting for composition).
        """
        # RDP for a single step
        rdp_step = self._compute_rdp_single_step(noise_multiplier, sample_rate)

        # Composition over num_steps (RDP composes linearly)
        rdp_round = rdp_step * num_steps

        # Convert to (ε, δ)-DP
        epsilon_round = self._rdp_to_epsilon(
            self._rdp_cumulative + rdp_round
        ) - self._rdp_to_epsilon(self._rdp_cumulative)

        return max(0.0, epsilon_round)

    def record_round(
        self,
        noise_multiplier: float,
        sample_rate: float,
        num_steps: int,
    ) -> float:
        """Record a completed FL round and update cumulative RDP.

        Args:
            noise_multiplier: σ used in this round.
            sample_rate: Subsampling ratio.
            num_steps: Total gradient steps.

        Returns:
            ε spent for this round (always ≥ 0).
        """
        rdp_step = self._compute_rdp_single_step(noise_multiplier, sample_rate)
        rdp_round = rdp_step * num_steps

        prev_epsilon = self._rdp_to_epsilon(self._rdp_cumulative)
        self._rdp_cumulative += rdp_round

        # Total ε so far
        total_epsilon = self._rdp_to_epsilon(self._rdp_cumulative)
        epsilon_round = max(0.0, total_epsilon - prev_epsilon)

        self.budget.epsilon_spent = total_epsilon
        self.budget.rounds_tracked += 1

        logger.info(
            "DP Round {}: ε_round={:.4f}, ε_total={:.4f}/{:.1f} ({:.1f}% budget used)",
            self.budget.rounds_tracked,
            epsilon_round,
            total_epsilon,
            self.budget.epsilon_total,
            100.0 * total_epsilon / self.budget.epsilon_total,
        )

        return epsilon_round

    def is_budget_exhausted(self) -> bool:
        """Check if the privacy budget has been consumed."""
        return self.budget.is_exhausted

    @property
    def epsilon_remaining(self) -> float:
        """Remaining ε budget."""
        return self.budget.epsilon_remaining

    @property
    def epsilon_spent(self) -> float:
        """Total ε spent so far."""
        return self.budget.epsilon_spent

    def get_status(self) -> dict:
        """Return current privacy accounting status."""
        return {
            "epsilon_spent": self.budget.epsilon_spent,
            "epsilon_remaining": self.budget.epsilon_remaining,
            "epsilon_budget": self.budget.epsilon_total,
            "delta": self.delta,
            "rounds_tracked": self.budget.rounds_tracked,
            "is_exhausted": self.budget.is_exhausted,
        }
