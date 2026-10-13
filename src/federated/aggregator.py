"""
G_One_Sync AI — Federated Aggregation Strategies
====================================================
Implements FedAvg, FedProx, and FedBN aggregation for
combining model updates from multiple hospital sites.

Aggregation operates on serialized state dicts — the coordinator
never sees raw patient data from any site.
"""

from __future__ import annotations

import abc
import copy
import io
from collections import OrderedDict
from typing import Optional

import numpy as np
import torch
from loguru import logger

from src.federated.fl_schemas import AggregationStrategy


class BaseAggregator(abc.ABC):
    """Abstract base class for federated aggregation strategies."""

    def __init__(self, strategy: AggregationStrategy):
        self.strategy = strategy

    @abc.abstractmethod
    def aggregate(
        self,
        state_dicts: list[dict[str, torch.Tensor]],
        weights: list[float],
    ) -> dict[str, torch.Tensor]:
        """Aggregate model state dicts from multiple sites.

        Args:
            state_dicts: List of model state dicts from participating sites.
            weights: Aggregation weights per site (typically proportional
                     to dataset size). Will be normalized to sum to 1.

        Returns:
            Aggregated global model state dict.
        """
        ...

    @staticmethod
    def _normalize_weights(weights: list[float]) -> list[float]:
        """Normalize weights to sum to 1.0."""
        total = sum(weights)
        if total <= 0:
            n = len(weights)
            return [1.0 / n] * n
        return [w / total for w in weights]


# ── FedAvg ───────────────────────────────────────────────────────────────


class FedAvgAggregator(BaseAggregator):
    """
    Federated Averaging (McMahan et al., 2017).

    Computes the weighted average of model parameters across sites:
        θ_global = Σ (nᵢ / N) × θᵢ

    where nᵢ = dataset size at site i, N = total across all sites.
    """

    def __init__(self):
        super().__init__(AggregationStrategy.FEDAVG)

    def aggregate(
        self,
        state_dicts: list[dict[str, torch.Tensor]],
        weights: list[float],
    ) -> dict[str, torch.Tensor]:
        if not state_dicts:
            raise ValueError("No state dicts to aggregate")

        if len(state_dicts) != len(weights):
            raise ValueError(
                f"Mismatch: {len(state_dicts)} state dicts vs {len(weights)} weights"
            )

        norm_weights = self._normalize_weights(weights)

        logger.info(
            "FedAvg aggregation: {} sites, weights={}",
            len(state_dicts),
            [f"{w:.3f}" for w in norm_weights],
        )

        # Start with zeros matching the first state dict's structure
        aggregated = OrderedDict()
        reference = state_dicts[0]

        for key in reference:
            aggregated[key] = torch.zeros_like(reference[key], dtype=torch.float32)

        # Weighted sum
        for state_dict, weight in zip(state_dicts, norm_weights):
            for key in aggregated:
                if key in state_dict:
                    aggregated[key] += weight * state_dict[key].float()

        logger.info(
            "FedAvg aggregation complete: {} parameters",
            len(aggregated),
        )

        return aggregated


# ── FedProx ──────────────────────────────────────────────────────────────


class FedProxAggregator(BaseAggregator):
    """
    Federated Proximal (Li et al., 2020).

    Same aggregation as FedAvg, but the local training objective at each
    site includes a proximal term:
        L_local = L_task + (μ/2) × ||θ_local - θ_global||²

    This regularizes local updates to stay close to the global model,
    which helps with non-IID data across hospitals. The proximal term
    is applied during local training (in participant.py), NOT during
    aggregation — but the aggregator tracks the μ parameter.
    """

    def __init__(self, mu: float = 0.01):
        super().__init__(AggregationStrategy.FEDPROX)
        self.mu = mu

    def aggregate(
        self,
        state_dicts: list[dict[str, torch.Tensor]],
        weights: list[float],
    ) -> dict[str, torch.Tensor]:
        """FedProx aggregation is the same as FedAvg — the proximal
        term only affects local training, not the global aggregation step.
        """
        if not state_dicts:
            raise ValueError("No state dicts to aggregate")

        if len(state_dicts) != len(weights):
            raise ValueError(
                f"Mismatch: {len(state_dicts)} state dicts vs {len(weights)} weights"
            )

        norm_weights = self._normalize_weights(weights)

        logger.info(
            "FedProx aggregation (μ={:.4f}): {} sites",
            self.mu,
            len(state_dicts),
        )

        aggregated = OrderedDict()
        reference = state_dicts[0]

        for key in reference:
            aggregated[key] = torch.zeros_like(reference[key], dtype=torch.float32)

        for state_dict, weight in zip(state_dicts, norm_weights):
            for key in aggregated:
                if key in state_dict:
                    aggregated[key] += weight * state_dict[key].float()

        logger.info("FedProx aggregation complete: {} parameters", len(aggregated))

        return aggregated


# ── FedBN ────────────────────────────────────────────────────────────────


class FedBNAggregator(BaseAggregator):
    """
    Federated Batch Normalization (Li et al., 2021).

    Aggregates all parameters EXCEPT BatchNorm layers. Each site retains
    its own BatchNorm statistics (running_mean, running_var, weight, bias)
    to account for site-specific data distributions.

    This is critical in clinical settings where:
    - Hospital A may see older cardiac patients
    - Hospital B may see younger trauma patients
    - Feature distributions differ naturally between sites

    BatchNorm keys are identified by patterns: 'bn', 'norm', 'running_mean',
    'running_var', 'num_batches_tracked'.
    """

    # Patterns indicating a BatchNorm parameter
    BN_PATTERNS = ("bn", "norm", "running_mean", "running_var", "num_batches_tracked")

    def __init__(self):
        super().__init__(AggregationStrategy.FEDBN)

    def _is_bn_key(self, key: str) -> bool:
        """Check if a state dict key belongs to a BatchNorm layer."""
        key_lower = key.lower()
        return any(pattern in key_lower for pattern in self.BN_PATTERNS)

    def aggregate(
        self,
        state_dicts: list[dict[str, torch.Tensor]],
        weights: list[float],
    ) -> dict[str, torch.Tensor]:
        if not state_dicts:
            raise ValueError("No state dicts to aggregate")

        if len(state_dicts) != len(weights):
            raise ValueError(
                f"Mismatch: {len(state_dicts)} state dicts vs {len(weights)} weights"
            )

        norm_weights = self._normalize_weights(weights)
        reference = state_dicts[0]

        # Separate BN and non-BN keys
        bn_keys = [k for k in reference if self._is_bn_key(k)]
        agg_keys = [k for k in reference if not self._is_bn_key(k)]

        logger.info(
            "FedBN aggregation: {} sites | {} shared params, {} BN params (site-local)",
            len(state_dicts),
            len(agg_keys),
            len(bn_keys),
        )

        # Aggregate only non-BN parameters (FedAvg on shared params)
        aggregated = OrderedDict()
        for key in agg_keys:
            aggregated[key] = torch.zeros_like(reference[key], dtype=torch.float32)

        for state_dict, weight in zip(state_dicts, norm_weights):
            for key in agg_keys:
                if key in state_dict:
                    aggregated[key] += weight * state_dict[key].float()

        # Keep BN params from first site as default (sites override locally)
        for key in bn_keys:
            aggregated[key] = reference[key].clone()

        logger.info("FedBN aggregation complete: {} parameters", len(aggregated))

        return aggregated


# ── Factory ──────────────────────────────────────────────────────────────


def create_aggregator(
    strategy: AggregationStrategy | str,
    **kwargs,
) -> BaseAggregator:
    """Factory function to create an aggregator by strategy name.

    Args:
        strategy: Aggregation strategy name or enum value.
        **kwargs: Extra args passed to the aggregator constructor.

    Returns:
        An instantiated aggregator.
    """
    if isinstance(strategy, str):
        strategy = AggregationStrategy(strategy.lower())

    if strategy == AggregationStrategy.FEDAVG:
        return FedAvgAggregator()
    elif strategy == AggregationStrategy.FEDPROX:
        mu = kwargs.get("mu", 0.01)
        return FedProxAggregator(mu=mu)
    elif strategy == AggregationStrategy.FEDBN:
        return FedBNAggregator()
    else:
        raise ValueError(f"Unknown aggregation strategy: {strategy}")


# ── Utility: State Dict Serialization ────────────────────────────────────


def serialize_state_dict(state_dict: dict[str, torch.Tensor]) -> bytes:
    """Serialize a PyTorch state dict to bytes for network transfer."""
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    return buffer.getvalue()


def deserialize_state_dict(data: bytes) -> dict[str, torch.Tensor]:
    """Deserialize bytes back to a PyTorch state dict."""
    buffer = io.BytesIO(data)
    return torch.load(buffer, map_location="cpu", weights_only=True)


def compute_aggregation_weights(
    dataset_sizes: list[int],
) -> list[float]:
    """Compute aggregation weights proportional to dataset sizes.

    Args:
        dataset_sizes: Number of training samples at each site.

    Returns:
        Normalized weights summing to 1.0.
    """
    total = sum(dataset_sizes)
    if total <= 0:
        n = len(dataset_sizes)
        return [1.0 / n] * n
    return [s / total for s in dataset_sizes]
