"""
G_One_Sync AI — Federated Learning Schemas
=============================================
Pydantic models for all federated learning messages,
round configurations, model updates, and status tracking.
"""

from __future__ import annotations

import time
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────────────────


class FLRole(str, Enum):
    """Role of a node in the federated learning network."""

    COORDINATOR = "coordinator"
    PARTICIPANT = "participant"
    STANDALONE = "standalone"


class AggregationStrategy(str, Enum):
    """Supported federated aggregation strategies."""

    FEDAVG = "fedavg"
    FEDPROX = "fedprox"
    FEDBN = "fedbn"


class RoundStatus(str, Enum):
    """Status of a federated learning round."""

    WAITING = "waiting"
    TRAINING = "training"
    AGGREGATING = "aggregating"
    COMPLETED = "completed"
    FAILED = "failed"


class ParticipantStatus(str, Enum):
    """Status of a federated learning participant."""

    REGISTERED = "registered"
    TRAINING = "training"
    SUBMITTED = "submitted"
    IDLE = "idle"
    DISCONNECTED = "disconnected"


# ── Round Configuration ──────────────────────────────────────────────────


class FLRoundConfig(BaseModel):
    """Configuration for a single federated learning round."""

    round_number: int = Field(ge=0, description="Current FL round index")
    total_rounds: int = Field(ge=1, description="Total planned rounds")
    local_epochs: int = Field(default=3, ge=1, description="Local training epochs per round")
    learning_rate: float = Field(default=1e-3, gt=0, description="Learning rate for local training")
    batch_size: int = Field(default=64, ge=1, description="Local batch size")
    aggregation_strategy: AggregationStrategy = AggregationStrategy.FEDAVG
    fedprox_mu: float = Field(default=0.01, ge=0, description="FedProx proximal term weight")
    dp_enabled: bool = Field(default=True, description="Enable differential privacy")
    dp_max_grad_norm: float = Field(default=1.0, gt=0, description="DP gradient clipping norm")
    dp_noise_multiplier: float = Field(default=1.1, gt=0, description="DP noise multiplier")


# ── Participant Registration ─────────────────────────────────────────────


class SiteRegistration(BaseModel):
    """Registration message from a participant site."""

    site_id: str = Field(description="Unique identifier for the hospital site")
    site_name: str = Field(default="", description="Human-readable site name")
    dataset_size: int = Field(ge=0, description="Number of training samples at this site")
    model_types: list[str] = Field(
        default=["bilstm", "transformer"],
        description="Model architectures available at this site",
    )
    gpu_available: bool = Field(default=False, description="Whether GPU is available")
    timestamp: float = Field(default_factory=time.time)


class SiteInfo(BaseModel):
    """Coordinator-side tracking of a registered site."""

    site_id: str
    site_name: str
    dataset_size: int
    model_types: list[str]
    gpu_available: bool
    status: ParticipantStatus = ParticipantStatus.REGISTERED
    last_heartbeat: float = Field(default_factory=time.time)
    rounds_completed: int = 0
    total_training_time_s: float = 0.0


# ── Model Updates ────────────────────────────────────────────────────────


class ModelUpdate(BaseModel):
    """Serialized model state from a participant site."""

    site_id: str = Field(description="Origin site identifier")
    round_number: int = Field(ge=0, description="FL round this update belongs to")
    model_type: str = Field(description="Model architecture (bilstm, transformer, xgboost)")
    state_dict_bytes: bytes = Field(description="Serialized model state dict")
    dataset_size: int = Field(ge=1, description="Number of samples used for training")
    training_time_s: float = Field(ge=0, description="Local training time in seconds")
    local_metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Local evaluation metrics (auroc, auprc, etc.)",
    )
    dp_epsilon_spent: float = Field(default=0.0, ge=0, description="Privacy budget spent this round")
    timestamp: float = Field(default_factory=time.time)

    model_config = {"arbitrary_types_allowed": True}


class AggregatedResult(BaseModel):
    """Result of federated aggregation for a round."""

    round_number: int = Field(ge=0)
    aggregation_strategy: AggregationStrategy
    model_type: str
    global_state_bytes: bytes = Field(description="Serialized aggregated model state")
    participating_sites: list[str] = Field(description="Sites that contributed to this round")
    site_weights: dict[str, float] = Field(
        description="Aggregation weights per site (proportional to dataset size)",
    )
    aggregated_metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Weighted average of site metrics",
    )
    timestamp: float = Field(default_factory=time.time)

    model_config = {"arbitrary_types_allowed": True}


# ── Round Status ─────────────────────────────────────────────────────────


class FLRoundStatus(BaseModel):
    """Status of a single FL round."""

    round_number: int
    status: RoundStatus
    config: FLRoundConfig
    participating_sites: list[str] = Field(default_factory=list)
    updates_received: int = 0
    updates_expected: int = 0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    aggregated_metrics: dict[str, float] = Field(default_factory=dict)


class FLSessionStatus(BaseModel):
    """Overall status of the federated learning session."""

    is_active: bool = False
    current_round: int = 0
    total_rounds: int = 0
    registered_sites: list[SiteInfo] = Field(default_factory=list)
    completed_rounds: list[FLRoundStatus] = Field(default_factory=list)
    global_dp_epsilon_spent: float = 0.0
    global_dp_budget: float = 8.0
    start_time: Optional[float] = None


# ── Privacy Accounting ───────────────────────────────────────────────────


class PrivacyBudget(BaseModel):
    """Tracks differential privacy budget consumption."""

    epsilon_total: float = Field(default=8.0, description="Total privacy budget (ε)")
    delta: float = Field(default=1e-5, description="Privacy parameter (δ)")
    epsilon_spent: float = Field(default=0.0, ge=0, description="Budget consumed so far")
    rounds_tracked: int = Field(default=0, ge=0)

    @property
    def epsilon_remaining(self) -> float:
        """Remaining privacy budget."""
        return max(0.0, self.epsilon_total - self.epsilon_spent)

    @property
    def is_exhausted(self) -> bool:
        """Whether the privacy budget is fully consumed."""
        return self.epsilon_spent >= self.epsilon_total

    def record_round(self, epsilon_round: float) -> None:
        """Record privacy budget consumption for one round."""
        self.epsilon_spent += epsilon_round
        self.rounds_tracked += 1
