"""
G_One_Sync AI — Federated Learning Coordinator
==================================================
Central server that orchestrates multi-site federated training.
Manages participant registration, round lifecycle, aggregation,
model versioning, and privacy budget tracking.

The coordinator:
    1. Waits for enough sites to register (fl_min_participants)
    2. Broadcasts the global model to all sites
    3. Collects encrypted/masked model updates after local training
    4. Aggregates updates using the configured strategy
    5. Tracks differential privacy budget
    6. Repeats for fl_rounds or until privacy budget is exhausted

Usage:
    python -m src.federated.coordinator
"""

from __future__ import annotations

import asyncio
import io
import json
import time
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import torch
from loguru import logger

from config.settings import federated_settings, model_settings
from src.federated.aggregator import (
    compute_aggregation_weights,
    create_aggregator,
    deserialize_state_dict,
    serialize_state_dict,
)
from src.federated.fl_schemas import (
    AggregatedResult,
    AggregationStrategy,
    FLRoundConfig,
    FLRoundStatus,
    FLSessionStatus,
    ModelUpdate,
    ParticipantStatus,
    RoundStatus,
    SiteInfo,
    SiteRegistration,
)
from src.federated.privacy import RDPAccountant
from src.federated.secure_aggregation import SecureAggregationManager


class FLCoordinator:
    """
    Federated Learning Coordinator Server.

    Manages the full lifecycle of federated training across
    multiple hospital sites. All coordination is in-memory;
    model checkpoints are persisted to disk.

    This class provides the business logic; the gRPC transport
    layer calls these methods.
    """

    def __init__(
        self,
        total_rounds: int | None = None,
        min_participants: int | None = None,
        aggregation_strategy: str | None = None,
        artifacts_dir: Path | None = None,
    ):
        self.total_rounds = total_rounds or federated_settings.fl_rounds
        self.min_participants = min_participants or federated_settings.fl_min_participants
        self.artifacts_dir = artifacts_dir or federated_settings.fl_artifacts_dir
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Aggregation
        strategy_name = aggregation_strategy or federated_settings.aggregation_strategy
        self.aggregation_strategy = AggregationStrategy(strategy_name)
        self.aggregator = create_aggregator(
            self.aggregation_strategy,
            mu=federated_settings.fedprox_mu,
        )

        # Site management
        self.sites: dict[str, SiteInfo] = {}

        # Round management
        self.current_round: int = 0
        self.round_status: RoundStatus = RoundStatus.WAITING
        self._round_updates: dict[str, ModelUpdate] = {}
        self._round_start_time: float = 0.0
        self.completed_rounds: list[FLRoundStatus] = []

        # Global model state
        self._global_state: dict[str, dict[str, torch.Tensor]] = {}

        # Privacy accounting
        self.dp_accountant = RDPAccountant(
            epsilon_budget=federated_settings.dp_epsilon,
            delta=federated_settings.dp_delta,
        )

        # Secure aggregation
        self.secure_agg: Optional[SecureAggregationManager] = None
        if federated_settings.secure_aggregation:
            self.secure_agg = SecureAggregationManager(round_number=0)

        # Session tracking
        self._is_active = False
        self._session_start_time: Optional[float] = None

        logger.info(
            "FL Coordinator initialized | rounds={} | min_sites={} | strategy={} | "
            "DP ε={} δ={} | secure_agg={}",
            self.total_rounds,
            self.min_participants,
            self.aggregation_strategy.value,
            federated_settings.dp_epsilon,
            federated_settings.dp_delta,
            federated_settings.secure_aggregation,
        )

    # ── Site Registration ────────────────────────────────────────────

    def register_site(self, registration: SiteRegistration) -> SiteInfo:
        """Register a new participant site.

        Args:
            registration: Site registration message.

        Returns:
            SiteInfo with assigned status.
        """
        site_info = SiteInfo(
            site_id=registration.site_id,
            site_name=registration.site_name,
            dataset_size=registration.dataset_size,
            model_types=registration.model_types,
            gpu_available=registration.gpu_available,
            status=ParticipantStatus.REGISTERED,
            last_heartbeat=time.time(),
        )
        self.sites[registration.site_id] = site_info

        logger.info(
            "Site registered: {} ({}) | {} samples | models: {}",
            registration.site_id,
            registration.site_name,
            registration.dataset_size,
            registration.model_types,
        )

        return site_info

    def unregister_site(self, site_id: str) -> bool:
        """Remove a site from the federation."""
        if site_id in self.sites:
            del self.sites[site_id]
            logger.info("Site unregistered: {}", site_id)
            return True
        return False

    def heartbeat(self, site_id: str) -> bool:
        """Update site heartbeat timestamp."""
        if site_id in self.sites:
            self.sites[site_id].last_heartbeat = time.time()
            return True
        return False

    @property
    def active_sites(self) -> list[SiteInfo]:
        """Get currently active (non-disconnected) sites."""
        cutoff = time.time() - 300  # 5 minute timeout
        active = []
        for site in self.sites.values():
            if site.last_heartbeat >= cutoff:
                active.append(site)
            elif site.status != ParticipantStatus.DISCONNECTED:
                site.status = ParticipantStatus.DISCONNECTED
                logger.warning("Site {} marked disconnected (heartbeat timeout)", site.site_id)
        return active

    @property
    def ready_to_start(self) -> bool:
        """Check if enough sites are registered to begin training."""
        return len(self.active_sites) >= self.min_participants

    # ── Global Model Management ──────────────────────────────────────

    def set_initial_model(
        self,
        model_type: str,
        state_dict: dict[str, torch.Tensor],
    ) -> None:
        """Set the initial global model for a model type.

        Called before FL rounds begin with a pre-trained or
        randomly initialized model.

        Args:
            model_type: Model architecture name (e.g., 'bilstm', 'transformer').
            state_dict: Initial model state dict.
        """
        self._global_state[model_type] = {
            k: v.cpu().clone() for k, v in state_dict.items()
        }
        logger.info(
            "Initial global model set for '{}': {} parameters",
            model_type,
            len(state_dict),
        )

    def get_global_model(self, model_type: str) -> Optional[dict[str, torch.Tensor]]:
        """Get the current global model state for a model type."""
        return self._global_state.get(model_type)

    def get_global_model_bytes(self, model_type: str) -> Optional[bytes]:
        """Get the serialized global model for network transfer."""
        state = self._global_state.get(model_type)
        if state is None:
            return None
        return serialize_state_dict(state)

    # ── Round Lifecycle ──────────────────────────────────────────────

    def start_round(self) -> FLRoundConfig:
        """Begin a new FL round.

        Returns:
            Round configuration for participants.

        Raises:
            RuntimeError: If not enough sites or privacy budget exhausted.
        """
        if not self.ready_to_start:
            raise RuntimeError(
                f"Cannot start round: {len(self.active_sites)} active sites "
                f"< {self.min_participants} required"
            )

        if self.dp_accountant.is_budget_exhausted():
            raise RuntimeError("Privacy budget exhausted — cannot start new round")

        self.current_round += 1
        self.round_status = RoundStatus.TRAINING
        self._round_updates.clear()
        self._round_start_time = time.time()

        # Update secure aggregation for new round
        if self.secure_agg:
            self.secure_agg = SecureAggregationManager(round_number=self.current_round)
            site_ids = [s.site_id for s in self.active_sites]
            self.secure_agg.register_sites(site_ids)

        # Mark all active sites as training
        for site in self.active_sites:
            site.status = ParticipantStatus.TRAINING

        if not self._is_active:
            self._is_active = True
            self._session_start_time = time.time()

        config = FLRoundConfig(
            round_number=self.current_round,
            total_rounds=self.total_rounds,
            local_epochs=federated_settings.fl_epochs_per_round,
            learning_rate=federated_settings.fl_learning_rate,
            batch_size=model_settings.batch_size,
            aggregation_strategy=self.aggregation_strategy,
            fedprox_mu=federated_settings.fedprox_mu,
            dp_enabled=federated_settings.dp_enabled,
            dp_max_grad_norm=federated_settings.dp_max_grad_norm,
            dp_noise_multiplier=federated_settings.dp_noise_multiplier,
        )

        logger.info(
            "╔═══ FL Round {}/{} STARTED ═══╗ | {} sites | strategy={}",
            self.current_round,
            self.total_rounds,
            len(self.active_sites),
            self.aggregation_strategy.value,
        )

        return config

    def submit_update(self, update: ModelUpdate) -> bool:
        """Receive a model update from a participant site.

        Args:
            update: Serialized model update from a site.

        Returns:
            True if the update was accepted.
        """
        if update.round_number != self.current_round:
            logger.warning(
                "Rejected update from {} — round mismatch ({} != {})",
                update.site_id,
                update.round_number,
                self.current_round,
            )
            return False

        if update.site_id not in self.sites:
            logger.warning("Rejected update from unregistered site: {}", update.site_id)
            return False

        self._round_updates[update.site_id] = update
        self.sites[update.site_id].status = ParticipantStatus.SUBMITTED
        self.sites[update.site_id].total_training_time_s += update.training_time_s

        logger.info(
            "Update received: site={} | round={} | model={} | "
            "train_time={:.1f}s | ε_spent={:.4f} | "
            "{}/{} updates collected",
            update.site_id,
            update.round_number,
            update.model_type,
            update.training_time_s,
            update.dp_epsilon_spent,
            len(self._round_updates),
            len(self.active_sites),
        )

        return True

    @property
    def round_complete(self) -> bool:
        """Check if all active sites have submitted updates."""
        active_ids = {s.site_id for s in self.active_sites}
        submitted_ids = set(self._round_updates.keys())
        return active_ids.issubset(submitted_ids)

    def aggregate_round(self) -> dict[str, AggregatedResult]:
        """Aggregate all collected updates for the current round.

        Returns:
            Dict of model_type → AggregatedResult.

        Raises:
            RuntimeError: If no updates to aggregate.
        """
        if not self._round_updates:
            raise RuntimeError("No updates to aggregate")

        self.round_status = RoundStatus.AGGREGATING

        # Group updates by model type
        updates_by_model: dict[str, list[ModelUpdate]] = {}
        for update in self._round_updates.values():
            updates_by_model.setdefault(update.model_type, []).append(update)

        results: dict[str, AggregatedResult] = {}

        for model_type, updates in updates_by_model.items():
            # Deserialize state dicts
            state_dicts = [
                deserialize_state_dict(u.state_dict_bytes) for u in updates
            ]
            dataset_sizes = [u.dataset_size for u in updates]
            weights = compute_aggregation_weights(dataset_sizes)

            # Aggregate
            aggregated_state = self.aggregator.aggregate(state_dicts, weights)

            # Store as new global model
            self._global_state[model_type] = aggregated_state

            # Compute weighted average of site metrics
            agg_metrics: dict[str, float] = {}
            for metric_key in updates[0].local_metrics:
                weighted_sum = sum(
                    u.local_metrics.get(metric_key, 0) * w
                    for u, w in zip(updates, weights)
                )
                agg_metrics[metric_key] = weighted_sum

            results[model_type] = AggregatedResult(
                round_number=self.current_round,
                aggregation_strategy=self.aggregation_strategy,
                model_type=model_type,
                global_state_bytes=serialize_state_dict(aggregated_state),
                participating_sites=[u.site_id for u in updates],
                site_weights={u.site_id: w for u, w in zip(updates, weights)},
                aggregated_metrics=agg_metrics,
            )

            logger.info(
                "Aggregated '{}': {} sites | metrics: {}",
                model_type,
                len(updates),
                {k: f"{v:.4f}" for k, v in agg_metrics.items()},
            )

        # Record DP budget
        if federated_settings.dp_enabled:
            sample_rate = model_settings.batch_size / max(
                sum(u.dataset_size for u in self._round_updates.values()), 1
            )
            num_steps = (
                federated_settings.fl_epochs_per_round
                * max(sum(u.dataset_size for u in self._round_updates.values()), 1)
                // model_settings.batch_size
            )
            self.dp_accountant.record_round(
                noise_multiplier=federated_settings.dp_noise_multiplier,
                sample_rate=sample_rate,
                num_steps=max(num_steps, 1),
            )

        # Save checkpoint
        self._save_round_checkpoint(results)

        # Record round status
        round_status = FLRoundStatus(
            round_number=self.current_round,
            status=RoundStatus.COMPLETED,
            config=FLRoundConfig(
                round_number=self.current_round,
                total_rounds=self.total_rounds,
            ),
            participating_sites=list(self._round_updates.keys()),
            updates_received=len(self._round_updates),
            updates_expected=len(self.active_sites),
            start_time=self._round_start_time,
            end_time=time.time(),
            aggregated_metrics=next(iter(results.values())).aggregated_metrics
            if results
            else {},
        )
        self.completed_rounds.append(round_status)

        # Update site stats
        for site_id in self._round_updates:
            if site_id in self.sites:
                self.sites[site_id].rounds_completed += 1
                self.sites[site_id].status = ParticipantStatus.IDLE

        self.round_status = RoundStatus.COMPLETED

        logger.info(
            "╚═══ FL Round {}/{} COMPLETED ═══╝ | ε_total={:.4f}/{:.1f}",
            self.current_round,
            self.total_rounds,
            self.dp_accountant.epsilon_spent,
            federated_settings.dp_epsilon,
        )

        return results

    def _save_round_checkpoint(
        self,
        results: dict[str, AggregatedResult],
    ) -> None:
        """Save round results to disk."""
        round_dir = self.artifacts_dir / f"round_{self.current_round:04d}"
        round_dir.mkdir(parents=True, exist_ok=True)

        for model_type, result in results.items():
            # Save aggregated model
            model_path = round_dir / f"{model_type}_global.pt"
            state_dict = deserialize_state_dict(result.global_state_bytes)
            torch.save(state_dict, model_path)

            # Save metadata
            meta = {
                "round_number": result.round_number,
                "model_type": model_type,
                "strategy": result.aggregation_strategy.value,
                "participating_sites": result.participating_sites,
                "site_weights": result.site_weights,
                "aggregated_metrics": result.aggregated_metrics,
                "dp_epsilon_spent": self.dp_accountant.epsilon_spent,
                "timestamp": result.timestamp,
            }
            meta_path = round_dir / f"{model_type}_meta.json"
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2, default=str)

        logger.info("Round {} checkpoint saved → {}", self.current_round, round_dir)

    # ── Session Status ───────────────────────────────────────────────

    def get_session_status(self) -> FLSessionStatus:
        """Get the overall federated learning session status."""
        return FLSessionStatus(
            is_active=self._is_active,
            current_round=self.current_round,
            total_rounds=self.total_rounds,
            registered_sites=list(self.sites.values()),
            completed_rounds=self.completed_rounds,
            global_dp_epsilon_spent=self.dp_accountant.epsilon_spent,
            global_dp_budget=federated_settings.dp_epsilon,
            start_time=self._session_start_time,
        )

    @property
    def is_training_complete(self) -> bool:
        """Check if all FL rounds are done or budget exhausted."""
        return (
            self.current_round >= self.total_rounds
            or self.dp_accountant.is_budget_exhausted()
        )

    # ── Full Training Loop ───────────────────────────────────────────

    async def run_training_loop(
        self,
        collect_timeout_s: float = 600.0,
        poll_interval_s: float = 5.0,
    ) -> FLSessionStatus:
        """Run the complete FL training loop (coordinator-side).

        Waits for site registrations, then iterates through rounds
        collecting updates and aggregating.

        Args:
            collect_timeout_s: Max seconds to wait for all updates per round.
            poll_interval_s: Polling interval for checking updates.

        Returns:
            Final session status.
        """
        logger.info("═══ FL Training Loop Started ═══")

        # Wait for minimum participants
        logger.info("Waiting for {} participants...", self.min_participants)
        while not self.ready_to_start:
            await asyncio.sleep(poll_interval_s)

        logger.info(
            "{} sites registered. Starting {} rounds.",
            len(self.active_sites),
            self.total_rounds,
        )

        while not self.is_training_complete:
            try:
                # Start round
                self.start_round()

                # Wait for all updates (with timeout)
                start = time.time()
                while not self.round_complete:
                    if time.time() - start > collect_timeout_s:
                        logger.warning(
                            "Round {} timed out after {:.0f}s — aggregating partial updates",
                            self.current_round,
                            collect_timeout_s,
                        )
                        break
                    await asyncio.sleep(poll_interval_s)

                # Aggregate
                if self._round_updates:
                    self.aggregate_round()
                else:
                    logger.error("Round {} — no updates received, skipping", self.current_round)

            except Exception as e:
                logger.error("Round {} failed: {}", self.current_round, e)
                self.round_status = RoundStatus.FAILED
                break

        self._is_active = False
        logger.info(
            "═══ FL Training Complete ═══ | {} rounds | ε_spent={:.4f}",
            self.current_round,
            self.dp_accountant.epsilon_spent,
        )

        return self.get_session_status()


# ── Entrypoint ───────────────────────────────────────────────────────────


if __name__ == "__main__":
    import sys

    if federated_settings.fl_role != "coordinator":
        logger.error("This node is not configured as an FL coordinator (fl_role={})",
                      federated_settings.fl_role)
        sys.exit(1)

    coordinator = FLCoordinator()
    asyncio.run(coordinator.run_training_loop())
