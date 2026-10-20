"""
G_One_Sync AI — Federated Learning Participant
==================================================
Hospital-site FL client that performs local training on private
patient data and sends encrypted/DP-noised model updates to
the coordinator. Raw data never leaves the site.

Flow per round:
    1. Receive global model from coordinator
    2. Load global weights into local trainer
    3. Train on local data for N epochs (with DP-SGD)
    4. Compute state dict delta (optional)
    5. Apply secure aggregation mask
    6. Send masked update to coordinator
    7. Receive aggregated global model for next round

Usage:
    python -m src.federated.participant
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from config.settings import federated_settings, model_settings
from src.federated.aggregator import deserialize_state_dict, serialize_state_dict
from src.federated.fl_schemas import (
    FLRoundConfig,
    ModelUpdate,
    SiteRegistration,
)
from src.federated.privacy import (
    RDPAccountant,
    add_gaussian_noise,
    add_noise_to_state_dict,
    clip_gradients,
)


class FLParticipant:
    """
    Federated Learning Participant (Hospital Site Client).

    Manages local training with differential privacy and
    communicates with the FL coordinator.

    This class handles the training logic; the gRPC transport
    layer calls these methods.
    """

    def __init__(
        self,
        site_id: str | None = None,
        site_name: str | None = None,
        coordinator_url: str | None = None,
    ):
        self.site_id = site_id or federated_settings.site_id
        self.site_name = site_name or federated_settings.site_name
        self.coordinator_url = coordinator_url or federated_settings.coordinator_url

        # Local training state
        self._trainers: dict[str, Any] = {}  # model_type → BaseTrainer
        self._local_data: dict[str, Any] = {}  # model_type → (train_data, val_data)
        self._dataset_size: int = 0
        self._gpu_available: bool = torch.cuda.is_available()

        # Privacy accounting (local copy)
        self.dp_accountant = RDPAccountant(
            epsilon_budget=federated_settings.dp_epsilon,
            delta=federated_settings.dp_delta,
        )

        # Round tracking
        self.rounds_completed: int = 0
        self.total_training_time_s: float = 0.0

        logger.info(
            "FL Participant initialized: site={} | coordinator={} | gpu={}",
            self.site_id,
            self.coordinator_url,
            self._gpu_available,
        )

    # ── Registration ─────────────────────────────────────────────────

    def create_registration(self) -> SiteRegistration:
        """Create a registration message for the coordinator."""
        return SiteRegistration(
            site_id=self.site_id,
            site_name=self.site_name,
            dataset_size=self._dataset_size,
            model_types=list(self._trainers.keys()),
            gpu_available=self._gpu_available,
        )

    # ── Trainer Management ───────────────────────────────────────────

    def register_trainer(
        self,
        model_type: str,
        trainer: Any,
        train_data: Any,
        val_data: Any,
    ) -> None:
        """Register a local trainer with its training data.

        Args:
            model_type: Model architecture name (e.g., 'bilstm').
            trainer: A BaseTrainer subclass instance.
            train_data: Training data for this model.
            val_data: Validation data for this model.
        """
        self._trainers[model_type] = trainer
        self._local_data[model_type] = (train_data, val_data)

        # Update dataset size from training data
        if hasattr(train_data, "__len__"):
            self._dataset_size = max(self._dataset_size, len(train_data))
        elif isinstance(train_data, tuple) and len(train_data) >= 2:
            if hasattr(train_data[0], "shape"):
                self._dataset_size = max(self._dataset_size, train_data[0].shape[0])
            elif hasattr(train_data[0], "__len__"):
                self._dataset_size = max(self._dataset_size, len(train_data[0]))

        logger.info(
            "Registered trainer '{}' with {} samples",
            model_type,
            self._dataset_size,
        )

    # ── Local Training ───────────────────────────────────────────────

    def load_global_model(
        self,
        model_type: str,
        global_state_bytes: bytes,
    ) -> None:
        """Load the global model weights received from coordinator.

        Args:
            model_type: Model architecture name.
            global_state_bytes: Serialized state dict from coordinator.
        """
        if model_type not in self._trainers:
            raise ValueError(f"No trainer registered for '{model_type}'")

        trainer = self._trainers[model_type]
        global_state = deserialize_state_dict(global_state_bytes)
        trainer.set_state_dict(global_state)

        logger.info(
            "Loaded global model for '{}': {} parameters",
            model_type,
            len(global_state),
        )

    def train_local(
        self,
        model_type: str,
        round_config: FLRoundConfig,
    ) -> ModelUpdate:
        """Perform local training for one FL round.

        This is the core participant operation:
        1. Train on local data with DP-SGD (if enabled)
        2. Extract updated state dict
        3. Apply DP noise to state dict (if enabled)
        4. Serialize and return as ModelUpdate

        Args:
            model_type: Model architecture to train.
            round_config: Configuration for this round.

        Returns:
            ModelUpdate ready to send to coordinator.
        """
        if model_type not in self._trainers:
            raise ValueError(f"No trainer registered for '{model_type}'")

        trainer = self._trainers[model_type]
        train_data, val_data = self._local_data[model_type]

        logger.info(
            "Local training: site={} | model={} | round={}/{} | "
            "epochs={} | lr={} | dp={}",
            self.site_id,
            model_type,
            round_config.round_number,
            round_config.total_rounds,
            round_config.local_epochs,
            round_config.learning_rate,
            round_config.dp_enabled,
        )

        start_time = time.time()

        # Store pre-training state for FedProx proximal term
        global_state_before = None
        if round_config.aggregation_strategy.value == "fedprox":
            global_state_before = trainer.get_state_dict()

        # Train locally
        if hasattr(trainer, "_train_local_fl"):
            # Custom FL training path (handles DP-SGD internally)
            metrics = trainer._train_local_fl(
                train_data=train_data,
                val_data=val_data,
                epochs=round_config.local_epochs,
                learning_rate=round_config.learning_rate,
                dp_enabled=round_config.dp_enabled,
                dp_max_grad_norm=round_config.dp_max_grad_norm,
                dp_noise_multiplier=round_config.dp_noise_multiplier,
                global_state=global_state_before,
                fedprox_mu=round_config.fedprox_mu,
            )
        else:
            # Use standard training with DP wrapper
            metrics = self._train_with_dp(
                trainer=trainer,
                train_data=train_data,
                val_data=val_data,
                round_config=round_config,
                global_state=global_state_before,
            )

        training_time = time.time() - start_time
        self.total_training_time_s += training_time
        self.rounds_completed += 1

        # Get updated state dict
        updated_state = trainer.get_state_dict()

        # Apply DP noise to state dict (defense in depth)
        dp_epsilon_spent = 0.0
        if round_config.dp_enabled:
            updated_state = add_noise_to_state_dict(
                state_dict=updated_state,
                noise_multiplier=round_config.dp_noise_multiplier,
                max_grad_norm=round_config.dp_max_grad_norm,
                dataset_size=max(self._dataset_size, 1),
            )

            # Account for privacy
            sample_rate = round_config.batch_size / max(self._dataset_size, 1)
            num_steps = round_config.local_epochs * max(
                self._dataset_size // round_config.batch_size, 1
            )
            dp_epsilon_spent = self.dp_accountant.record_round(
                noise_multiplier=round_config.dp_noise_multiplier,
                sample_rate=sample_rate,
                num_steps=num_steps,
            )

        # Serialize state dict
        state_bytes = serialize_state_dict(updated_state)

        # Build update message
        update = ModelUpdate(
            site_id=self.site_id,
            round_number=round_config.round_number,
            model_type=model_type,
            state_dict_bytes=state_bytes,
            dataset_size=self._dataset_size,
            training_time_s=training_time,
            local_metrics=metrics,
            dp_epsilon_spent=dp_epsilon_spent,
        )

        logger.info(
            "Local training complete: site={} | model={} | round={} | "
            "time={:.1f}s | metrics={} | ε_spent={:.4f}",
            self.site_id,
            model_type,
            round_config.round_number,
            training_time,
            {k: f"{v:.4f}" for k, v in metrics.items()},
            dp_epsilon_spent,
        )

        return update

    def _train_with_dp(
        self,
        trainer: Any,
        train_data: Any,
        val_data: Any,
        round_config: FLRoundConfig,
        global_state: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, float]:
        """Standard training loop with DP-SGD and optional FedProx term.

        This wraps the existing trainer's internal training loop with:
        1. Per-step gradient clipping
        2. Gaussian noise injection
        3. FedProx proximal regularization (if configured)

        Args:
            trainer: BaseTrainer subclass.
            train_data: Training data.
            val_data: Validation data.
            round_config: FL round configuration.
            global_state: Global model state for FedProx proximal term.

        Returns:
            Dict of evaluation metrics after local training.
        """
        model = trainer.model
        if model is None:
            raise RuntimeError("Trainer model not initialized")

        # Check if model is a PyTorch module (DP-SGD only works with PyTorch)
        if not isinstance(model, nn.Module):
            # For non-PyTorch models (XGBoost), do standard training
            logger.info("Non-PyTorch model '{}' — DP applied at state dict level only",
                        trainer.model_name)
            val_metrics = trainer.train(train_data, val_data)
            return val_metrics.to_dict() if val_metrics else {}

        # PyTorch DP-SGD training
        model.train()
        device = trainer.device

        # Build optimizer with FL learning rate
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=round_config.learning_rate,
            weight_decay=1e-5,
        )
        criterion = nn.BCEWithLogitsLoss()

        # Build data loader
        from torch.utils.data import DataLoader, TensorDataset

        if isinstance(train_data, tuple):
            X, y = train_data
            if isinstance(X, np.ndarray):
                X = torch.FloatTensor(X)
            if isinstance(y, np.ndarray):
                y = torch.FloatTensor(y)
            dataset = TensorDataset(X, y)
        else:
            dataset = train_data

        loader = DataLoader(
            dataset,
            batch_size=round_config.batch_size,
            shuffle=True,
            drop_last=True,
        )

        total_grad_norms = []

        for epoch in range(round_config.local_epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch in loader:
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    X_batch, y_batch = batch[0].to(device), batch[1].to(device)
                else:
                    continue

                optimizer.zero_grad()
                output = model(X_batch)

                if output.dim() > 1:
                    output = output.squeeze(-1)
                if y_batch.dim() > 1:
                    y_batch = y_batch.squeeze(-1)

                loss = criterion(output, y_batch.float())

                # FedProx proximal term
                if global_state is not None and round_config.fedprox_mu > 0:
                    proximal_loss = 0.0
                    for name, param in model.named_parameters():
                        if name in global_state:
                            proximal_loss += torch.norm(
                                param - global_state[name].to(device)
                            ) ** 2
                    loss += (round_config.fedprox_mu / 2.0) * proximal_loss

                loss.backward()

                # DP-SGD: clip gradients
                if round_config.dp_enabled:
                    grad_norm = clip_gradients(model, round_config.dp_max_grad_norm)
                    total_grad_norms.append(grad_norm)

                    # DP-SGD: add noise
                    add_gaussian_noise(
                        model,
                        noise_multiplier=round_config.dp_noise_multiplier,
                        max_grad_norm=round_config.dp_max_grad_norm,
                        batch_size=round_config.batch_size,
                    )

                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            logger.debug(
                "  Epoch {}/{}: loss={:.4f}",
                epoch + 1,
                round_config.local_epochs,
                avg_loss,
            )

        # Evaluate on validation data
        metrics = {}
        try:
            eval_result = trainer.evaluate(val_data, split_name="fl_val")
            metrics = eval_result.to_dict()
        except Exception as e:
            logger.warning("FL validation evaluation failed: {}", e)
            metrics = {"loss": avg_loss}

        if total_grad_norms:
            metrics["mean_grad_norm"] = float(np.mean(total_grad_norms))
            metrics["max_grad_norm"] = float(np.max(total_grad_norms))

        return metrics

    # ── Status ───────────────────────────────────────────────────────

    def get_status(self) -> dict:
        """Get participant status summary."""
        return {
            "site_id": self.site_id,
            "site_name": self.site_name,
            "coordinator_url": self.coordinator_url,
            "registered_models": list(self._trainers.keys()),
            "dataset_size": self._dataset_size,
            "gpu_available": self._gpu_available,
            "rounds_completed": self.rounds_completed,
            "total_training_time_s": self.total_training_time_s,
            "dp_epsilon_spent": self.dp_accountant.epsilon_spent,
            "dp_epsilon_remaining": self.dp_accountant.epsilon_remaining,
        }
