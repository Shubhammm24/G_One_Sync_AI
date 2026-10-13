"""
G_One_Sync AI — Federated Learning Tests
============================================
Comprehensive test suite for Phase 6 federated learning:
- Aggregation strategies (FedAvg, FedProx, FedBN)
- Differential privacy (gradient clipping, noise, accounting)
- Secure aggregation (mask generation, cancellation)
- Coordinator (site registration, round lifecycle)
- Participant (local training, model update creation)
- BaseTrainer FL extensions (get_state_dict, set_state_dict)
- End-to-end simulation (3 sites, multiple rounds)
"""

from __future__ import annotations

import io
import time
from collections import OrderedDict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from config.settings import FederatedSettings
from src.federated.aggregator import (
    FedAvgAggregator,
    FedBNAggregator,
    FedProxAggregator,
    compute_aggregation_weights,
    create_aggregator,
    deserialize_state_dict,
    serialize_state_dict,
)
from src.federated.fl_schemas import (
    AggregationStrategy,
    FLRoundConfig,
    FLRoundStatus,
    FLSessionStatus,
    ModelUpdate,
    ParticipantStatus,
    PrivacyBudget,
    RoundStatus,
    SiteInfo,
    SiteRegistration,
)
from src.federated.privacy import (
    RDPAccountant,
    add_gaussian_noise,
    add_noise_to_state_dict,
    clip_gradients,
)
from src.federated.secure_aggregation import SecureAggregationManager


# ── Fixtures ─────────────────────────────────────────────────────────────


class SimpleModel(nn.Module):
    """Minimal PyTorch model for testing."""

    def __init__(self, input_size: int = 10, hidden_size: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.bn1(self.fc1(x)))
        return torch.sigmoid(self.fc2(x))


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return SimpleModel(input_size=10, hidden_size=8)


@pytest.fixture
def two_state_dicts():
    """Create two slightly different state dicts simulating two sites."""
    model1 = SimpleModel(input_size=10, hidden_size=8)
    model2 = SimpleModel(input_size=10, hidden_size=8)

    # Make model2 weights slightly different
    with torch.no_grad():
        for p in model2.parameters():
            p.add_(torch.randn_like(p) * 0.1)

    return model1.state_dict(), model2.state_dict()


@pytest.fixture
def three_state_dicts():
    """Create three state dicts simulating three hospital sites."""
    models = [SimpleModel(input_size=10, hidden_size=8) for _ in range(3)]
    with torch.no_grad():
        for i, model in enumerate(models):
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.1 * (i + 1))
    return [m.state_dict() for m in models]


@pytest.fixture
def round_config():
    """Create a standard FL round configuration."""
    return FLRoundConfig(
        round_number=1,
        total_rounds=10,
        local_epochs=3,
        learning_rate=1e-3,
        batch_size=32,
        aggregation_strategy=AggregationStrategy.FEDAVG,
        dp_enabled=True,
        dp_max_grad_norm=1.0,
        dp_noise_multiplier=1.1,
    )


# ═════════════════════════════════════════════════════════════════════════
# AGGREGATION STRATEGY TESTS
# ═════════════════════════════════════════════════════════════════════════


class TestFedAvgAggregator:
    """Tests for Federated Averaging aggregation."""

    def test_equal_weight_aggregation(self, two_state_dicts):
        """FedAvg with equal weights should average parameters."""
        sd1, sd2 = two_state_dicts
        aggregator = FedAvgAggregator()
        result = aggregator.aggregate([sd1, sd2], [1.0, 1.0])

        for key in sd1:
            expected = (sd1[key].float() + sd2[key].float()) / 2.0
            torch.testing.assert_close(result[key], expected, atol=1e-5, rtol=1e-5)

    def test_weighted_aggregation(self, two_state_dicts):
        """FedAvg with 3:1 weights should favor first site."""
        sd1, sd2 = two_state_dicts
        aggregator = FedAvgAggregator()
        result = aggregator.aggregate([sd1, sd2], [3.0, 1.0])

        for key in sd1:
            expected = 0.75 * sd1[key].float() + 0.25 * sd2[key].float()
            torch.testing.assert_close(result[key], expected, atol=1e-5, rtol=1e-5)

    def test_single_site(self, two_state_dicts):
        """Single site aggregation should return the same state dict."""
        sd1, _ = two_state_dicts
        aggregator = FedAvgAggregator()
        result = aggregator.aggregate([sd1], [1.0])

        for key in sd1:
            torch.testing.assert_close(result[key], sd1[key].float(), atol=1e-6, rtol=1e-6)

    def test_three_sites(self, three_state_dicts):
        """FedAvg across three sites should produce correct weighted average."""
        aggregator = FedAvgAggregator()
        weights = [5000, 3000, 2000]  # Proportional to dataset sizes
        result = aggregator.aggregate(three_state_dicts, weights)

        # Verify result has same keys
        assert set(result.keys()) == set(three_state_dicts[0].keys())

        # Verify result is not identical to any single site
        for sd in three_state_dicts:
            for key in sd:
                if not torch.equal(result[key], sd[key].float()):
                    break

    def test_empty_state_dicts_raises(self):
        """Empty input should raise ValueError."""
        aggregator = FedAvgAggregator()
        with pytest.raises(ValueError, match="No state dicts"):
            aggregator.aggregate([], [])

    def test_mismatched_lengths_raises(self, two_state_dicts):
        """Mismatched state dicts and weights should raise."""
        sd1, sd2 = two_state_dicts
        aggregator = FedAvgAggregator()
        with pytest.raises(ValueError, match="Mismatch"):
            aggregator.aggregate([sd1, sd2], [1.0])


class TestFedProxAggregator:
    """Tests for FedProx aggregation (same as FedAvg at aggregation step)."""

    def test_same_as_fedavg(self, two_state_dicts):
        """FedProx aggregation should be identical to FedAvg."""
        sd1, sd2 = two_state_dicts
        fedavg = FedAvgAggregator()
        fedprox = FedProxAggregator(mu=0.01)

        result_avg = fedavg.aggregate([sd1, sd2], [1.0, 1.0])
        result_prox = fedprox.aggregate([sd1, sd2], [1.0, 1.0])

        for key in result_avg:
            torch.testing.assert_close(result_avg[key], result_prox[key], atol=1e-6, rtol=1e-6)

    def test_mu_stored(self):
        """FedProx should store the proximal term weight."""
        aggregator = FedProxAggregator(mu=0.05)
        assert aggregator.mu == 0.05
        assert aggregator.strategy == AggregationStrategy.FEDPROX


class TestFedBNAggregator:
    """Tests for FedBN aggregation (skip BatchNorm parameters)."""

    def test_bn_layers_excluded(self, two_state_dicts):
        """BatchNorm parameters should not be averaged — kept from first site."""
        sd1, sd2 = two_state_dicts
        aggregator = FedBNAggregator()
        result = aggregator.aggregate([sd1, sd2], [1.0, 1.0])

        # BN keys should match first site (not averaged)
        for key in sd1:
            if aggregator._is_bn_key(key):
                torch.testing.assert_close(result[key], sd1[key], atol=1e-7, rtol=1e-7)

    def test_non_bn_layers_averaged(self, two_state_dicts):
        """Non-BN parameters should be averaged like FedAvg."""
        sd1, sd2 = two_state_dicts
        aggregator = FedBNAggregator()
        result = aggregator.aggregate([sd1, sd2], [1.0, 1.0])

        for key in sd1:
            if not aggregator._is_bn_key(key):
                expected = (sd1[key].float() + sd2[key].float()) / 2.0
                torch.testing.assert_close(result[key], expected, atol=1e-5, rtol=1e-5)

    def test_bn_key_detection(self):
        """Verify BN key pattern matching."""
        aggregator = FedBNAggregator()
        assert aggregator._is_bn_key("bn1.weight") is True
        assert aggregator._is_bn_key("bn1.running_mean") is True
        assert aggregator._is_bn_key("layer_norm.weight") is True
        assert aggregator._is_bn_key("fc1.weight") is False
        assert aggregator._is_bn_key("fc2.bias") is False


class TestAggregatorFactory:
    """Tests for the create_aggregator factory function."""

    def test_create_fedavg(self):
        agg = create_aggregator("fedavg")
        assert isinstance(agg, FedAvgAggregator)

    def test_create_fedprox(self):
        agg = create_aggregator("fedprox", mu=0.05)
        assert isinstance(agg, FedProxAggregator)
        assert agg.mu == 0.05

    def test_create_fedbn(self):
        agg = create_aggregator("fedbn")
        assert isinstance(agg, FedBNAggregator)

    def test_create_from_enum(self):
        agg = create_aggregator(AggregationStrategy.FEDAVG)
        assert isinstance(agg, FedAvgAggregator)

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError):
            create_aggregator("unknown")


# ═════════════════════════════════════════════════════════════════════════
# SERIALIZATION TESTS
# ═════════════════════════════════════════════════════════════════════════


class TestSerialization:
    """Tests for state dict serialization/deserialization."""

    def test_roundtrip(self, simple_model):
        """Serialize and deserialize should produce identical state dict."""
        original = simple_model.state_dict()
        data = serialize_state_dict(original)
        restored = deserialize_state_dict(data)

        for key in original:
            torch.testing.assert_close(original[key], restored[key])

    def test_serialized_is_bytes(self, simple_model):
        """Serialized output should be bytes."""
        data = serialize_state_dict(simple_model.state_dict())
        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_aggregation_weights(self):
        """compute_aggregation_weights should normalize correctly."""
        weights = compute_aggregation_weights([5000, 3000, 2000])
        assert abs(sum(weights) - 1.0) < 1e-6
        assert abs(weights[0] - 0.5) < 1e-6
        assert abs(weights[1] - 0.3) < 1e-6
        assert abs(weights[2] - 0.2) < 1e-6

    def test_zero_weights_equal(self):
        """Zero dataset sizes should produce equal weights."""
        weights = compute_aggregation_weights([0, 0, 0])
        assert all(abs(w - 1.0 / 3) < 1e-6 for w in weights)


# ═════════════════════════════════════════════════════════════════════════
# DIFFERENTIAL PRIVACY TESTS
# ═════════════════════════════════════════════════════════════════════════


class TestGradientClipping:
    """Tests for gradient clipping in DP-SGD."""

    def test_clips_large_gradients(self, simple_model):
        """Gradients exceeding max norm should be clipped."""
        # Create large gradients
        dummy = torch.randn(4, 10)
        output = simple_model(dummy)
        loss = output.sum()
        loss.backward()

        # Scale gradients to be large
        with torch.no_grad():
            for p in simple_model.parameters():
                if p.grad is not None:
                    p.grad *= 100.0

        original_norm = clip_gradients(simple_model, max_grad_norm=1.0)
        assert original_norm > 1.0  # Was large before clipping

        # Check clipped norm
        clipped_norm = torch.norm(
            torch.stack([torch.norm(p.grad, 2) for p in simple_model.parameters() if p.grad is not None]),
            2,
        ).item()
        assert clipped_norm <= 1.0 + 1e-5

    def test_no_clip_small_gradients(self, simple_model):
        """Small gradients should not be clipped."""
        dummy = torch.randn(4, 10)
        output = simple_model(dummy)
        loss = output.sum()
        loss.backward()

        # Scale gradients to be small
        with torch.no_grad():
            for p in simple_model.parameters():
                if p.grad is not None:
                    p.grad *= 0.001

        original_norm = clip_gradients(simple_model, max_grad_norm=100.0)
        assert original_norm < 100.0  # Already within bound


class TestGaussianNoise:
    """Tests for Gaussian noise injection."""

    def test_noise_modifies_gradients(self, simple_model):
        """Adding noise should change gradient values."""
        dummy = torch.randn(8, 10)
        output = simple_model(dummy)
        loss = output.sum()
        loss.backward()

        grads_before = {
            name: p.grad.clone()
            for name, p in simple_model.named_parameters()
            if p.grad is not None
        }

        add_gaussian_noise(simple_model, noise_multiplier=1.0, max_grad_norm=1.0, batch_size=8)

        for name, p in simple_model.named_parameters():
            if p.grad is not None and name in grads_before:
                assert not torch.equal(p.grad, grads_before[name])

    def test_state_dict_noise(self, simple_model):
        """add_noise_to_state_dict should modify floating point tensors."""
        sd = simple_model.state_dict()
        noised = add_noise_to_state_dict(sd, noise_multiplier=1.0, max_grad_norm=1.0, dataset_size=100)

        for key in sd:
            if sd[key].is_floating_point():
                assert not torch.equal(sd[key], noised[key])

    def test_state_dict_noise_preserves_int_tensors(self, simple_model):
        """Non-floating point tensors should not be noised."""
        sd = simple_model.state_dict()
        noised = add_noise_to_state_dict(sd, noise_multiplier=1.0, max_grad_norm=1.0, dataset_size=100)

        for key in sd:
            if not sd[key].is_floating_point():
                torch.testing.assert_close(sd[key], noised[key])


class TestRDPAccountant:
    """Tests for Rényi Differential Privacy accounting."""

    def test_initial_state(self):
        """Fresh accountant should have zero spent."""
        acc = RDPAccountant(epsilon_budget=8.0, delta=1e-5)
        assert acc.epsilon_spent == 0.0
        assert acc.epsilon_remaining == 8.0
        assert not acc.is_budget_exhausted()

    def test_budget_decreases(self):
        """Recording a round should consume some budget."""
        acc = RDPAccountant(epsilon_budget=8.0, delta=1e-5)
        epsilon = acc.record_round(
            noise_multiplier=1.1,
            sample_rate=0.01,
            num_steps=100,
        )
        assert epsilon > 0
        assert acc.epsilon_spent > 0
        assert acc.epsilon_remaining < 8.0

    def test_budget_exhaustion(self):
        """Many rounds should eventually exhaust the budget."""
        acc = RDPAccountant(epsilon_budget=1.0, delta=1e-5)

        for _ in range(1000):
            acc.record_round(
                noise_multiplier=0.5,
                sample_rate=0.1,
                num_steps=500,
            )
            if acc.is_budget_exhausted():
                break

        assert acc.is_budget_exhausted()

    def test_compute_round_epsilon(self):
        """compute_round_epsilon should return positive ε."""
        acc = RDPAccountant(epsilon_budget=8.0, delta=1e-5)
        eps = acc.compute_round_epsilon(
            noise_multiplier=1.1,
            sample_rate=0.01,
            num_steps=100,
        )
        assert eps >= 0

    def test_status_dict(self):
        """get_status should return a well-formed dict."""
        acc = RDPAccountant(epsilon_budget=8.0, delta=1e-5)
        status = acc.get_status()
        assert "epsilon_spent" in status
        assert "epsilon_remaining" in status
        assert "epsilon_budget" in status
        assert "delta" in status
        assert status["epsilon_budget"] == 8.0


# ═════════════════════════════════════════════════════════════════════════
# SECURE AGGREGATION TESTS
# ═════════════════════════════════════════════════════════════════════════


class TestSecureAggregation:
    """Tests for additive secret sharing secure aggregation."""

    def test_mask_cancellation_two_sites(self):
        """Masks from two sites should cancel to zero."""
        sa = SecureAggregationManager(round_number=1)
        site_ids = ["site-001", "site-002"]
        sa.register_sites(site_ids)

        shapes = {"fc.weight": torch.Size([8, 10]), "fc.bias": torch.Size([8])}
        dtypes = {"fc.weight": torch.float32, "fc.bias": torch.float32}

        assert sa.verify_mask_cancellation(site_ids, shapes, dtypes)

    def test_mask_cancellation_three_sites(self):
        """Masks from three sites should cancel to zero."""
        sa = SecureAggregationManager(round_number=1)
        site_ids = ["site-001", "site-002", "site-003"]
        sa.register_sites(site_ids)

        shapes = {
            "fc1.weight": torch.Size([8, 10]),
            "fc1.bias": torch.Size([8]),
            "fc2.weight": torch.Size([1, 8]),
        }
        dtypes = {k: torch.float32 for k in shapes}

        assert sa.verify_mask_cancellation(site_ids, shapes, dtypes)

    def test_masked_aggregation_recovers_true_aggregate(self):
        """Summing masked state dicts should approximately equal summing unmasked."""
        sa = SecureAggregationManager(round_number=1)
        site_ids = ["site-A", "site-B"]
        sa.register_sites(site_ids)

        # Create two state dicts
        sd1 = OrderedDict([
            ("w", torch.randn(4, 3)),
            ("b", torch.randn(4)),
        ])
        sd2 = OrderedDict([
            ("w", torch.randn(4, 3)),
            ("b", torch.randn(4)),
        ])

        shapes = {k: v.shape for k, v in sd1.items()}
        dtypes = {k: v.dtype for k, v in sd1.items()}

        # Generate masks
        mask1 = sa.generate_mask("site-A", site_ids, shapes, dtypes)
        mask2 = sa.generate_mask("site-B", site_ids, shapes, dtypes)

        # Apply masks
        masked1 = sa.apply_mask(sd1, mask1)
        masked2 = sa.apply_mask(sd2, mask2)

        # Aggregate with equal weights
        result = sa.unmask_aggregate([masked1, masked2], [1.0, 1.0])

        # Compare to true aggregate
        true_agg = OrderedDict()
        for key in sd1:
            true_agg[key] = (sd1[key].float() + sd2[key].float()) / 2.0

        for key in true_agg:
            torch.testing.assert_close(result[key], true_agg[key], atol=1e-4, rtol=1e-4)

    def test_different_rounds_different_masks(self):
        """Masks for different rounds should be different."""
        sa1 = SecureAggregationManager(round_number=1)
        sa2 = SecureAggregationManager(round_number=2)
        site_ids = ["site-A", "site-B"]
        sa1.register_sites(site_ids)
        sa2.register_sites(site_ids)

        shapes = {"w": torch.Size([4, 3])}
        dtypes = {"w": torch.float32}

        mask1 = sa1.generate_mask("site-A", site_ids, shapes, dtypes)
        mask2 = sa2.generate_mask("site-A", site_ids, shapes, dtypes)

        assert not torch.equal(mask1["w"], mask2["w"])


# ═════════════════════════════════════════════════════════════════════════
# FL SCHEMAS TESTS
# ═════════════════════════════════════════════════════════════════════════


class TestFLSchemas:
    """Tests for Pydantic FL message schemas."""

    def test_site_registration_valid(self):
        """Valid site registration should succeed."""
        reg = SiteRegistration(
            site_id="hospital-alpha",
            site_name="Alpha General Hospital",
            dataset_size=5000,
            model_types=["bilstm", "transformer"],
            gpu_available=True,
        )
        assert reg.site_id == "hospital-alpha"
        assert reg.dataset_size == 5000
        assert reg.timestamp > 0

    def test_round_config_defaults(self):
        """FLRoundConfig should have sensible defaults."""
        config = FLRoundConfig(round_number=1, total_rounds=50)
        assert config.local_epochs == 3
        assert config.learning_rate == 1e-3
        assert config.dp_enabled is True

    def test_privacy_budget(self):
        """PrivacyBudget tracking should work correctly."""
        budget = PrivacyBudget(epsilon_total=8.0, delta=1e-5)
        assert budget.epsilon_remaining == 8.0
        assert not budget.is_exhausted

        budget.record_round(2.0)
        assert budget.epsilon_spent == 2.0
        assert budget.epsilon_remaining == 6.0
        assert budget.rounds_tracked == 1

        budget.record_round(6.0)
        assert budget.is_exhausted

    def test_session_status_initial(self):
        """Initial session status should be inactive."""
        status = FLSessionStatus()
        assert status.is_active is False
        assert status.current_round == 0

    def test_model_update_schema(self):
        """ModelUpdate should accept valid data."""
        sd_bytes = serialize_state_dict({"w": torch.randn(4, 3)})
        update = ModelUpdate(
            site_id="site-001",
            round_number=1,
            model_type="bilstm",
            state_dict_bytes=sd_bytes,
            dataset_size=5000,
            training_time_s=120.5,
            local_metrics={"auroc": 0.85, "auprc": 0.72},
            dp_epsilon_spent=0.5,
        )
        assert update.dataset_size == 5000
        assert update.local_metrics["auroc"] == 0.85


# ═════════════════════════════════════════════════════════════════════════
# COORDINATOR TESTS
# ═════════════════════════════════════════════════════════════════════════


class TestFLCoordinator:
    """Tests for the FL Coordinator server."""

    def _make_coordinator(self, **kwargs):
        """Helper to create a coordinator with test defaults."""
        from src.federated.coordinator import FLCoordinator

        defaults = {
            "total_rounds": 5,
            "min_participants": 2,
            "aggregation_strategy": "fedavg",
        }
        defaults.update(kwargs)
        return FLCoordinator(**defaults)

    def test_site_registration(self):
        """Sites should register successfully."""
        coord = self._make_coordinator()
        reg = SiteRegistration(
            site_id="site-001",
            site_name="Test Hospital",
            dataset_size=5000,
            model_types=["bilstm"],
        )
        info = coord.register_site(reg)
        assert info.site_id == "site-001"
        assert info.status == ParticipantStatus.REGISTERED
        assert len(coord.sites) == 1

    def test_minimum_participants_check(self):
        """Coordinator should not start until min_participants met."""
        coord = self._make_coordinator(min_participants=2)
        assert not coord.ready_to_start

        coord.register_site(SiteRegistration(
            site_id="site-001", dataset_size=5000,
        ))
        assert not coord.ready_to_start

        coord.register_site(SiteRegistration(
            site_id="site-002", dataset_size=3000,
        ))
        assert coord.ready_to_start

    def test_start_round(self):
        """Starting a round should return valid config."""
        coord = self._make_coordinator(min_participants=2)
        coord.register_site(SiteRegistration(site_id="site-001", dataset_size=5000))
        coord.register_site(SiteRegistration(site_id="site-002", dataset_size=3000))

        config = coord.start_round()
        assert config.round_number == 1
        assert config.total_rounds == 5
        assert coord.round_status == RoundStatus.TRAINING

    def test_submit_and_aggregate(self):
        """Submitting updates and aggregating should produce valid result."""
        coord = self._make_coordinator(min_participants=2)
        coord.register_site(SiteRegistration(site_id="s1", dataset_size=5000))
        coord.register_site(SiteRegistration(site_id="s2", dataset_size=3000))

        # Set initial model
        model = SimpleModel()
        coord.set_initial_model("bilstm", model.state_dict())

        config = coord.start_round()

        # Submit updates
        sd1 = serialize_state_dict(SimpleModel().state_dict())
        sd2 = serialize_state_dict(SimpleModel().state_dict())

        coord.submit_update(ModelUpdate(
            site_id="s1", round_number=1, model_type="bilstm",
            state_dict_bytes=sd1, dataset_size=5000, training_time_s=60.0,
            local_metrics={"auroc": 0.85},
        ))
        coord.submit_update(ModelUpdate(
            site_id="s2", round_number=1, model_type="bilstm",
            state_dict_bytes=sd2, dataset_size=3000, training_time_s=40.0,
            local_metrics={"auroc": 0.82},
        ))

        assert coord.round_complete

        # Aggregate
        results = coord.aggregate_round()
        assert "bilstm" in results
        assert results["bilstm"].round_number == 1
        assert len(results["bilstm"].participating_sites) == 2
        assert coord.round_status == RoundStatus.COMPLETED

    def test_reject_wrong_round_update(self):
        """Updates for wrong round should be rejected."""
        coord = self._make_coordinator(min_participants=1)
        coord.register_site(SiteRegistration(site_id="s1", dataset_size=5000))
        coord.start_round()

        result = coord.submit_update(ModelUpdate(
            site_id="s1", round_number=99, model_type="bilstm",
            state_dict_bytes=b"", dataset_size=5000, training_time_s=0,
        ))
        assert result is False

    def test_session_status(self):
        """Session status should reflect current state."""
        coord = self._make_coordinator(min_participants=1)
        status = coord.get_session_status()
        assert status.is_active is False
        assert status.current_round == 0

    def test_unregister_site(self):
        """Unregistering a site should remove it."""
        coord = self._make_coordinator()
        coord.register_site(SiteRegistration(site_id="s1", dataset_size=5000))
        assert len(coord.sites) == 1
        assert coord.unregister_site("s1") is True
        assert len(coord.sites) == 0

    def test_training_complete(self):
        """is_training_complete should be True after all rounds."""
        coord = self._make_coordinator(total_rounds=1, min_participants=1)
        coord.register_site(SiteRegistration(site_id="s1", dataset_size=5000))

        coord.set_initial_model("bilstm", SimpleModel().state_dict())
        coord.start_round()

        sd = serialize_state_dict(SimpleModel().state_dict())
        coord.submit_update(ModelUpdate(
            site_id="s1", round_number=1, model_type="bilstm",
            state_dict_bytes=sd, dataset_size=5000, training_time_s=10,
            local_metrics={"auroc": 0.85},
        ))
        coord.aggregate_round()

        assert coord.is_training_complete


# ═════════════════════════════════════════════════════════════════════════
# PARTICIPANT TESTS
# ═════════════════════════════════════════════════════════════════════════


class TestFLParticipant:
    """Tests for the FL Participant (hospital site client)."""

    def test_registration_message(self):
        """Participant should produce a valid registration message."""
        from src.federated.participant import FLParticipant

        participant = FLParticipant(site_id="test-site", site_name="Test Hospital")
        reg = participant.create_registration()
        assert reg.site_id == "test-site"
        assert reg.site_name == "Test Hospital"

    def test_status_dict(self):
        """Status should include key info."""
        from src.federated.participant import FLParticipant

        participant = FLParticipant(site_id="test-site")
        status = participant.get_status()
        assert status["site_id"] == "test-site"
        assert status["rounds_completed"] == 0
        assert status["dp_epsilon_spent"] == 0.0


# ═════════════════════════════════════════════════════════════════════════
# BASE TRAINER FL EXTENSION TESTS
# ═════════════════════════════════════════════════════════════════════════


try:
    import mlflow  # noqa: F401
    _has_mlflow = True
except ImportError:
    _has_mlflow = False


@pytest.mark.skipif(not _has_mlflow, reason="mlflow not installed")
class TestBaseTrainerFL:
    """Tests for get_state_dict and set_state_dict in BaseTrainer."""

    def test_pytorch_get_state_dict(self):
        """PyTorch model state export should work."""
        from src.modeling.base_trainer import BaseTrainer

        class DummyTrainer(BaseTrainer):
            def _build_model(self, **kwargs):
                return SimpleModel()
            def _train_impl(self, train_data, val_data, **kwargs):
                return {}
            def _predict_proba_impl(self, data):
                return np.array([0.5])

        trainer = DummyTrainer("test", use_mlflow=False)
        trainer.model = SimpleModel()
        trainer.is_trained = True

        sd = trainer.get_state_dict()
        assert isinstance(sd, dict)
        assert len(sd) > 0
        # Ensure all tensors are on CPU
        for v in sd.values():
            assert v.device == torch.device("cpu")

    def test_pytorch_set_state_dict(self):
        """PyTorch model state import should work."""
        from src.modeling.base_trainer import BaseTrainer

        class DummyTrainer(BaseTrainer):
            def _build_model(self, **kwargs):
                return SimpleModel()
            def _train_impl(self, train_data, val_data, **kwargs):
                return {}
            def _predict_proba_impl(self, data):
                return np.array([0.5])

        trainer = DummyTrainer("test", use_mlflow=False)
        trainer.model = SimpleModel()
        trainer.is_trained = True

        # Get original state, modify, set back
        original_sd = trainer.get_state_dict()
        modified_sd = {k: v + 1.0 for k, v in original_sd.items() if v.is_floating_point()}
        trainer.set_state_dict({**original_sd, **modified_sd})

    def test_untrained_model_raises(self):
        """get_state_dict on untrained model should raise."""
        from src.modeling.base_trainer import BaseTrainer

        class DummyTrainer(BaseTrainer):
            def _build_model(self, **kwargs):
                return SimpleModel()
            def _train_impl(self, train_data, val_data, **kwargs):
                return {}
            def _predict_proba_impl(self, data):
                return np.array([0.5])

        trainer = DummyTrainer("test", use_mlflow=False)
        with pytest.raises(RuntimeError, match="must be trained"):
            trainer.get_state_dict()


# ═════════════════════════════════════════════════════════════════════════
# FEDERATED SETTINGS TESTS
# ═════════════════════════════════════════════════════════════════════════


class TestFederatedSettings:
    """Tests for FederatedSettings configuration."""

    def test_defaults(self):
        """Default settings should have FL disabled."""
        settings = FederatedSettings()
        assert settings.fl_enabled is False
        assert settings.fl_role == "standalone"
        assert settings.aggregation_strategy == "fedavg"
        assert settings.dp_enabled is True
        assert settings.dp_epsilon == 8.0

    def test_env_prefix(self):
        """Settings should use JEEVAN_FL_ prefix."""
        assert FederatedSettings.model_config["env_prefix"] == "JEEVAN_FL_"


# ═════════════════════════════════════════════════════════════════════════
# END-TO-END SIMULATION
# ═════════════════════════════════════════════════════════════════════════


class TestEndToEndSimulation:
    """Simulate a complete FL training session with 3 sites."""

    def test_three_site_two_rounds(self):
        """Run 2 FL rounds with 3 hospital sites."""
        from src.federated.coordinator import FLCoordinator

        # Create coordinator
        coord = FLCoordinator(
            total_rounds=2,
            min_participants=3,
            aggregation_strategy="fedavg",
        )

        # Register 3 sites
        for i in range(3):
            coord.register_site(SiteRegistration(
                site_id=f"hospital-{i+1}",
                site_name=f"Hospital {i+1}",
                dataset_size=1000 * (i + 1),
                model_types=["bilstm"],
            ))

        assert coord.ready_to_start

        # Set initial model
        init_model = SimpleModel()
        coord.set_initial_model("bilstm", init_model.state_dict())

        # Run 2 rounds
        for round_num in range(1, 3):
            config = coord.start_round()
            assert config.round_number == round_num

            # Each site submits an update
            for i in range(3):
                site_model = SimpleModel()
                # Simulate training by slightly modifying weights
                with torch.no_grad():
                    for p in site_model.parameters():
                        p.add_(torch.randn_like(p) * 0.01)

                sd_bytes = serialize_state_dict(site_model.state_dict())
                coord.submit_update(ModelUpdate(
                    site_id=f"hospital-{i+1}",
                    round_number=round_num,
                    model_type="bilstm",
                    state_dict_bytes=sd_bytes,
                    dataset_size=1000 * (i + 1),
                    training_time_s=30.0 * (i + 1),
                    local_metrics={"auroc": 0.80 + i * 0.02},
                ))

            assert coord.round_complete
            results = coord.aggregate_round()
            assert "bilstm" in results
            assert len(results["bilstm"].participating_sites) == 3

        assert coord.is_training_complete
        assert coord.current_round == 2
        assert len(coord.completed_rounds) == 2

        # Verify session status
        status = coord.get_session_status()
        assert status.current_round == 2
        assert len(status.completed_rounds) == 2
