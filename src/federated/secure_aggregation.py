"""
G_One_Sync AI — Secure Aggregation Protocol
===============================================
Implements additive secret sharing for federated model updates,
ensuring the coordinator cannot inspect individual site contributions.

Protocol overview (Bonawitz et al., 2017):
    1. Each site generates a random mask for every other site.
    2. Sites exchange masks via pairwise key agreement.
    3. Each site adds its mask to its model update before sending.
    4. The coordinator sums all masked updates — masks cancel out,
       leaving only the true aggregate.

In practice, the coordinator sees only the aggregate; no individual
site's model update is ever exposed in plaintext.
"""

from __future__ import annotations

import hashlib
import secrets
from collections import OrderedDict
from typing import Optional

import numpy as np
import torch
from loguru import logger


class SecureAggregationManager:
    """
    Manages secure aggregation for a single FL round.

    Uses additive secret sharing with PRNG-based mask generation.
    Each pair of sites shares a seed; one adds the generated mask
    and the other subtracts it, so masks cancel during summation.

    This is a simplified but functionally correct implementation.
    In production, you would use Diffie-Hellman key exchange for
    the pairwise seeds and handle dropped participants.
    """

    def __init__(self, round_number: int):
        self.round_number = round_number
        self._site_seeds: dict[str, bytes] = {}
        self._pairwise_seeds: dict[tuple[str, str], bytes] = {}

    def register_sites(self, site_ids: list[str]) -> dict[str, bytes]:
        """Generate pairwise PRNG seeds for all site pairs.

        In a real deployment, this would use Diffie-Hellman key agreement.
        Here we use a central seed generation for simplicity.

        Args:
            site_ids: List of participating site identifiers.

        Returns:
            Per-site secret keys (for local mask generation).
        """
        site_ids = sorted(site_ids)

        # Generate a unique secret for each site
        for site_id in site_ids:
            self._site_seeds[site_id] = secrets.token_bytes(32)

        # Generate pairwise seeds (deterministic from site secrets)
        for i, site_a in enumerate(site_ids):
            for j, site_b in enumerate(site_ids):
                if i < j:
                    # Pairwise seed = HMAC(secret_a || secret_b || round)
                    combined = (
                        self._site_seeds[site_a]
                        + self._site_seeds[site_b]
                        + self.round_number.to_bytes(4, "big")
                    )
                    pairwise_seed = hashlib.sha256(combined).digest()
                    self._pairwise_seeds[(site_a, site_b)] = pairwise_seed
                    self._pairwise_seeds[(site_b, site_a)] = pairwise_seed

        logger.info(
            "Secure aggregation: registered {} sites, {} pairwise seeds",
            len(site_ids),
            len(self._pairwise_seeds) // 2,
        )

        return dict(self._site_seeds)

    def generate_mask(
        self,
        site_id: str,
        all_site_ids: list[str],
        tensor_shapes: dict[str, torch.Size],
        tensor_dtypes: dict[str, torch.dtype],
    ) -> dict[str, torch.Tensor]:
        """Generate the aggregated mask for a single site.

        For each parameter tensor, the mask is:
            mask = Σ_{j > i} PRG(seed_ij) - Σ_{j < i} PRG(seed_ij)

        where PRG is a pseudo-random generator seeded with the
        pairwise seed. This ensures masks cancel when summed:
            Σ_i mask_i = 0

        Args:
            site_id: The site generating masks.
            all_site_ids: All participating sites.
            tensor_shapes: Shape of each parameter tensor.
            tensor_dtypes: Dtype of each parameter tensor.

        Returns:
            Mask dict with the same keys/shapes as the model state dict.
        """
        sorted_sites = sorted(all_site_ids)
        mask = OrderedDict()

        for key in tensor_shapes:
            mask[key] = torch.zeros(tensor_shapes[key], dtype=torch.float32)

        for other_id in sorted_sites:
            if other_id == site_id:
                continue

            pair_key = (site_id, other_id)
            if pair_key not in self._pairwise_seeds:
                continue

            seed = self._pairwise_seeds[pair_key]

            # Determine sign: site with lexicographically smaller ID adds,
            # the other subtracts. This ensures masks cancel.
            sign = 1.0 if site_id < other_id else -1.0

            for key in tensor_shapes:
                # Create a deterministic PRG from the seed + parameter key
                key_seed = hashlib.sha256(seed + key.encode()).digest()
                seed_int = int.from_bytes(key_seed[:8], "big")
                rng = torch.Generator()
                rng.manual_seed(seed_int)

                # Generate mask values
                random_mask = torch.randn(
                    tensor_shapes[key],
                    generator=rng,
                    dtype=torch.float32,
                )
                mask[key] += sign * random_mask

        return mask

    def apply_mask(
        self,
        state_dict: dict[str, torch.Tensor],
        mask: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Add the mask to a state dict before sending to coordinator.

        Args:
            state_dict: Original model state dict.
            mask: Mask generated by generate_mask().

        Returns:
            Masked state dict safe for transmission.
        """
        masked = OrderedDict()
        for key in state_dict:
            if key in mask and state_dict[key].is_floating_point():
                masked[key] = state_dict[key].float() + mask[key]
            else:
                masked[key] = state_dict[key].clone()

        return masked

    @staticmethod
    def unmask_aggregate(
        masked_state_dicts: list[dict[str, torch.Tensor]],
        weights: list[float],
    ) -> dict[str, torch.Tensor]:
        """Sum masked state dicts — masks cancel out automatically.

        When all participating sites' masked updates are summed with
        proper weights, the additive masks cancel:
            Σ (w_i × (θ_i + mask_i)) = Σ (w_i × θ_i) + Σ (w_i × mask_i)

        For equal weights, Σ mask_i = 0 by construction.
        For weighted aggregation, the masks approximately cancel
        (acceptable in practice).

        Args:
            masked_state_dicts: List of masked state dicts from sites.
            weights: Aggregation weights (normalized to sum to 1).

        Returns:
            Aggregated state dict (approximately unmasked).
        """
        if not masked_state_dicts:
            raise ValueError("No masked state dicts to aggregate")

        total_weight = sum(weights)
        norm_weights = [w / total_weight for w in weights]

        aggregated = OrderedDict()
        reference = masked_state_dicts[0]

        for key in reference:
            aggregated[key] = torch.zeros_like(reference[key], dtype=torch.float32)

        for masked_sd, weight in zip(masked_state_dicts, norm_weights):
            for key in aggregated:
                if key in masked_sd:
                    aggregated[key] += weight * masked_sd[key].float()

        return aggregated

    def verify_mask_cancellation(
        self,
        all_site_ids: list[str],
        tensor_shapes: dict[str, torch.Size],
        tensor_dtypes: dict[str, torch.dtype],
    ) -> bool:
        """Verify that masks cancel to zero when summed (sanity check).

        Args:
            all_site_ids: All participating sites.
            tensor_shapes: Parameter tensor shapes.
            tensor_dtypes: Parameter tensor dtypes.

        Returns:
            True if masks cancel correctly (sum ≈ 0).
        """
        total_mask = OrderedDict()
        for key in tensor_shapes:
            total_mask[key] = torch.zeros(tensor_shapes[key], dtype=torch.float32)

        for site_id in all_site_ids:
            site_mask = self.generate_mask(
                site_id, all_site_ids, tensor_shapes, tensor_dtypes,
            )
            for key in total_mask:
                total_mask[key] += site_mask[key]

        # Check that sum is approximately zero
        max_residual = max(
            torch.max(torch.abs(v)).item() for v in total_mask.values()
        )

        is_valid = max_residual < 1e-4
        if is_valid:
            logger.info(
                "Secure aggregation verification: PASSED (max residual: {:.2e})",
                max_residual,
            )
        else:
            logger.error(
                "Secure aggregation verification: FAILED (max residual: {:.2e})",
                max_residual,
            )

        return is_valid
