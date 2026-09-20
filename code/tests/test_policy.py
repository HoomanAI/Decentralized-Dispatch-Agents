from __future__ import annotations

import numpy as np
import torch

from code.policy.mappo import generalized_advantage_estimate
from code.policy.model import ExploitationActorCritic


def test_shared_actor_is_vehicle_permutation_equivariant() -> None:
    torch.manual_seed(7)
    model = ExploitationActorCritic(3, 5, 6, 6, 8)
    state = {
        "node_features": torch.rand(4, 3),
        "edge_index": torch.tensor([[0, 1, 2], [1, 2, 3]]),
        "edge_features": torch.rand(3, 5),
        "vehicle_features": torch.rand(3, 6),
        "vehicle_nodes": torch.tensor([0, 1, 2]),
        "patient_features": torch.rand(1, 6),
        "patient_node": torch.tensor([3]),
        "travel_bias": torch.tensor([-0.1, -0.2, -0.3]),
    }
    logits, value = model(**state)
    permutation = torch.tensor([2, 0, 1])
    permuted = dict(state)
    permuted["vehicle_features"] = state["vehicle_features"][permutation]
    permuted["vehicle_nodes"] = state["vehicle_nodes"][permutation]
    permuted["travel_bias"] = state["travel_bias"][permutation]
    permuted_logits, permuted_value = model(**permuted)
    assert torch.allclose(permuted_logits, logits[permutation], atol=1.0e-6)
    assert torch.allclose(permuted_value, value, atol=1.0e-6)


def test_gae_respects_episode_boundary() -> None:
    advantages, returns = generalized_advantage_estimate(
        rewards=np.array([1.0, 2.0]),
        values=np.array([0.5, 0.25]),
        terminals=np.array([False, True]),
        discount=0.9,
        gae_lambda=1.0,
    )
    assert np.allclose(advantages, [2.3, 1.75])
    assert np.allclose(returns, [2.8, 2.0])
