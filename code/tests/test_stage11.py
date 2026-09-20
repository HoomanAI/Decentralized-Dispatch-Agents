"""Cheap gates for Stage 11 lookahead state and greedy-residual initialization."""

from __future__ import annotations

import networkx as nx
import numpy as np
import torch

from code.belief.beta import BetaBelief
from code.env.vehicle_centric import LOOKAHEAD_FEATURE_NAMES, _lookahead_features
from code.policy.vehicle_centric import LookaheadResidualPolicy, RouteActorQCritic


def test_lookahead_residual_initializes_exactly_to_greedy_logits() -> None:
    model = LookaheadResidualPolicy(hidden=8, max_vehicles=4)
    state = {
        "vehicle_index": torch.tensor(0),
        "vehicle_features": torch.tensor([1.0, 0.0, 0.0, 0.0]),
        "candidate_features": torch.tensor(
            [[1.0, 0.0, 0.0, 0.1, 1.5, 0.2, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
        ),
        "arrival_bias": torch.tensor([1.5, 0.0]),
        "lookahead_features": torch.zeros(len(LOOKAHEAD_FEATURE_NAMES)),
    }
    logits, _ = model(state)
    assert torch.equal(logits, state["arrival_bias"])


def test_lookahead_features_do_not_read_future_exposure() -> None:
    graph = nx.Graph()
    graph.add_edge(0, 1, edge_id=0, class_group="arterial")
    graph.add_edge(1, 2, edge_id=1, class_group="local")
    graph.graph["edge_day_flags"] = {
        0: {str(day): int(day == 7) for day in range(7, 13)},
        1: {str(day): 0 for day in range(7, 13)},
    }
    belief = BetaBelief.from_reliability(np.array([0.8, 0.8]), 8.0, 0.95, 0.15)
    vehicle = {"node": 0, "available": 0.0}
    vehicles = [vehicle]
    before = _lookahead_features(
        graph, graph, vehicle, vehicles, [], belief, {0: 0, 1: 1}, 8, 0.0
    )
    graph.graph["edge_day_flags"][0]["12"] = 1
    graph.graph["edge_day_flags"][1]["12"] = 1
    after = _lookahead_features(
        graph, graph, vehicle, vehicles, [], belief, {0: 0, 1: 1}, 8, 0.0
    )
    assert len(before) == len(LOOKAHEAD_FEATURE_NAMES) == 16
    assert np.array_equal(before, after)


def test_route_policy_initializes_to_no_exploration() -> None:
    model = RouteActorQCritic(hidden=8, max_vehicles=12)
    state = {
        "vehicle_index": torch.tensor(7),
        "route_features": torch.zeros((4, 11)),
        "route_rank": torch.tensor([0, 1, 2, 3]),
    }
    logits, _ = model(state)
    assert int(torch.argmax(logits).item()) == 0
    assert float(logits[0].item()) == 0.0
    assert torch.equal(logits[1:], torch.full((3,), -4.0))
