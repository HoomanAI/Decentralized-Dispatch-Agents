"""Stage 9 comparison policies adapted to the common dispatch interface.

These are problem-level reimplementations, not reproductions of the authors' neural
architectures. Yan retains event-driven waiting-time dispatch, Ahmadi retains
survival-aware deadline-masked dispatch, and Peng retains decentralized masked action
selection. GA and ALNS optimize a shared linear dispatch score on calibration seeds.
All methods use the simulator's fixed collision rule and receive the same belief state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


FEATURE_NAMES = (
    "weighted_survival",
    "waiting_time",
    "negative_arrival_time",
    "negative_route_distance",
    "belief_information",
)


def candidate_features(context: dict[str, Any], index: int) -> np.ndarray:
    """Return stable, set-normalized features for one feasible action."""
    candidate = context["candidates"][index]
    if candidate.patient is None:
        return np.zeros(len(FEATURE_NAMES), dtype=float)
    now = float(context["now"])
    waiting = max(now - float(candidate.patient["onset_minute"]), 0.0)
    distance = float(candidate.inbound[2]) + float(candidate.outbound[2])
    raw = np.asarray(
        [
            float(candidate.immediate_survival),
            waiting / 720.0,
            -max(float(candidate.estimated_arrival) - now, 0.0) / 120.0,
            -distance / 50.0,
            float(candidate.information_value),
        ],
        dtype=float,
    )
    return raw


@dataclass
class LinearDispatchPolicy:
    """Masked per-vehicle dispatch rule with a fixed linear action score."""

    weights: np.ndarray

    def propose(self, context: dict[str, Any]) -> tuple[int, float]:
        allowed = context["allowed_indices"]
        scores = {i: float(self.weights @ candidate_features(context, i)) for i in allowed}
        choice = max(allowed, key=lambda i: (scores[i], -i))
        return choice, scores[choice]

    def observe(self, reward: float, terminal: bool = False) -> None:
        del reward, terminal


def weights_for(method: str) -> np.ndarray:
    """Return the stated adaptation for each published comparison."""
    if method == "yan":
        # Event-driven response-time objective with waiting-time priority.
        return np.asarray([0.0, 2.0, 1.0, 0.0, 0.0])
    if method == "ahmadi":
        # Survival-aware masked dispatch. Feasibility already enforces deadlines.
        return np.asarray([1.0, 0.25, 0.0, 0.0, 0.0])
    if method == "peng":
        # Decentralized invalid-action masking with travel-efficiency preference.
        return np.asarray([0.75, 0.0, 0.5, 0.25, 0.0])
    raise ValueError(f"Unknown Stage 9 method: {method}")


def make_policies(method: str, vehicle_count: int, weights: np.ndarray | None = None) -> list[LinearDispatchPolicy]:
    """Create independent policy objects with a shared deterministic score."""
    selected = weights_for(method) if weights is None else np.asarray(weights, dtype=float)
    return [LinearDispatchPolicy(selected.copy()) for _ in range(vehicle_count)]
