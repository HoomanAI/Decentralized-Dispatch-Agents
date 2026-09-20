"""Discounted Beta posterior updates with optional corridor pooling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BetaBelief:
    """Maintain per-edge passability beliefs and their shared priors."""

    a: np.ndarray
    b: np.ndarray
    a0: np.ndarray
    b0: np.ndarray
    discount: float
    pooling: float

    @classmethod
    def from_reliability(
        cls,
        reliability: np.ndarray,
        prior_strength: float,
        discount: float,
        pooling: float,
    ) -> "BetaBelief":
        """Initialize a numerically valid Beta prior from daily reliability."""
        epsilon = 1.0e-6
        mean = np.clip(np.asarray(reliability, dtype=float), epsilon, 1.0 - epsilon)
        a0 = prior_strength * mean
        b0 = prior_strength * (1.0 - mean)
        return cls(a0.copy(), b0.copy(), a0, b0, discount, pooling)

    @property
    def mean(self) -> np.ndarray:
        """Return posterior mean passability."""
        return self.a / (self.a + self.b)

    def update(
        self,
        traversed_index: int,
        passed: bool,
        peers: np.ndarray | None = None,
    ) -> None:
        """Discount all arcs, update one observation, and pool weakly to its peers."""
        observed = np.zeros_like(self.a, dtype=bool)
        observed[traversed_index] = True
        self.a[~observed] = (
            self.discount * self.a[~observed]
            + (1.0 - self.discount) * self.a0[~observed]
        )
        self.b[~observed] = (
            self.discount * self.b[~observed]
            + (1.0 - self.discount) * self.b0[~observed]
        )
        self.a[traversed_index] = self.discount * self.a[traversed_index] + float(passed)
        self.b[traversed_index] = self.discount * self.b[traversed_index] + float(not passed)
        if peers is not None and self.pooling > 0.0:
            peer_index = np.asarray(peers, dtype=int)
            peer_index = peer_index[peer_index != traversed_index]
            self.a[peer_index] += self.pooling * float(passed)
            self.b[peer_index] += self.pooling * float(not passed)

