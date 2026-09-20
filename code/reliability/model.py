"""Deterministic latent damage model and constrained likelihood estimator.

The latent state is deterministic once the damage parameters and exposure history are
given. No filtering, smoothing, or expectation maximization is required. Exposure and
closure must be separate observation streams. Kappa_1 is fitted, while the operational
threshold is parameterized directly inside its valid unit interval.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, xlog1py, xlogy

Specification = Literal["full", "memoryless", "pooled", "linear"]
SPECIFICATIONS: tuple[Specification, ...] = (
    "full",
    "memoryless",
    "pooled",
    "linear",
)
MAX_EMISSION_SLOPE = 100.0


@dataclass(frozen=True)
class ReliabilityFit:
    """Contain an estimated specification and its predictive score."""

    specification: Specification
    alpha: np.ndarray
    gamma: np.ndarray
    kappa_0: float
    kappa_1: float
    train_log_likelihood: float
    held_out_log_likelihood: float
    converged: bool
    optimizer_message: str

    @property
    def phi_min(self) -> float:
        """Return the emission midpoint as the operational threshold."""
        threshold = 1.0 + self.kappa_0 / self.kappa_1
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Invalid operational threshold: {threshold}")
        return threshold


def half_life(gamma: np.ndarray | float) -> np.ndarray:
    """Convert geometric clearance rates to reporting-scale half lives in days."""
    values = np.asarray(gamma, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.log(2.0) / -np.log1p(-values)
    return np.where(values >= 1.0, 0.0, result)


def compute_damage(
    exposure: np.ndarray,
    class_index: np.ndarray,
    alpha: np.ndarray,
    gamma: np.ndarray,
    specification: Specification = "full",
) -> np.ndarray:
    """Compute daily latent damage for arcs by deterministic recurrence."""
    exposure = np.asarray(exposure, dtype=float)
    class_index = np.asarray(class_index, dtype=int)
    if exposure.ndim != 2 or exposure.shape[0] != class_index.size:
        raise ValueError("Exposure must have shape [arcs, days]")
    damage = np.zeros_like(exposure, dtype=float)
    previous = np.zeros(exposure.shape[0], dtype=float)
    arc_alpha = np.asarray(alpha, dtype=float)[class_index]
    arc_gamma = np.asarray(gamma, dtype=float)[class_index]
    for day in range(exposure.shape[1]):
        if specification == "linear":
            current = np.maximum(0.0, arc_alpha * exposure[:, day] + previous - arc_gamma)
        else:
            current = np.minimum(
                1.0,
                arc_alpha * exposure[:, day] + (1.0 - arc_gamma) * previous,
            )
        damage[:, day] = current
        previous = current
    return damage


def fleet_reliability(damage: np.ndarray, upsilon: np.ndarray | float) -> np.ndarray:
    """Return effective fleet reliability over the operational design domain."""
    damage = np.asarray(damage, dtype=float)
    sensitivity = np.asarray(upsilon, dtype=float)
    return (1.0 - damage) * (1.0 - sensitivity * damage)


def _unpack_parameters(
    values: np.ndarray, specification: Specification
) -> tuple[np.ndarray, np.ndarray, float, float]:
    if specification in {"full", "linear"}:
        alpha = values[0:2]
        gamma = values[2:4]
        phi_min, kappa_1 = values[4:6]
    elif specification == "memoryless":
        alpha = values[0:2]
        gamma = np.ones(2)
        phi_min, kappa_1 = values[2:4]
    elif specification == "pooled":
        alpha = np.repeat(values[0], 2)
        gamma = np.repeat(values[1], 2)
        phi_min, kappa_1 = values[2:4]
    else:
        raise ValueError(f"Unknown specification: {specification}")
    kappa_0 = float(kappa_1 * (phi_min - 1.0))
    return alpha, gamma, kappa_0, float(kappa_1)


def _initial_values(specification: Specification) -> np.ndarray:
    if specification in {"full", "linear"}:
        return np.array([0.75, 0.75, 0.5, 0.5, 0.5, 2.0])
    if specification == "memoryless":
        return np.array([0.75, 0.75, 0.5, 2.0])
    return np.array([0.75, 0.5, 0.5, 2.0])


def _bounds(specification: Specification, epsilon: float) -> list[tuple[float, float]]:
    unit = (epsilon, 1.0)
    threshold = (0.0, 1.0)
    slope = (epsilon, MAX_EMISSION_SLOPE)
    if specification in {"full", "linear"}:
        return [unit, unit, unit, unit, threshold, slope]
    return [unit, unit, threshold, slope]


def _log_likelihood(
    observed: np.ndarray, damage: np.ndarray, kappa_0: float, kappa_1: float
) -> float:
    probability = expit(kappa_0 + kappa_1 * damage)
    return float(
        np.sum(xlogy(observed, probability) + xlog1py(1.0 - observed, -probability))
    )


def fit_reliability_model(
    exposure: np.ndarray,
    observed_closure: np.ndarray,
    class_index: np.ndarray,
    specification: Specification,
    train_day_count: int = 5,
    epsilon: float = 1.0e-6,
    tolerance: float = 1.0e-9,
    restarts: int = 4,
    seed: int = 0,
    initial: np.ndarray | None = None,
) -> ReliabilityFit:
    """Fit one specification using distinct exposure and closure observations."""
    exposure = np.asarray(exposure, dtype=float)
    observed = np.asarray(observed_closure, dtype=float)
    if exposure.shape != observed.shape:
        raise ValueError("Exposure and closure observations must have matching shapes")
    if observed.shape[1] <= train_day_count:
        raise ValueError("A held-out day is required")

    def objective(values: np.ndarray) -> float:
        alpha, gamma, kappa_0, kappa_1 = _unpack_parameters(values, specification)
        damage = compute_damage(
            exposure[:, :train_day_count], class_index, alpha, gamma, specification
        )
        return -_log_likelihood(
            observed[:, :train_day_count], damage, kappa_0, kappa_1
        )

    rng = np.random.default_rng(seed)
    starts = [_initial_values(specification) if initial is None else np.asarray(initial)]
    for _ in range(max(0, restarts - 1)):
        candidate = starts[0].copy()
        candidate[:-2] = rng.uniform(0.1, 0.9, size=candidate.size - 2)
        candidate[-2] = rng.uniform(0.1, 0.9)
        candidate[-1] = rng.uniform(0.5, 5.0)
        starts.append(candidate)
    solutions = [
        minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=_bounds(specification, epsilon),
            options={"ftol": tolerance, "maxiter": 2000},
        )
        for start in starts
    ]
    result = min(solutions, key=lambda item: item.fun)
    alpha, gamma, kappa_0, kappa_1 = _unpack_parameters(result.x, specification)
    damage = compute_damage(exposure, class_index, alpha, gamma, specification)
    train_ll = _log_likelihood(
        observed[:, :train_day_count],
        damage[:, :train_day_count],
        kappa_0,
        kappa_1,
    )
    held_out_ll = _log_likelihood(
        observed[:, train_day_count:],
        damage[:, train_day_count:],
        kappa_0,
        kappa_1,
    )
    fit = ReliabilityFit(
        specification=specification,
        alpha=alpha,
        gamma=gamma,
        kappa_0=kappa_0,
        kappa_1=kappa_1,
        train_log_likelihood=train_ll,
        held_out_log_likelihood=held_out_ll,
        converged=bool(result.success),
        optimizer_message=str(result.message),
    )
    if not 0.0 <= fit.phi_min <= 1.0:
        raise ValueError(f"Invalid operational threshold: {fit.phi_min}")
    return fit
