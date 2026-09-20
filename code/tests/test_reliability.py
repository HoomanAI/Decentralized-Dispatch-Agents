from __future__ import annotations

import numpy as np

from code.reliability.model import (
    compute_damage,
    fit_reliability_model,
    fleet_reliability,
    half_life,
)


def test_recurrence_matches_unrolled_form_before_cap() -> None:
    exposure = np.array([[1.0, 0.5, 0.0, 0.25]])
    alpha = np.array([0.2])
    gamma = np.array([0.3])
    observed = compute_damage(exposure, np.array([0]), alpha, gamma)[0]
    expected = np.array(
        [
            alpha[0]
            * sum((1.0 - gamma[0]) ** k * exposure[0, day - k] for k in range(day + 1))
            for day in range(exposure.shape[1])
        ]
    )
    np.testing.assert_allclose(observed, expected)


def test_half_life_and_crewed_fleet_identity() -> None:
    assert half_life(0.5) == np.log(2.0) / -np.log(0.5)
    damage = np.array([0.0, 0.2, 0.8, 1.0])
    np.testing.assert_allclose(fleet_reliability(damage, 0.0), 1.0 - damage)


def test_estimator_enforces_normalization_and_parameter_bounds() -> None:
    observed = np.array(
        [
            [1, 1, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 1],
            [1, 0, 0, 0, 0, 1],
        ],
        dtype=float,
    )
    classes = np.array([0, 0, 1, 1])
    fit = fit_reliability_model(
        observed,
        1.0 - observed,
        classes,
        "full",
        restarts=2,
        seed=23,
        tolerance=1.0e-8,
    )
    assert fit.converged
    assert fit.kappa_1 > 0.0
    assert np.all((fit.alpha > 0.0) & (fit.alpha <= 1.0))
    assert np.all((fit.gamma > 0.0) & (fit.gamma <= 1.0))
    assert np.isfinite(fit.held_out_log_likelihood)
    assert 0.0 <= fit.phi_min <= 1.0


def test_estimator_rejects_reused_exposure_shape_mismatch() -> None:
    exposure = np.zeros((3, 6))
    closures = np.zeros((2, 6))
    classes = np.zeros(3, dtype=int)
    try:
        fit_reliability_model(exposure, closures, classes, "pooled")
    except ValueError as error:
        assert "matching shapes" in str(error)
    else:
        raise AssertionError("Mismatched observation streams must fail")
