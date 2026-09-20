"""Prepare independent observations and run reliability estimation when identified."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from code.data.loaders import DAY_COLUMNS, ROUTABLE_CLASSES
from code.reliability.model import (
    SPECIFICATIONS,
    ReliabilityFit,
    fit_reliability_model,
    half_life,
)

LOGGER = logging.getLogger(__name__)
ARTERIAL_CLASSES = frozenset(
    {
        "trunk",
        "trunk_link",
        "primary",
        "primary_link",
        "secondary",
        "secondary_link",
        "tertiary",
        "tertiary_link",
    }
)


@dataclass(frozen=True)
class EstimationStatus:
    """Describe whether independent closure coverage supports estimation."""

    mode: str
    covered_arcs: int
    routable_arcs: int
    coverage_fraction: float
    source_url: str
    reason: str


def prepare_exposure(roads: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Select dispatch-routable features and return exposure with class groups."""
    names = roads["fclass"].astype("string").str.lower()
    routable = roads.loc[names.isin(ROUTABLE_CLASSES)].copy()
    routable_names = routable["fclass"].astype("string").str.lower()
    class_index = (~routable_names.isin(ARTERIAL_CLASSES)).to_numpy(dtype=int)
    exposure = routable.loc[:, DAY_COLUMNS].to_numpy(dtype=float)
    if exposure.shape != (1297, 6):
        raise ValueError(f"Expected 1,297 routable arcs by 6 days, observed {exposure.shape}")
    counts = np.bincount(class_index, minlength=2)
    if counts.tolist() != [138, 1159]:
        raise ValueError(f"Expected class counts [138, 1159], observed {counts.tolist()}")
    return routable, exposure, class_index


def load_independent_closures(path: Path) -> pd.DataFrame:
    """Load arc-day closure observations that are independent of fire exposure."""
    if not path.exists():
        raise FileNotFoundError(
            f"Independent closure file is unavailable: {path}. Reliability parameters "
            "must remain scenario inputs."
        )
    frame = pd.read_csv(path)
    required = {"osm_id", "day", "closed"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Independent closure file is missing columns: {missing}")
    if not set(frame["closed"].dropna().unique()).issubset({0, 1}):
        raise ValueError("Independent closure observations must be binary")
    return frame


def join_independent_closures(
    routable: pd.DataFrame, closures: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Join closure observations to exposure arcs and return covered row positions."""
    if "osm_id" not in routable.columns:
        return np.empty((0, len(DAY_COLUMNS))), np.array([], dtype=int)
    wide = closures.pivot_table(index="osm_id", columns="day", values="closed")
    wide.columns = wide.columns.astype(str)
    available_days = [day for day in DAY_COLUMNS if day in wide.columns]
    if available_days != list(DAY_COLUMNS):
        return np.empty((0, len(DAY_COLUMNS))), np.array([], dtype=int)
    joined = routable[["osm_id"]].join(wide.loc[:, DAY_COLUMNS], on="osm_id")
    covered = joined.loc[:, DAY_COLUMNS].notna().all(axis=1).to_numpy()
    return joined.loc[covered, DAY_COLUMNS].to_numpy(dtype=float), np.flatnonzero(covered)


def fit_all_specifications(
    exposure: np.ndarray,
    closures: np.ndarray,
    class_index: np.ndarray,
    config: dict[str, Any],
) -> list[ReliabilityFit]:
    """Fit all candidate models when independent observations are available."""
    fits: list[ReliabilityFit] = []
    for offset, specification in enumerate(SPECIFICATIONS):
        fit = fit_reliability_model(
            exposure,
            closures,
            class_index,
            specification,
            train_day_count=len(config["train_days"]),
            epsilon=config["parameter_epsilon"],
            tolerance=config["optimizer_tolerance"],
            restarts=config["optimizer_restarts"],
            seed=config["bootstrap_seed"] + offset,
        )
        if not fit.converged:
            raise RuntimeError(f"{specification} fit failed: {fit.optimizer_message}")
        if not 0.0 <= fit.phi_min <= 1.0:
            raise RuntimeError(f"{specification} produced invalid phi_min {fit.phi_min}")
        fits.append(fit)
    return fits


def write_scenario_fallback(
    roads: pd.DataFrame,
    config: dict[str, Any],
    raw_data_dir: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write a loud non-estimation result when independent closures are unavailable."""
    routable, _, _ = prepare_exposure(roads)
    closure_path = raw_data_dir / config["closure_local_file"]
    covered_arcs = 0
    reason = (
        "The specified Caltrans catalog URL returns 404, no matching package is in the "
        "current catalog, no local historical closure file exists, and the exposure "
        "export has no osm_id for an arc-level join."
    )
    if closure_path.exists():
        closures = load_independent_closures(closure_path)
        _, covered = join_independent_closures(routable, closures)
        covered_arcs = int(covered.size)
        reason = "Independent closure coverage is too thin for estimation."
    status = EstimationStatus(
        mode="swept_scenario_parameters",
        covered_arcs=covered_arcs,
        routable_arcs=len(routable),
        coverage_fraction=covered_arcs / len(routable),
        source_url=config["closure_source_url"],
        reason=reason,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "reliability_estimation_status.json"
    status_path.write_text(json.dumps(asdict(status), indent=2), encoding="utf-8")
    grid = pd.MultiIndex.from_product(
        [config["scenario_alpha"], config["scenario_gamma"]],
        names=["alpha", "gamma"],
    ).to_frame(index=False)
    grid["half_life_days"] = half_life(grid["gamma"].to_numpy())
    grid["status"] = "scenario_parameter_not_estimate"
    grid_path = output_dir / "reliability_scenario_grid.csv"
    grid.to_csv(grid_path, index=False)
    LOGGER.warning("Reliability estimation unavailable: %s", reason)
    return status_path, grid_path
