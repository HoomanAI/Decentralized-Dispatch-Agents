"""Load the released inputs and make every known defect explicit.

The affected population workbook uses ``Total`` as a union footprint count. It is
not a sum across daily columns, which overlap spatially. The road export must expose
``osm_id`` before it can be attached to a routable graph. Its absence is recorded as
a synthetic-network requirement, but it does not prevent analysis of the real flags.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)
DAY_COLUMNS = tuple(str(day) for day in range(7, 13))
WATERWAY_CLASSES = frozenset({"stream", "drain"})
NON_ROAD_POI_CLASSES = frozenset({"christian_methodist"})
ROUTABLE_CLASSES = (
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "living_street",
    "residential",
    "service",
    "unclassified",
)
AGE_65_COLUMNS = (
    "Totalpopulation65to74years",
    "Totalpopulation75to84years",
    "Totalpopulation85yearsandover",
)


class InputContractError(ValueError):
    """Report a raw input that cannot be interpreted without guessing."""


@dataclass(frozen=True)
class InputBundle:
    """Hold cleaned tables, audit notes, and the network fallback decision."""

    roads: pd.DataFrame
    population: pd.DataFrame
    landcover_weights: pd.DataFrame
    affected_population: pd.DataFrame
    audit: dict[str, Any]
    requires_synthetic_network: bool


def _require_columns(frame: pd.DataFrame, required: set[str], source: Path) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise InputContractError(f"{source.name} is missing required columns: {missing}")


def load_roads(path: Path) -> tuple[pd.DataFrame, dict[str, Any], bool]:
    """Load road flags, remove waterways and spatial-join length artifacts."""
    frame = pd.read_csv(path)
    _require_columns(
        frame,
        {"Layer", "fclass", "Shape_Length", *DAY_COLUMNS},
        path,
    )
    if not set(frame.loc[:, DAY_COLUMNS].stack().unique()).issubset({0, 1}):
        raise InputContractError(f"{path.name} contains nonbinary daily flags")

    artifact_columns = [
        name for name in frame.columns if name.startswith("Shape_Length_")
    ]
    nonconstant = [name for name in artifact_columns if frame[name].nunique() > 1]
    if nonconstant:
        raise InputContractError(
            f"Spatial-join columns are no longer constant: {nonconstant}"
        )
    LOGGER.info(
        "Dropping six constant fire-perimeter boundary length columns: %s",
        artifact_columns,
    )

    class_names = frame["fclass"].astype("string").str.lower()
    waterway_mask = class_names.isin(WATERWAY_CLASSES)
    poi_mask = class_names.isin(NON_ROAD_POI_CLASSES)
    waterway_counts = class_names[waterway_mask].value_counts().to_dict()
    roads = frame.loc[~waterway_mask & ~poi_mask].drop(columns=artifact_columns).copy()
    LOGGER.info(
        "Filtered confirmed waterways by class: %d stream and %d drain features",
        waterway_counts.get("stream", 0),
        waterway_counts.get("drain", 0),
    )

    layer_groups = frame["Layer"].ffill().value_counts().to_dict()
    dropped_poi = frame.loc[poi_mask, "fclass"].astype(str).value_counts().to_dict()
    if dropped_poi:
        LOGGER.warning(
            "Dropped non-road point of interest rows from the waterway block: %s",
            dropped_poi,
        )

    requires_synthetic = "osm_id" not in roads.columns
    if requires_synthetic:
        LOGGER.warning(
            "Road input has no osm_id join key. Real flags cannot be attached to a "
            "routable graph, so network stages must use a visibly tagged synthetic input"
        )

    durations = frame.loc[:, DAY_COLUMNS].sum(axis=1).value_counts().sort_index()
    road_durations = roads.loc[:, DAY_COLUMNS].sum(axis=1).value_counts().sort_index()
    routable_mask = roads["fclass"].astype("string").str.lower().isin(ROUTABLE_CLASSES)
    missing_length_count = int(roads["Shape_Length"].isna().sum())
    audit = {
        "source_feature_count": int(len(frame)),
        "road_feature_count": int(len(roads)),
        "waterway_class_counts": {key: int(value) for key, value in waterway_counts.items()},
        "forward_filled_layer_counts": {
            str(key): int(value) for key, value in layer_groups.items()
        },
        "dropped_join_artifact_columns": artifact_columns,
        "exposure_duration_counts_all_features": {
            str(int(key)): int(value) for key, value in durations.items()
        },
        "exposure_duration_counts_roads_only": {
            str(int(key)): int(value) for key, value in road_durations.items()
        },
        "dropped_non_road_poi": {
            "count": int(poi_mask.sum()),
            "fclass_values": {key: int(value) for key, value in dropped_poi.items()},
            "reason": "Point of interest in the waterway block with no segment length",
        },
        "routable_feature_count": int(routable_mask.sum()),
        "routable_fclasses": list(ROUTABLE_CLASSES),
        "missing_shape_length_after_cleaning": missing_length_count,
        "road_total_shape_length_native_units": float(roads["Shape_Length"].sum()),
        "has_osm_id": not requires_synthetic,
    }
    return roads, audit, requires_synthetic


def load_population(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load block demographics and calculate the reported population checks."""
    frame = pd.read_csv(path)
    _require_columns(
        frame,
        {"Totalpopulation", "Totalhousingunits", *AGE_65_COLUMNS},
        path,
    )
    age_65 = frame.loc[:, AGE_65_COLUMNS].sum(axis=1)
    share = age_65 / frame["Totalpopulation"]
    audit = {
        "block_count": int(len(frame)),
        "total_population": int(frame["Totalpopulation"].sum()),
        "housing_units": int(frame["Totalhousingunits"].sum()),
        "population_65_plus": int(age_65.sum()),
        "population_65_plus_share": float(age_65.sum() / frame["Totalpopulation"].sum()),
        "block_age_65_share_min": float(share.min()),
        "block_age_65_share_max": float(share.max()),
    }
    return frame, audit


def load_landcover_weights(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the table whose header begins at row two and column two."""
    raw = pd.read_excel(path, header=None)
    header = raw.iloc[1, 1:3].tolist()
    expected_header = ["Land Cover Type", "Relative Weighted Value (RA)"]
    if header != expected_header:
        raise InputContractError(
            f"{path.name} offset header changed: observed {header}"
        )
    table = raw.iloc[2:, 1:3].copy()
    table.columns = ["land_cover_type", "relative_weight"]
    table = table.dropna(how="all").reset_index(drop=True)
    table["relative_weight"] = pd.to_numeric(table["relative_weight"], errors="raise")
    if table["land_cover_type"].isna().any() or (table["relative_weight"] < 0).any():
        raise InputContractError(f"{path.name} contains invalid land-cover weights")
    LOGGER.info("Read land-cover header from offset row two and column two")
    return table, {"landcover_class_count": int(len(table))}


def load_affected_population(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load daily totals and treat Total as a union footprint count, not a sum."""
    sheets = pd.read_excel(path, sheet_name=None)
    rows: list[pd.DataFrame] = []
    union_checks: dict[str, dict[str, float]] = {}
    for sheet_name, frame in sheets.items():
        if "Total" not in frame.columns:
            raise InputContractError(f"Sheet {sheet_name} has no Total column")
        long = frame.melt(id_vars="Total", var_name="day", value_name="affected_population")
        long.insert(0, "region", sheet_name)
        long["day"] = pd.to_numeric(long["day"], errors="raise").astype(int)
        daily_sum = float(long["affected_population"].sum())
        union_total = float(frame["Total"].iloc[0])
        if union_total >= daily_sum:
            raise InputContractError(
                f"Sheet {sheet_name} no longer supports the union-count interpretation"
            )
        union_checks[sheet_name] = {
            "union_total": union_total,
            "sum_across_days": daily_sum,
        }
        rows.append(long)
        LOGGER.info(
            "Treating %s Total %.3f as a union count, below daily sum %.3f",
            sheet_name,
            union_total,
            daily_sum,
        )
    return pd.concat(rows, ignore_index=True), {"union_count_checks": union_checks}


def load_all_inputs(raw_data_dir: Path) -> InputBundle:
    """Load all four released files and return a unified audit record."""
    roads, roads_audit, requires_synthetic = load_roads(
        raw_data_dir / "PDS_roads_daily_affected.csv"
    )
    population, population_audit = load_population(raw_data_dir / "pop_PDS.csv")
    weights, weights_audit = load_landcover_weights(
        raw_data_dir / "dasymetric_landcover_weights.xlsx"
    )
    affected, affected_audit = load_affected_population(
        raw_data_dir / "daily_affected_population_totals.xlsx"
    )
    return InputBundle(
        roads=roads,
        population=population,
        landcover_weights=weights,
        affected_population=affected,
        audit={
            "roads": roads_audit,
            "population": population_audit,
            "landcover_weights": weights_audit,
            "affected_population": affected_audit,
        },
        requires_synthetic_network=requires_synthetic,
    )


def validate_release(bundle: InputBundle, expected: dict[str, Any]) -> list[str]:
    """Return human-readable mismatches against the documented release checks."""
    mismatches: list[str] = []

    def check(label: str, actual: int, target: int) -> None:
        if actual != target:
            mismatches.append(f"{label}: observed {actual}, expected {target}")

    roads = bundle.audit["roads"]
    population = bundle.audit["population"]
    check("source features", roads["source_feature_count"], expected["expected_source_features"])
    check("road features", roads["road_feature_count"], expected["expected_road_features"])
    check(
        "routable features",
        roads["routable_feature_count"],
        expected["expected_routable_features"],
    )
    check(
        "stream features",
        roads["waterway_class_counts"].get("stream", 0),
        expected["expected_stream_features"],
    )
    check(
        "drain features",
        roads["waterway_class_counts"].get("drain", 0),
        expected["expected_drain_features"],
    )
    check("population", population["total_population"], expected["expected_population"])
    check("housing units", population["housing_units"], expected["expected_housing_units"])
    check(
        "population 65 plus",
        population["population_65_plus"],
        expected["expected_population_65_plus"],
    )
    check(
        "land-cover classes",
        bundle.audit["landcover_weights"]["landcover_class_count"],
        expected["expected_landcover_classes"],
    )
    observed_durations = [
        roads["exposure_duration_counts_all_features"].get(str(day), 0)
        for day in range(6)
    ]
    if observed_durations != expected["expected_duration_counts_all_features"]:
        mismatches.append(
            f"all-feature exposure duration counts: observed {observed_durations}, "
            f"expected {expected['expected_duration_counts_all_features']}"
        )
    observed_road_durations = [
        roads["exposure_duration_counts_roads_only"].get(str(day), 0)
        for day in range(6)
    ]
    if observed_road_durations != expected["expected_duration_counts_roads_only"]:
        mismatches.append(
            f"road-only exposure duration counts: observed {observed_road_durations}, "
            f"expected {expected['expected_duration_counts_roads_only']}"
        )
    if roads["missing_shape_length_after_cleaning"] != 0:
        mismatches.append("cleaned roads contain missing segment lengths")
    pds = bundle.affected_population.query("region == 'PDS'").sort_values("day")
    if not np.allclose(
        pds["affected_population"].to_numpy(),
        expected["expected_pds_daily_population"],
        atol=expected["decimal_tolerance"],
        rtol=0.0,
    ):
        mismatches.append("PDS daily affected population differs from documented values")
    return mismatches
