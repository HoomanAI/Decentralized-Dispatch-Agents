"""Perfect-information myopic bound on the value of network information."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from code.demand.patients import generate_patients
from code.env.dispatch import build_runtime_network, run_episode


def run_perfect_information_myopic(
    affected_population: pd.DataFrame,
    config: dict[str, Any],
    data_dir: Path,
    results_dir: Path,
) -> Path:
    """Reveal realized arc reliability before dispatch and run the same myopic rule."""
    graph, hospitals, edge_lookup = build_runtime_network(
        data_dir / "synthetic_road_features.geojson",
        data_dir / "synthetic_fire_fronts.geojson",
        config["network"],
    )
    nodes = np.array([node for node, attributes in graph.nodes(data=True) if not attributes.get("hospital")])
    age_share = config["validation"]["expected_population_65_plus"] / config["validation"]["expected_population"]
    pds = affected_population.query("region == 'PDS'").set_index("day")
    union_population = float(pds["Total"].iloc[0])
    rows = []
    for day in range(7, 13):
        patients = generate_patients(day, float(pds.loc[day, "affected_population"]), union_population, nodes, age_share, config["demand"], int(config["environment"]["seed"]) + day)
        summary, _ = run_episode(graph, hospitals, edge_lookup, patients, day, "perfect_information", config["network"], config["belief"], config["environment"], float(pds.loc[day, "affected_population"]))
        rows.append({"synthetic_input": True, **summary})
    path = results_dir / "perfect_information_myopic.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
