"""Diagnose surprise-on-contact behavior for reliability-blind dispatch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from code.demand.patients import generate_patients
from code.env.dispatch import (
    build_runtime_network,
    class_parameter_by_edge,
    damage_for_day,
    run_episode,
)


def run_blind_diagnostic(
    affected_population: pd.DataFrame,
    config: dict[str, Any],
    data_dir: Path,
    results_dir: Path,
) -> Path:
    """Compare two equivalent masks and one genuinely different blocked mask."""
    graph, hospitals, edge_lookup = build_runtime_network(
        data_dir / "synthetic_road_features.geojson",
        data_dir / "synthetic_fire_fronts.geojson",
        config["network"],
    )
    nodes = np.array(
        [node for node, attributes in graph.nodes(data=True) if not attributes.get("hospital")]
    )
    age_share = config["validation"]["expected_population_65_plus"] / config["validation"][
        "expected_population"
    ]
    pds = affected_population.query("region == 'PDS'").set_index("day")
    union_population = float(pds["Total"].iloc[0])
    patients_by_day = {
        day: generate_patients(
            day,
            float(pds.loc[day, "affected_population"]),
            union_population,
            nodes,
            age_share,
            config["demand"],
            config["environment"]["seed"] + day,
        )
        for day in range(7, 13)
    }
    specifications = {
        "homogeneous_0.5_0.75": (0.5, 0.75, None),
        "homogeneous_1.0_1.0": (1.0, 1.0, None),
        "heterogeneous_0.5_0.75_k20": (0.5, 0.75, 20.0),
    }
    original = (
        config["environment"]["scenario_alpha"],
        config["environment"]["scenario_gamma"],
        config["environment"].get("scenario_alpha_concentration"),
    )
    output: dict[str, Any] = {"synthetic_input": True, "specifications": {}}
    try:
        for label, (alpha, gamma, concentration) in specifications.items():
            config["environment"]["scenario_alpha"] = alpha
            config["environment"]["scenario_gamma"] = gamma
            config["environment"]["scenario_alpha_concentration"] = concentration
            days = []
            traversal_log = []
            for day in range(7, 13):
                damage = damage_for_day(
                    graph,
                    day,
                    alpha,
                    gamma,
                    concentration,
                    config["environment"]["alpha_heterogeneity_seed"],
                )
                upsilon = class_parameter_by_edge(
                    graph, config["environment"]["upsilon"], "upsilon"
                )
                reliability = (1.0 - damage) * (1.0 - upsilon * damage)
                summary, trace = run_episode(
                    graph,
                    hospitals,
                    edge_lookup,
                    patients_by_day[day],
                    day,
                    "reliability_blind",
                    config["network"],
                    config["belief"],
                    config["environment"],
                    float(pds.loc[day, "affected_population"]),
                )
                days.append(
                    {
                        **summary,
                        "blocked_mask_edge_ids": [
                            int(data["edge_id"])
                            for index, (_, _, data) in enumerate(graph.edges(data=True))
                            if reliability[index] < config["environment"]["phi_min"]
                        ],
                    }
                )
                traversal_log.extend(
                    {
                        "day": day,
                        "patient_id": item["patient_id"],
                        "arcs_traversed": item["arcs_traversed"],
                        "blocked_arc_encounters": item["blocked_arc_encounters"],
                        "outcome": item["outcome"],
                    }
                    for item in trace
                )
            output["specifications"][label] = {
                "alpha": alpha,
                "gamma": gamma,
                "alpha_concentration": concentration,
                "survival_total": sum(item["survival_total"] for item in days),
                "blocked_arc_encounters": sum(
                    item["blocked_arc_encounters"] for item in days
                ),
                "days": days,
                "traversal_log": traversal_log,
            }
    finally:
        (
            config["environment"]["scenario_alpha"],
            config["environment"]["scenario_gamma"],
            config["environment"]["scenario_alpha_concentration"],
        ) = original

    first = output["specifications"]["homogeneous_0.5_0.75"]
    second = output["specifications"]["homogeneous_1.0_1.0"]
    third = output["specifications"]["heterogeneous_0.5_0.75_k20"]
    output["comparisons"] = {
        "requested_pair_blocked_masks_identical": all(
            left["blocked_mask_edge_ids"] == right["blocked_mask_edge_ids"]
            for left, right in zip(first["days"], second["days"])
        ),
        "requested_pair_traversal_logs_identical": (
            first["traversal_log"] == second["traversal_log"]
        ),
        "discriminating_pair_blocked_masks_differ": any(
            left["blocked_mask_edge_ids"] != right["blocked_mask_edge_ids"]
            for left, right in zip(first["days"], third["days"])
        ),
        "discriminating_pair_traversal_logs_differ": (
            first["traversal_log"] != third["traversal_log"]
        ),
        "discriminating_pair_survival_differs": not np.isclose(
            first["survival_total"], third["survival_total"], atol=1.0e-9
        ),
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "stage4_blind_contact_diagnostic.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output_path
