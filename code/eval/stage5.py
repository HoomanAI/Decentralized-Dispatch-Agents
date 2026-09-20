"""Run the optimistic bound and targeted travel-capacity checks."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from code.baselines.bound import run_optimistic_bound
from code.demand.patients import generate_patients
from code.env.dispatch import build_runtime_network

LOGGER = logging.getLogger(__name__)


def run_stage5(
    affected_population: pd.DataFrame,
    config: dict[str, Any],
    data_dir: Path,
    results_dir: Path,
) -> tuple[Path, Path]:
    """Solve one realized-reliability dispatch MILP for each study day."""
    graph, hospitals, edge_lookup = build_runtime_network(
        data_dir / "synthetic_road_features.geojson",
        data_dir / "synthetic_fire_fronts.geojson",
        config["network"],
    )
    nodes = np.array(
        [node for node, attributes in graph.nodes(data=True) if not attributes.get("hospital")]
    )
    age_share = config["validation"]["expected_population_65_plus"] / config["validation"]["expected_population"]
    pds = affected_population.query("region == 'PDS'").set_index("day")
    union_population = float(pds["Total"].iloc[0])
    summaries = []
    assignments = {}
    for day in range(7, 13):
        patients = generate_patients(
            day,
            float(pds.loc[day, "affected_population"]),
            union_population,
            nodes,
            age_share,
            config["demand"],
            config["environment"]["seed"] + day,
        )
        summary, rows = run_optimistic_bound(
            graph,
            hospitals,
            edge_lookup,
            patients,
            day,
            config["network"],
            config["environment"],
            float(config["demand"]["operational_minutes"]),
            float(pds.loc[day, "affected_population"]),
        )
        summaries.append(summary)
        assignments[str(day)] = rows
        LOGGER.info("Stage 5 day %d solver status: %s", day, summary["solver_status"])
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "stage5_optimistic_bound_summary.csv"
    assignment_path = results_dir / "stage5_optimistic_bound_assignments.json"
    summary_frame = pd.DataFrame(summaries)
    myopic_path = results_dir / "stage4_myopic_summary.csv"
    if myopic_path.exists():
        myopic = pd.read_csv(myopic_path)
        best_myopic = myopic.groupby("day")["survival_total"].max()
        comparison = summary_frame.set_index("day")["survival_total"] - best_myopic
        assert bool((comparison >= -1.0e-9).all())
    summary_frame.to_csv(summary_path, index=False)

    exact_rows = []
    for day in (8, 12):
        patients = generate_patients(
            day,
            float(pds.loc[day, "affected_population"]),
            union_population,
            nodes,
            age_share,
            config["demand"],
            config["environment"]["seed"] + day,
        )
        exact, _ = run_optimistic_bound(
            graph,
            hospitals,
            edge_lookup,
            patients,
            day,
            config["network"],
            config["environment"],
            float(config["demand"]["operational_minutes"]),
            float(pds.loc[day, "affected_population"]),
            charge_travel_capacity=True,
        )
        relaxed = summary_frame.set_index("day").loc[day]
        policy = float(best_myopic.loc[day]) if myopic_path.exists() else float("nan")
        exact_rows.append(
            {
                **exact,
                "relaxed_bound": float(relaxed["survival_total"]),
                "best_myopic": policy,
                "relaxation_slack": float(relaxed["survival_total"] - exact["survival_total"]),
                "reportable_gap": float(exact["survival_total"] - policy),
            }
        )
    pd.DataFrame(exact_rows).to_csv(
        results_dir / "stage5_travel_capacity_check.csv", index=False
    )
    assignment_path.write_text(
        json.dumps({"synthetic_input": True, "days": assignments}, indent=2),
        encoding="utf-8",
    )
    return summary_path, assignment_path
