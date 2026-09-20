"""Compare dispatch policies on distinct active-memory network realizations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from code.demand.patients import generate_patients
from code.env.dispatch import build_runtime_network, run_episode


def run_active_memory_comparison(
    affected_population: pd.DataFrame,
    config: dict[str, Any],
    data_dir: Path,
    results_dir: Path,
) -> Path:
    """Run each distinct plausible blocked-mask sequence with active memory once."""
    memory = pd.read_csv(results_dir / "stage4_memory_phi_sweep.csv")
    candidates = memory.loc[
        memory["memory_active"]
        & (memory["alpha"] >= config["reliability"]["passability_min_alpha"])
        & (memory["gamma"] >= config["reliability"]["passability_min_gamma"])
    ].copy()
    representatives = candidates.drop_duplicates("network_realization_id")
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
    original = (
        config["environment"]["scenario_alpha"],
        config["environment"]["scenario_gamma"],
        config["environment"]["phi_min"],
        config["environment"].get("scenario_alpha_concentration"),
    )
    output_rows = []
    try:
        for _, specification in representatives.iterrows():
            concentration = specification["alpha_concentration"]
            concentration = None if pd.isna(concentration) else float(concentration)
            config["environment"]["scenario_alpha"] = float(specification["alpha"])
            config["environment"]["scenario_gamma"] = float(specification["gamma"])
            config["environment"]["phi_min"] = float(specification["phi_min"])
            config["environment"]["scenario_alpha_concentration"] = concentration
            summaries = {"belief_aware": {}, "reliability_blind": {}}
            for baseline in summaries:
                for day, patients in patients_by_day.items():
                    summary, _ = run_episode(
                        graph,
                        hospitals,
                        edge_lookup,
                        patients,
                        day,
                        baseline,
                        config["network"],
                        config["belief"],
                        config["environment"],
                        float(pds.loc[day, "affected_population"]),
                    )
                    summaries[baseline][day] = summary
            aware_total = sum(item["survival_total"] for item in summaries["belief_aware"].values())
            blind_total = sum(item["survival_total"] for item in summaries["reliability_blind"].values())
            row = {
                "network_realization_id": specification["network_realization_id"],
                "alpha": specification["alpha"],
                "gamma": specification["gamma"],
                "phi_min": specification["phi_min"],
                "alpha_concentration": concentration,
                "memory_effect_total": int(specification["memory_effect_total"]),
                "memory_effect_peak": int(specification["memory_effect_peak"]),
                "aware_total": aware_total,
                "blind_total": blind_total,
                "overall_difference": aware_total - blind_total,
                "blind_blocked_arc_encounters": sum(
                    item["blocked_arc_encounters"]
                    for item in summaries["reliability_blind"].values()
                ),
            }
            for day in range(7, 13):
                aware = summaries["belief_aware"][day]["survival_total"]
                blind = summaries["reliability_blind"][day]["survival_total"]
                row[f"day_{day}_aware"] = aware
                row[f"day_{day}_blind"] = blind
                row[f"day_{day}_difference"] = aware - blind
                row[f"day_{day}_blocked_not_exposed"] = int(
                    specification[f"day_{day}_blocked_not_exposed"]
                )
            output_rows.append(row)
    finally:
        (
            config["environment"]["scenario_alpha"],
            config["environment"]["scenario_gamma"],
            config["environment"]["phi_min"],
            config["environment"]["scenario_alpha_concentration"],
        ) = original
    output_path = results_dir / "stage4_active_memory_policy_comparison.csv"
    pd.DataFrame(output_rows).to_csv(output_path, index=False)
    return output_path
