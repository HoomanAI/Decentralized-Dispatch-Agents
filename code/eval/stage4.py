"""Run the two myopic baselines through one replayable six-day episode."""

from __future__ import annotations

import json
import logging
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

LOGGER = logging.getLogger(__name__)
BASELINES = ("belief_aware", "reliability_blind")


def run_stage4(
    affected_population: pd.DataFrame,
    config: dict[str, Any],
    data_dir: Path,
    results_dir: Path,
) -> tuple[Path, Path, Path, Path, Path]:
    """Run identical demand through both myopic information assumptions."""
    graph, hospitals, edge_lookup = build_runtime_network(
        data_dir / "synthetic_road_features.geojson",
        data_dir / "synthetic_fire_fronts.geojson",
        config["network"],
    )
    core_nodes = np.array(
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
            core_nodes,
            age_share,
            config["demand"],
            config["environment"]["seed"] + day,
        )
        for day in range(7, 13)
    }
    summaries = []
    traces: dict[str, list[dict[str, Any]]] = {baseline: [] for baseline in BASELINES}
    for baseline in BASELINES:
        for day, patients in patients_by_day.items():
            summary, trace = run_episode(
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
            summaries.append(summary)
            traces[baseline].append(
                {
                    "day": day,
                    "patients": patients.to_dict("records"),
                    "epochs": trace,
                }
            )
    summary_frame = pd.DataFrame(summaries)
    day_7 = summary_frame.loc[summary_frame["day"] == 7]
    assert int(day_7["unreachable"].max()) > 0
    assert bool(summary_frame["belief_changed"].any())
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "stage4_myopic_summary.csv"
    trace_path = results_dir / "stage4_episode_trace.json"
    sweep_path = results_dir / "stage4_day7_alpha_sweep.csv"
    grid_path = results_dir / "stage4_reversal_robustness.csv"
    comparison_path = results_dir / "stage4_policy_comparison_invariance.csv"
    summary_frame.to_csv(summary_path, index=False)

    sweep_rows = []
    original_alpha = config["environment"]["scenario_alpha"]
    original_gamma = config["environment"]["scenario_gamma"]
    original_concentration = config["environment"].get(
        "scenario_alpha_concentration"
    )
    try:
        for alpha in config["reliability"]["scenario_alpha"]:
            config["environment"]["scenario_alpha"] = float(alpha)
            for concentration in config["reliability"][
                "scenario_alpha_concentration"
            ]:
                config["environment"]["scenario_alpha_concentration"] = concentration
                # Gamma cannot affect the first day because prior damage is zero.
                alpha_summaries = {}
                for baseline in BASELINES:
                    summary, _ = run_episode(
                        graph,
                        hospitals,
                        edge_lookup,
                        patients_by_day[7],
                        7,
                        baseline,
                        config["network"],
                        config["belief"],
                        config["environment"],
                        float(pds.loc[7, "affected_population"]),
                    )
                    alpha_summaries[baseline] = summary
                for gamma in config["reliability"]["scenario_gamma"]:
                    for baseline, summary in alpha_summaries.items():
                        sweep_rows.append(
                            {
                                "alpha": alpha,
                                "gamma": gamma,
                                "alpha_concentration": concentration,
                                "heterogeneous_alpha": concentration is not None,
                                "alpha_heterogeneity_seed": config["environment"][
                                    "alpha_heterogeneity_seed"
                                ],
                                **summary,
                            }
                        )
    finally:
        config["environment"]["scenario_alpha"] = original_alpha
        config["environment"]["scenario_gamma"] = original_gamma
        config["environment"]["scenario_alpha_concentration"] = original_concentration
    pd.DataFrame(sweep_rows).to_csv(sweep_path, index=False)

    grid_rows = []
    recovery_ratio = float(config["reliability"]["reversal_recovery_ratio"])
    minimum_relapse_arcs = int(
        np.ceil(
            config["reliability"]["reversal_min_arc_fraction"]
            * config["network"]["routable_edge_count"]
        )
    )
    for alpha in config["reliability"]["scenario_alpha"]:
        for gamma in config["reliability"]["scenario_gamma"]:
            for concentration in config["reliability"][
                "scenario_alpha_concentration"
            ]:
                blocked = {}
                for day in range(7, 13):
                    damage = damage_for_day(
                        graph,
                        day,
                        float(alpha),
                        float(gamma),
                        concentration,
                        config["environment"]["alpha_heterogeneity_seed"],
                    )
                    upsilon = class_parameter_by_edge(
                        graph, config["environment"]["upsilon"], "upsilon"
                    )
                    reliability = (1.0 - damage) * (1.0 - upsilon * damage)
                    blocked[day] = int(
                        np.count_nonzero(
                            reliability < config["environment"]["phi_min"]
                        )
                    )
                recovery_limit = recovery_ratio * blocked[8]
                recovery_holds = max(blocked[9], blocked[10], blocked[11]) <= recovery_limit
                relapse_holds = blocked[12] - blocked[11] >= minimum_relapse_arcs
                grid_rows.append(
                    {
                        "alpha": alpha,
                        "gamma": gamma,
                        "alpha_concentration": concentration,
                        "heterogeneous_alpha": concentration is not None,
                        "phi_min": config["environment"]["phi_min"],
                        "alpha_heterogeneity_seed": config["environment"][
                            "alpha_heterogeneity_seed"
                        ],
                        **{f"blocked_day_{day}": blocked[day] for day in range(7, 13)},
                        "recovery_limit_ratio": recovery_ratio,
                        "minimum_relapse_arcs": minimum_relapse_arcs,
                        "recovery_holds": recovery_holds,
                        "relapse_holds": relapse_holds,
                        "reversal_holds": recovery_holds and relapse_holds,
                    }
                )
    pd.DataFrame(grid_rows).to_csv(grid_path, index=False)

    comparison_rows = []
    minimum_alpha = float(config["reliability"]["passability_min_alpha"])
    minimum_gamma = float(config["reliability"]["passability_min_gamma"])
    comparison_tolerance = float(
        config["reliability"]["policy_comparison_tolerance"]
    )
    try:
        for alpha in config["reliability"]["scenario_alpha"]:
            if float(alpha) < minimum_alpha:
                continue
            for gamma in config["reliability"]["scenario_gamma"]:
                if float(gamma) < minimum_gamma:
                    continue
                for concentration in config["reliability"][
                    "scenario_alpha_concentration"
                ]:
                    config["environment"]["scenario_alpha"] = float(alpha)
                    config["environment"]["scenario_gamma"] = float(gamma)
                    config["environment"][
                        "scenario_alpha_concentration"
                    ] = concentration
                    by_baseline = {baseline: {} for baseline in BASELINES}
                    for baseline in BASELINES:
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
                            by_baseline[baseline][day] = summary
                    aware_total = sum(
                        item["survival_total"]
                        for item in by_baseline["belief_aware"].values()
                    )
                    blind_total = sum(
                        item["survival_total"]
                        for item in by_baseline["reliability_blind"].values()
                    )
                    row = {
                        "alpha": alpha,
                        "gamma": gamma,
                        "phi_min": config["environment"]["phi_min"],
                        "alpha_concentration": concentration,
                        "heterogeneous_alpha": concentration is not None,
                        "alpha_heterogeneity_seed": config["environment"][
                            "alpha_heterogeneity_seed"
                        ],
                        "aware_total": aware_total,
                        "blind_total": blind_total,
                        "overall_difference": aware_total - blind_total,
                        "comparison_tolerance": comparison_tolerance,
                        "overall_aware_not_worse": (
                            aware_total >= blind_total - comparison_tolerance
                        ),
                        "aware_blocked_arc_encounters": sum(
                            item["blocked_arc_encounters"]
                            for item in by_baseline["belief_aware"].values()
                        ),
                        "blind_blocked_arc_encounters": sum(
                            item["blocked_arc_encounters"]
                            for item in by_baseline["reliability_blind"].values()
                        ),
                    }
                    for day in range(7, 13):
                        aware = by_baseline["belief_aware"][day]["survival_total"]
                        blind = by_baseline["reliability_blind"][day]["survival_total"]
                        row[f"day_{day}_aware"] = aware
                        row[f"day_{day}_blind"] = blind
                        row[f"day_{day}_difference"] = aware - blind
                        row[f"day_{day}_blind_blocked_arc_encounters"] = (
                            by_baseline["reliability_blind"][day][
                                "blocked_arc_encounters"
                            ]
                        )
                    row["relapse_day_aware_not_worse"] = (
                        row["day_12_difference"] >= -comparison_tolerance
                    )
                    comparison_rows.append(row)
    finally:
        config["environment"]["scenario_alpha"] = original_alpha
        config["environment"]["scenario_gamma"] = original_gamma
        config["environment"]["scenario_alpha_concentration"] = original_concentration
    comparison_frame = pd.DataFrame(comparison_rows)
    comparison_frame.to_csv(comparison_path, index=False)
    trace_path.write_text(
        json.dumps(
            {
                "seed": config["environment"]["seed"],
                "synthetic_input": True,
                "baselines": traces,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    LOGGER.info(
        "Stage 4 completed with daily patient counts %s",
        {day: len(patients) for day, patients in patients_by_day.items()},
    )
    return summary_path, trace_path, sweep_path, grid_path, comparison_path
