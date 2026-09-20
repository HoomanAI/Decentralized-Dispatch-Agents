"""Run Stage 9 baselines across communication reliability with stale central views."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from code.baselines.stage9 import make_policies
from code.demand.patients import generate_patients
from code.env.dispatch import build_runtime_network
from code.env.vehicle_centric import ProtocolMyopic, run_vehicle_episode


METHODS = (
    "protocol_myopic",
    "reliability_blind_myopic",
    "yan",
    "ahmadi",
    "peng",
    "ga",
    "alns",
)
CENTRALIZED_METHODS = frozenset(("yan", "ahmadi", "ga", "alns"))
PRIMARY_CONNECTIVITY_LEVELS = (1.00, 0.50, 0.25, 0.00)
POSTURE = (0.375, 0.50, 0.50)
_CONTEXT: tuple[Any, ...] | None = None


def _load_tuned_weights(results_dir: Path) -> dict[str, list[float]]:
    tuning = pd.read_csv(results_dir / "stage9_metaheuristic_tuning.csv")
    output = {}
    for method in ("ga", "alns"):
        row = tuning[tuning.method.eq(method)].sort_values(
            "objective", ascending=False
        ).iloc[0]
        output[method] = [float(row[f"w_{index}"]) for index in range(5)]
    return output


def _initialize(
    affected: pd.DataFrame,
    config: dict[str, Any],
    data_dir: Path,
    tuned: dict[str, list[float]],
) -> None:
    global _CONTEXT
    network = config["network"]
    demand = config["demand"]
    environment = dict(config["environment"])
    environment.update(
        {
            "scenario_alpha": POSTURE[0],
            "scenario_gamma": POSTURE[1],
            "phi_min": POSTURE[2],
            "fleet_by_class": {"A": 4, "B": 4, "C": 4},
            "operational_minutes": float(demand["operational_minutes"]),
            "exploration_enabled": False,
            "exploration_budgets": (0.0, 0.0, 0.0),
        }
    )
    graph, hospitals, lookup = build_runtime_network(
        data_dir / "synthetic_road_features.geojson",
        data_dir / "synthetic_fire_fronts.geojson",
        network,
    )
    assert graph.number_of_edges() - len(hospitals) == 1297
    pds = affected.query("region == 'PDS'").set_index("day")
    union = float(pds["Total"].iloc[0])
    age = (
        config["validation"]["expected_population_65_plus"]
        / config["validation"]["expected_population"]
    )
    nodes = np.asarray(
        [node for node, attrs in graph.nodes(data=True) if not attrs.get("hospital")]
    )
    _CONTEXT = (
        config,
        network,
        demand,
        environment,
        graph,
        hospitals,
        lookup,
        pds,
        union,
        age,
        nodes,
        tuned,
    )


def _run_task(task: tuple[float, str, int]) -> list[dict[str, Any]]:
    if _CONTEXT is None:
        raise RuntimeError("Stage 9 connectivity worker was not initialized")
    q, method, seed = task
    (
        config,
        network,
        demand,
        environment,
        graph,
        hospitals,
        lookup,
        pds,
        union,
        age,
        nodes,
        tuned,
    ) = _CONTEXT
    vehicle_count = sum(int(value) for value in environment["fleet_by_class"].values())
    belief_evidence: list[tuple[np.ndarray, np.ndarray]] = []
    planner_evidence: list[tuple[np.ndarray, np.ndarray]] = []
    centralized = method in CENTRALIZED_METHODS
    rows = []
    for day in range(8, 13):
        patient_seed = int(config["environment"]["seed"]) + 300000 + seed * 100 + day
        patients = generate_patients(
            day,
            float(pds.loc[day, "affected_population"]),
            union,
            nodes,
            age,
            demand,
            patient_seed,
        )
        local_environment = dict(environment)
        if method == "reliability_blind_myopic":
            local_environment["belief_blind"] = True
        if method in ("protocol_myopic", "reliability_blind_myopic"):
            policies = [ProtocolMyopic() for _ in range(vehicle_count)]
        elif method in ("ga", "alns"):
            policies = make_policies(
                method, vehicle_count, np.asarray(tuned[method], dtype=float)
            )
        else:
            policies = make_policies(method, vehicle_count)
        summary, _ = run_vehicle_episode(
            graph,
            hospitals,
            lookup,
            patients,
            day,
            policies,
            network,
            config["belief"],
            local_environment,
            float(pds.loc[day, "affected_population"]),
            communication_reliability=q,
            seed=patient_seed,
            belief_evidence=belief_evidence,
            centralized_stale_view=centralized,
            planner_evidence=planner_evidence if centralized else None,
        )
        rows.append(
            {
                "method": method,
                "seed": seed,
                "day": day,
                "survival_total": float(summary["survival_total"]),
                "patients": int(summary["patients"]),
                "served": int(summary["served"]),
                "communication_reliability": q,
                "communication_model": (
                    "stale_global_view" if centralized else "decentralized_component_merge"
                ),
                "instance": "full_1297_arc",
                "fleet": 12,
                "fleet_A": 4,
                "fleet_B": 4,
                "fleet_C": 4,
                "alpha": POSTURE[0],
                "gamma": POSTURE[1],
                "phi_min": POSTURE[2],
            }
        )
    return rows


def _summary(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for q in PRIMARY_CONNECTIVITY_LEVELS:
        level = raw[raw.communication_reliability.eq(q)]
        reference = level[level.method.eq("protocol_myopic")]
        for method in METHODS:
            arm = level[level.method.eq(method)]
            for endpoint, days in {
                "five_day": [8, 9, 10, 11, 12],
                "day_12": [12],
            }.items():
                arm_values = arm[arm.day.isin(days)].groupby("seed").survival_total.sum()
                ref_values = (
                    reference[reference.day.isin(days)]
                    .groupby("seed")
                    .survival_total.sum()
                )
                difference = arm_values - ref_values.loc[arm_values.index]
                mean = float(difference.mean())
                if float(difference.std(ddof=1)) == 0.0:
                    lower = upper = mean
                    p_value = 1.0 if mean == 0.0 else 0.0
                else:
                    lower, upper = stats.t.interval(
                        0.95,
                        len(difference) - 1,
                        loc=mean,
                        scale=stats.sem(difference),
                    )
                    p_value = float(stats.ttest_1samp(difference, 0.0).pvalue)
                rows.append(
                    {
                        "communication_reliability": q,
                        "method": method,
                        "endpoint": endpoint,
                        "survival_mean": float(arm_values.mean()),
                        "paired_difference": mean,
                        "lower": float(lower),
                        "upper": float(upper),
                        "p": p_value,
                        "n": len(difference),
                        "service_rate": float(arm.served.sum() / arm.patients.sum()),
                        "communication_model": arm.communication_model.iloc[0],
                        "instance": "full_1297_arc",
                        "fleet": "A:4 B:4 C:4",
                        "posture": "alpha=0.375 gamma=0.50 phi_min=0.50",
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    results = root / "results"
    config = yaml.safe_load(
        (root / "code/config/default.yaml").read_text(encoding="utf-8")
    )
    affected = pd.read_csv(root / "data/daily_affected_population_clean.csv")
    tuned = _load_tuned_weights(results)
    tasks = [
        (q, method, seed)
        for q in PRIMARY_CONNECTIVITY_LEVELS
        for method in METHODS
        for seed in range(30)
    ]
    partial = results / "stage9_connectivity_sweep.partial.csv"
    collected: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=8,
        initializer=_initialize,
        initargs=(affected, config, root / "data", tuned),
    ) as executor:
        for rows in executor.map(_run_task, tasks):
            collected.extend(rows)
            pd.DataFrame(collected).to_csv(partial, index=False)
    raw = pd.DataFrame(collected).sort_values(
        ["communication_reliability", "method", "seed", "day"],
        ascending=[False, True, True, True],
    )
    raw.to_csv(results / "stage9_connectivity_sweep.csv", index=False)
    _summary(raw).to_csv(results / "stage9_connectivity_summary.csv", index=False)


if __name__ == "__main__":
    main()
