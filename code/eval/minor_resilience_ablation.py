"""Controlled belief-architecture resilience ablation on the full instance."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from code.demand.patients import generate_patients
from code.env.vehicle_centric import run_vehicle_episode
from code.eval.stage11_component_b import FIXED_BUDGETS, _safe_summary, _training_setup
from code.exploration.triage import ExplorationLedger, InformationRoutePolicy


ARCHITECTURES = (
    "centralized_stale_global",
    "decentralized_merge",
    "decentralized_no_merge",
)
CONNECTIVITY_LEVELS = (1.00, 0.50, 0.25, 0.00)
POSTURE = (0.375, 0.50, 0.50)
_CONTEXT: tuple[Any, ...] | None = None


def _initialize(
    affected: pd.DataFrame,
    config: dict[str, Any],
    data_dir: Path,
) -> None:
    global _CONTEXT
    _CONTEXT = (*_training_setup(affected, config, data_dir, POSTURE), config)


def _architecture_argument(name: str) -> str:
    if name == "centralized_stale_global":
        return "centralized_coordinator"
    return name


def _run_task(task: tuple[float, str, int]) -> list[dict[str, Any]]:
    if _CONTEXT is None:
        raise RuntimeError("Resilience worker was not initialized")
    q, architecture, seed = task
    pc, nc, dc, graph, hospitals, lookup, pds, union, age, env, nodes, config = (
        _CONTEXT
    )
    fleet_count = sum(int(value) for value in env["fleet_by_class"].values())
    belief_evidence: list[tuple[np.ndarray, np.ndarray]] = []
    architecture_state: dict[str, Any] = {}
    isolation: list[int] = []
    rows = []
    for day in range(8, 13):
        patient_seed = int(config["environment"]["seed"]) + 300000 + seed * 100 + day
        patients = generate_patients(
            day,
            float(pds.loc[day, "affected_population"]),
            union,
            nodes,
            age,
            dc,
            patient_seed,
        )
        ledger = ExplorationLedger(FIXED_BUDGETS)
        policies = [
            InformationRoutePolicy(ledger, 1.0) for _ in range(fleet_count)
        ]
        audit: dict[str, Any] = {}
        summary, _ = run_vehicle_episode(
            graph,
            hospitals,
            lookup,
            patients,
            day,
            policies,
            nc,
            config["belief"],
            env,
            float(pds.loc[day, "affected_population"]),
            communication_reliability=q,
            seed=patient_seed,
            belief_evidence=belief_evidence,
            isolation_epochs_state=isolation,
            belief_architecture=_architecture_argument(architecture),
            architecture_state=architecture_state,
            architecture_audit=audit,
        )
        rows.append(
            {
                "architecture": architecture,
                "seed": seed,
                "day": day,
                "survival_total": float(summary["survival_total"]),
                "patients": int(summary["patients"]),
                "served": int(summary["served"]),
                "communication_reliability": q,
                "mean_belief_error": float(audit["mean_belief_error"]),
                "observations_incorporated": int(audit["observations_incorporated"]),
                "observations_pending": int(audit["observations_pending"]),
                "epsilon_1": FIXED_BUDGETS[0],
                "epsilon_2": FIXED_BUDGETS[1],
                "epsilon_3": FIXED_BUDGETS[2],
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


def _summarize(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for q in CONNECTIVITY_LEVELS:
        level = raw[raw.communication_reliability.eq(q)]
        reference = level[level.architecture.eq("centralized_stale_global")]
        for architecture in ARCHITECTURES:
            arm = level[level.architecture.eq(architecture)]
            for endpoint, days in {
                "five_day": [8, 9, 10, 11, 12],
                "day_12": [12],
            }.items():
                arm_values = arm[arm.day.isin(days)].groupby("seed").survival_total.sum()
                reference_values = (
                    reference[reference.day.isin(days)]
                    .groupby("seed")
                    .survival_total.sum()
                )
                difference = arm_values - reference_values.loc[arm_values.index]
                point, lower, upper, p_value = _safe_summary(difference)
                rows.append(
                    {
                        "communication_reliability": q,
                        "architecture": architecture,
                        "endpoint": endpoint,
                        "survival_mean": float(arm_values.mean()),
                        "paired_difference_from_centralized": point,
                        "lower": lower,
                        "upper": upper,
                        "p": p_value,
                        "n": len(difference),
                        "service_rate": float(arm.served.sum() / arm.patients.sum()),
                        "end_horizon_observations_incorporated": float(
                            arm[arm.day.eq(12)].observations_incorporated.mean()
                        ),
                        "end_day_mean_belief_error": float(
                            arm[arm.day.eq(12)].mean_belief_error.mean()
                        ),
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
    q1_tasks = [
        (1.0, architecture, seed)
        for architecture in ARCHITECTURES
        for seed in range(30)
    ]
    remaining_tasks = [
        (q, architecture, seed)
        for q in CONNECTIVITY_LEVELS
        if q != 1.0
        for architecture in ARCHITECTURES
        for seed in range(30)
    ]
    partial = results / "minor_resilience_ablation.partial.csv"
    collected: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=8,
        initializer=_initialize,
        initargs=(affected, config, root / "data"),
    ) as executor:
        for rows in executor.map(_run_task, q1_tasks):
            collected.extend(rows)
            pd.DataFrame(collected).to_csv(partial, index=False)
        q1 = pd.DataFrame(collected)
        comparison = q1.pivot_table(
            index=["seed", "day"],
            columns="architecture",
            values="survival_total",
        )
        maximum_difference = float(
            (
                comparison["centralized_stale_global"]
                - comparison["decentralized_merge"]
            )
            .abs()
            .max()
        )
        gate = pd.DataFrame(
            [
                {
                    "communication_reliability": 1.0,
                    "maximum_absolute_daily_survival_difference": maximum_difference,
                    "tolerance": 1.0e-9,
                    "pass": maximum_difference <= 1.0e-9,
                    "n": 30,
                }
            ]
        )
        gate.to_csv(results / "minor_resilience_q1_harness_gate.csv", index=False)
        if maximum_difference > 1.0e-9:
            raise RuntimeError(
                "Centralized and decentralized merge differ at q=1; stopping ablation"
            )
        for rows in executor.map(_run_task, remaining_tasks):
            collected.extend(rows)
            pd.DataFrame(collected).to_csv(partial, index=False)
    raw = pd.DataFrame(collected).sort_values(
        ["communication_reliability", "architecture", "seed", "day"],
        ascending=[False, True, True, True],
    )
    raw.to_csv(results / "minor_resilience_ablation.csv", index=False)
    _summarize(raw).to_csv(
        results / "minor_resilience_ablation_summary.csv", index=False
    )


if __name__ == "__main__":
    main()
