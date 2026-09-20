"""Evaluate the saved permissive Component B policy across connectivity levels."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from code.demand.patients import generate_patients
from code.env.vehicle_centric import run_vehicle_episode
from code.eval.stage11_component_b import (
    FIXED_BUDGETS,
    _safe_summary,
    _training_setup,
)
from code.exploration.triage import ExplorationLedger, InformationRoutePolicy
from code.policy.vehicle_centric import NeuralRouteController, RouteActorQCritic


CONNECTIVITY_LEVELS = (1.00, 0.75, 0.50, 0.25, 0.10, 0.00)
POSTURE = (0.375, 0.50, 0.50)
_CONTEXT: tuple[Any, ...] | None = None


def _initialize(
    affected: pd.DataFrame,
    config: dict[str, Any],
    data_dir: Path,
    checkpoint: Path,
) -> None:
    global _CONTEXT
    setup = _training_setup(affected, config, data_dir, POSTURE)
    pc, nc, dc, graph, hospitals, lookup, pds, union, age, env, nodes = setup
    fleet_count = sum(int(value) for value in env["fleet_by_class"].values())
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = RouteActorQCritic(int(pc["hidden_dim"]), fleet_count)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    _CONTEXT = (*setup, config, model)


def _run_task(task: tuple[float, int]) -> list[dict[str, Any]]:
    if _CONTEXT is None:
        raise RuntimeError("Connectivity worker was not initialized")
    q, seed = task
    pc, nc, dc, graph, hospitals, lookup, pds, union, age, env, nodes, config, model = (
        _CONTEXT
    )
    fleet_count = sum(int(value) for value in env["fleet_by_class"].values())
    learned_evidence: list[tuple[np.ndarray, np.ndarray]] = []
    fixed_evidence: list[tuple[np.ndarray, np.ndarray]] = []
    learned_isolation: list[int] = []
    fixed_isolation: list[int] = []
    learned_controls = [
        NeuralRouteController(model, lookup, greedy=True) for _ in range(fleet_count)
    ]
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
        learned, learned_trace = run_vehicle_episode(
            graph,
            hospitals,
            lookup,
            patients,
            day,
            learned_controls,
            nc,
            config["belief"],
            env,
            float(pds.loc[day, "affected_population"]),
            communication_reliability=q,
            seed=patient_seed,
            belief_evidence=learned_evidence,
            isolation_epochs_state=learned_isolation,
        )
        ledger = ExplorationLedger(FIXED_BUDGETS)
        fixed_controls = [
            InformationRoutePolicy(ledger, 1.0) for _ in range(fleet_count)
        ]
        fixed, fixed_trace = run_vehicle_episode(
            graph,
            hospitals,
            lookup,
            patients,
            day,
            fixed_controls,
            nc,
            config["belief"],
            env,
            float(pds.loc[day, "affected_population"]),
            communication_reliability=q,
            seed=patient_seed,
            belief_evidence=fixed_evidence,
            isolation_epochs_state=fixed_isolation,
        )
        learned_cost_2 = sum(
            float(item["exploration_cost"])
            for item in learned_trace
            if int(item["triage"]) == 2
        )
        rows.append(
            {
                "synthetic_input": True,
                "seed": seed,
                "day": day,
                "communication_reliability": q,
                "scenario_alpha": POSTURE[0],
                "scenario_gamma": POSTURE[1],
                "phi_min": POSTURE[2],
                "fleet": "A4_B4_C4",
                "learned_survival": learned["survival_total"],
                "fixed_ration_survival": fixed["survival_total"],
                "difference": learned["survival_total"] - fixed["survival_total"],
                "learned_detours": sum(
                    int(item["route_rank"] > 0) for item in learned_trace
                ),
                "fixed_ration_detours": sum(
                    int(item["route_rank"] > 0) for item in fixed_trace
                ),
                "learned_cost_2": learned_cost_2,
                "fixed_ration_cost_2": ledger.costs[2],
            }
        )
    return rows


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    results = root / "results"
    config = yaml.safe_load(
        (root / "code/config/default.yaml").read_text(encoding="utf-8")
    )
    affected = pd.read_csv(root / "data/daily_affected_population_clean.csv")
    seeds = range(int(config["policy_training"]["evaluation_seeds"]))
    tasks = [(q, seed) for q in CONNECTIVITY_LEVELS for seed in seeds]
    checkpoint = results / "stage11_component_b_ration_init_permissive.pt"
    partial = results / "stage11_component_b_connectivity_evaluation.partial.csv"
    collected: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=8,
        initializer=_initialize,
        initargs=(affected, config, root / "data", checkpoint),
    ) as executor:
        for rows in executor.map(_run_task, tasks):
            collected.extend(rows)
            pd.DataFrame(collected).to_csv(partial, index=False)
    frame = pd.DataFrame(collected)
    evaluation = results / "stage11_component_b_connectivity_evaluation.csv"
    frame.to_csv(evaluation, index=False)
    summaries = []
    endpoint_days = {
        "five_day": [8, 9, 10, 11, 12],
        "days_9_11": [9, 10, 11],
        "day_12": [12],
    }
    for q in CONNECTIVITY_LEVELS:
        level = frame[frame.communication_reliability.eq(q)]
        for endpoint, days in endpoint_days.items():
            selected = level[level.day.isin(days)]
            values = selected.groupby("seed").difference.sum()
            point, lower, upper, p_value = _safe_summary(values)
            summaries.append(
                {
                    "communication_reliability": q,
                    "endpoint": endpoint,
                    "point": point,
                    "lower": lower,
                    "upper": upper,
                    "p": p_value,
                    "n": len(values),
                    "learned_detours_per_day": float(level.learned_detours.mean()),
                    "fixed_ration_detours_per_day": float(
                        level.fixed_ration_detours.mean()
                    ),
                    "learned_cost_2_per_day": float(level.learned_cost_2.mean()),
                    "fixed_ration_cost_2_per_day": float(
                        level.fixed_ration_cost_2.mean()
                    ),
                    "scenario_alpha": POSTURE[0],
                    "scenario_gamma": POSTURE[1],
                    "phi_min": POSTURE[2],
                    "fleet": "A4_B4_C4",
                }
            )
    pd.DataFrame(summaries).to_csv(
        results / "stage11_component_b_connectivity_summary.csv", index=False
    )


if __name__ == "__main__":
    main()
