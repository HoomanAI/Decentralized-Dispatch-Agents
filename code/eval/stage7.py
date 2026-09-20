"""Stage 7 triage-rationed route-exploration experiments."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from code.demand.patients import generate_patients
from code.env.dispatch import build_runtime_network
from code.env.vehicle_centric import ProtocolMyopic, run_vehicle_episode
from code.eval.stage6_vehicle import _setup
from code.exploration.triage import ExplorationLedger, InformationRoutePolicy


def _run(
    affected: pd.DataFrame,
    config: dict[str, Any],
    data_dir: Path,
    budgets: tuple[float, float, float],
    information_weight: float,
    seeds: list[int],
    days: list[int],
    communication_reliability: float = 1.0,
    perfect_information: bool = False,
    reduced_instance: bool = True,
    day12_reveal_edges: set[int] | None = None,
    audit_records: list[dict[str, Any]] | None = None,
    communication_audit_records: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    if reduced_instance:
        pc, nc, dc, graph, hospitals, lookup, pds, union, age, env, nodes = _setup(affected, config, data_dir)
    else:
        pc = config["policy_training"]
        nc = config["network"]
        dc = config["demand"]
        graph, hospitals, lookup = build_runtime_network(
            data_dir / "synthetic_road_features.geojson",
            data_dir / "synthetic_fire_fronts.geojson",
            nc,
        )
        pds = affected.query("region == 'PDS'").set_index("day")
        union = float(pds["Total"].iloc[0])
        age = config["validation"]["expected_population_65_plus"] / config["validation"]["expected_population"]
        env = dict(config["environment"])
        fleet_per_class = int(env["fleet_per_class"])
        env["fleet_by_class"] = {"A": fleet_per_class, "B": fleet_per_class, "C": fleet_per_class}
        env["operational_minutes"] = float(dc["operational_minutes"])
        nodes = np.array([node for node, attributes in graph.nodes(data=True) if not attributes.get("hospital")])
    fleet_count = sum(int(value) for value in env["fleet_by_class"].values())
    exploration_enabled = information_weight > 0.0 and any(value > 0.0 for value in budgets)
    env.update({"exploration_enabled": exploration_enabled, "exploration_budgets": budgets, "exploration_k": 3, "exploration_detour_minutes": 20.0, "perfect_information": perfect_information, "day12_reveal_edges": day12_reveal_edges or set()})
    seed_base = int(pc["seed"]) if reduced_instance else int(config["environment"]["seed"])
    rows = []
    for seed in seeds:
        belief_evidence: list[tuple[np.ndarray, np.ndarray]] = []
        alternative_arcs: set[int] = set()
        for day in days:
            patient_seed = seed_base + 300000 + seed * 100 + day
            patients = generate_patients(day, float(pds.loc[day, "affected_population"]), union, nodes, age, dc, patient_seed)
            ledger = ExplorationLedger(budgets)
            controls = [InformationRoutePolicy(ledger, information_weight) for _ in range(fleet_count)]
            posterior: dict[str, Any] = {}
            communication_audit: dict[str, int] = {}
            summary, trace = run_vehicle_episode(graph, hospitals, lookup, patients, day, controls, nc, config["belief"], env, float(pds.loc[day, "affected_population"]), communication_reliability=communication_reliability, seed=patient_seed, belief_evidence=belief_evidence, alternative_route_audit=alternative_arcs if day <= 11 else None, posterior_audit=posterior if day == 12 else None, communication_audit=communication_audit if communication_audit_records is not None else None)
            if communication_audit_records is not None:
                communication_audit_records.append(
                    {
                        "seed": seed,
                        "day": day,
                        "communication_reliability": communication_reliability,
                        **communication_audit,
                    }
                )
            if audit_records is not None and day == 12:
                inverse_lookup = {index: edge_id for edge_id, index in lookup.items()}
                audit_records.append({"seed": seed, "communication_reliability": communication_reliability, "alternative_arcs": set(alternative_arcs), "mismatch_edge_ids": {int(inverse_lookup[index]) for index in posterior.get("mismatch_indices", set())}, "traffic_edge_ids": {int(edge_id) for item in trace for edge_id in item.get("arcs", [])}})
            detours = [item for item in trace if float(item["detour_distance_km"]) > 0.0]
            total_distance = sum(float(item["detour_distance_km"]) for item in detours)
            rows.append({"synthetic_input": True, "seed": seed, "day": day, "communication_reliability": communication_reliability, "perfect_information": perfect_information, "epsilon_1": budgets[0], "epsilon_2": budgets[1], "epsilon_3": budgets[2], "information_weight": information_weight, **summary, "exploration_cost_1": ledger.costs[1], "exploration_cost_2": ledger.costs[2], "exploration_cost_3": ledger.costs[3], "detour_count": len(detours), "mean_detour_distance_km": total_distance / max(len(detours), 1), "detour_distance_A": ledger.detour_distance["A"], "detour_distance_B": ledger.detour_distance["B"], "detour_distance_C": ledger.detour_distance["C"]})
    return pd.DataFrame(rows)


def _run_one_seed(args: tuple[Any, ...]) -> pd.DataFrame:
    """Pickle-safe worker for an independent Stage 7 rollout."""
    (
        affected,
        config,
        data_dir,
        budgets,
        information_weight,
        seed,
        days,
        communication_reliability,
        perfect_information,
        reduced_instance,
    ) = args
    return _run(
        affected,
        config,
        data_dir,
        budgets,
        information_weight,
        [seed],
        days,
        communication_reliability=communication_reliability,
        perfect_information=perfect_information,
        reduced_instance=reduced_instance,
    )


def _run_parallel(
    affected: pd.DataFrame,
    config: dict[str, Any],
    data_dir: Path,
    budgets: tuple[float, float, float],
    information_weight: float,
    seeds: list[int],
    days: list[int],
    communication_reliability: float,
    perfect_information: bool = False,
    reduced_instance: bool = False,
    max_workers: int = 8,
) -> pd.DataFrame:
    """Evaluate independent seeds concurrently and return deterministic seed order."""
    tasks = [
        (
            affected,
            config,
            data_dir,
            budgets,
            information_weight,
            seed,
            days,
            communication_reliability,
            perfect_information,
            reduced_instance,
        )
        for seed in seeds
    ]
    with ProcessPoolExecutor(max_workers=min(max_workers, len(tasks))) as executor:
        frames = list(executor.map(_run_one_seed, tasks))
    return pd.concat(frames, ignore_index=True)


def run_stage7(
    affected: pd.DataFrame,
    config: dict[str, Any],
    data_dir: Path,
    results_dir: Path,
) -> tuple[Path, Path]:
    """Tune the information scale and budgets, then run the paired 30-seed comparison."""
    calibration_seeds = list(range(5))
    training_days = [8, 9, 10, 11]
    scale_frames = []
    for weight in [1.0, 5.0, 10.0, 25.0, 50.0]:
        frame = _run(affected, config, data_dir, (np.inf, np.inf, np.inf), weight, calibration_seeds, training_days)
        scale_frames.append(frame)
    scale = pd.concat(scale_frames, ignore_index=True)
    scale_summary = scale.groupby("information_weight").agg(survival=("survival_total", "mean"), detours=("detour_count", "mean"), cost=("exploration_cost_1", "mean")).reset_index()
    eligible = scale_summary.query("detours > 0")
    information_weight = float((eligible if not eligible.empty else scale_summary).sort_values(["survival", "information_weight"], ascending=[False, True]).iloc[0]["information_weight"])

    budget_frames = []
    budget_values = [0.0, 0.1, 0.25, 0.5]
    for epsilon_2 in budget_values:
        for epsilon_3 in budget_values:
            budget_frames.append(_run(affected, config, data_dir, (0.0, epsilon_2, epsilon_3), information_weight, calibration_seeds, training_days))
    budget = pd.concat(budget_frames, ignore_index=True)
    budget_summary = budget.groupby(["epsilon_2", "epsilon_3"]).agg(survival=("survival_total", "mean"), type_1=("survival_type_1", "mean"), detours=("detour_count", "mean"), cost_2=("exploration_cost_2", "mean"), cost_3=("exploration_cost_3", "mean")).reset_index()
    selected = budget_summary.sort_values(
        ["survival", "epsilon_2", "epsilon_3"],
        ascending=[False, True, True],
    ).iloc[0]
    selected_budgets = (0.0, float(selected.epsilon_2), float(selected.epsilon_3))

    seeds = list(range(int(config["policy_training"]["evaluation_seeds"])))
    days = [8, 9, 10, 11, 12]
    exploitation = _run(affected, config, data_dir, (0.0, 0.0, 0.0), 0.0, seeds, days).assign(method="exploitation_only")
    unconstrained = _run(affected, config, data_dir, (np.inf, np.inf, np.inf), information_weight, seeds, days).assign(method="unconstrained")
    rationed = _run(affected, config, data_dir, selected_budgets, information_weight, seeds, days).assign(method="triage_rationed")
    evaluation = pd.concat([exploitation, unconstrained, rationed], ignore_index=True)

    budget_path = results_dir / "stage7_budget_sweep.csv"
    evaluation_path = results_dir / "stage7_evaluation.csv"
    scale.to_csv(results_dir / "stage7_information_scale_sweep.csv", index=False)
    budget.to_csv(budget_path, index=False)
    evaluation.to_csv(evaluation_path, index=False)
    return budget_path, evaluation_path


def run_route_set_audit(
    affected: pd.DataFrame,
    config: dict[str, Any],
    data_dir: Path,
    results_dir: Path,
) -> Path:
    """Record empirical route-set sizes on the complete Stage 7 evaluation panel."""
    seeds = list(range(int(config["policy_training"]["evaluation_seeds"])))
    frame = _run(
        affected,
        config,
        data_dir,
        (0.0, 0.0, 0.0),
        0.0,
        seeds,
        [8, 9, 10, 11, 12],
    )
    path = results_dir / "stage7_route_set_audit.csv"
    frame.to_csv(path, index=False)
    return path


def run_stage7_communication_cross(
    affected: pd.DataFrame,
    config: dict[str, Any],
    data_dir: Path,
    results_dir: Path,
) -> tuple[Path, Path]:
    """Cross the ration grid with communication reliability and evaluate each optimum."""
    reliabilities = [1.0, 0.75, 0.5, 0.25, 0.0]
    budgets = [0.0, 0.1, 0.25, 0.5]
    partial_sweep_path = results_dir / "stage7_communication_budget_sweep.partial.csv"
    if partial_sweep_path.exists():
        partial_sweep = pd.read_csv(partial_sweep_path)
        calibration = [partial_sweep]
        completed = set(
            zip(
                partial_sweep["communication_reliability"],
                partial_sweep["epsilon_2"],
                partial_sweep["epsilon_3"],
            )
        )
    else:
        calibration = []
        completed = set()
    for reliability in reliabilities:
        for epsilon_2 in budgets:
            for epsilon_3 in budgets:
                key = (reliability, epsilon_2, epsilon_3)
                if key in completed:
                    continue
                frame = _run(
                    affected,
                    config,
                    data_dir,
                    (0.0, epsilon_2, epsilon_3),
                    1.0,
                    list(range(5)),
                    [8, 9, 10, 11],
                    communication_reliability=reliability,
                )
                calibration.append(frame)
                pd.concat(calibration, ignore_index=True).to_csv(partial_sweep_path, index=False)
    sweep = pd.concat(calibration, ignore_index=True)
    means = sweep.groupby(["communication_reliability", "epsilon_2", "epsilon_3"], as_index=False).agg(survival=("survival_total", "mean"), detours=("detour_count", "mean"), cost_2=("exploration_cost_2", "mean"), cost_3=("exploration_cost_3", "mean"))
    selected = means.sort_values(["communication_reliability", "survival", "epsilon_2", "epsilon_3"], ascending=[False, False, True, True]).groupby("communication_reliability", as_index=False).first()
    positive_selected = (
        means.loc[(means["epsilon_2"] > 0.0) | (means["epsilon_3"] > 0.0)]
        .sort_values(
            ["communication_reliability", "survival", "epsilon_2", "epsilon_3"],
            ascending=[False, False, True, True],
        )
        .groupby("communication_reliability", as_index=False)
        .first()
        .set_index("communication_reliability")
    )
    partial_evaluation_path = results_dir / "stage7_communication_selected_evaluation.partial.csv"
    if partial_evaluation_path.exists():
        partial_evaluation = pd.read_csv(partial_evaluation_path)
        evaluations = [partial_evaluation]
        evaluated = set(
            zip(
                partial_evaluation["communication_reliability"],
                partial_evaluation["method"],
            )
        )
    else:
        evaluations = []
        evaluated = set()
    for row in selected.itertuples(index=False):
        common = {
            "affected": affected,
            "config": config,
            "data_dir": data_dir,
            "seeds": list(range(int(config["policy_training"]["evaluation_seeds"]))),
            "days": [8, 9, 10, 11, 12],
            "communication_reliability": float(row.communication_reliability),
        }
        selected_budgets = (0.0, float(row.epsilon_2), float(row.epsilon_3))
        selected_weight = 0.0 if selected_budgets == (0.0, 0.0, 0.0) else 1.0
        positive_row = positive_selected.loc[float(row.communication_reliability)]
        positive_budgets = (
            0.0,
            float(positive_row.epsilon_2),
            float(positive_row.epsilon_3),
        )
        arms = [
            ("triage_rationed", selected_budgets, selected_weight, False),
            ("best_positive_ration", positive_budgets, 1.0, False),
            ("belief_aware", (0.0, 0.0, 0.0), 0.0, False),
            ("perfect_information", (0.0, 0.0, 0.0), 0.0, True),
        ]
        for method, arm_budgets, arm_weight, perfect in arms:
            key = (float(row.communication_reliability), method)
            if key in evaluated:
                continue
            frame = _run(
                common["affected"], common["config"], common["data_dir"],
                arm_budgets,
                arm_weight,
                common["seeds"], common["days"],
                communication_reliability=common["communication_reliability"],
                perfect_information=perfect,
            ).assign(method=method)
            evaluations.append(frame)
            pd.concat(evaluations, ignore_index=True).to_csv(partial_evaluation_path, index=False)
    evaluation = pd.concat(evaluations, ignore_index=True)
    sweep_path = results_dir / "stage7_communication_budget_sweep.csv"
    evaluation_path = results_dir / "stage7_communication_selected_evaluation.csv"
    sweep.to_csv(sweep_path, index=False)
    evaluation.to_csv(evaluation_path, index=False)
    return sweep_path, evaluation_path


def run_stage7_full_cross(
    affected: pd.DataFrame,
    config: dict[str, Any],
    data_dir: Path,
    results_dir: Path,
) -> tuple[Path, Path]:
    """Full 1,297-arc rollout-only ration by communication experiment."""
    reliabilities = [1.0, 0.75, 0.5, 0.25, 0.0]
    budgets = [0.0, 0.05, 0.1]
    partial_sweep_path = results_dir / "stage7_full_communication_budget_sweep.partial.csv"
    frames = [pd.read_csv(partial_sweep_path)] if partial_sweep_path.exists() else []
    completed = set()
    if frames:
        completed = set(zip(frames[0]["communication_reliability"], frames[0]["epsilon_2"], frames[0]["epsilon_3"]))
    for reliability in reliabilities:
        for epsilon_2 in budgets:
            for epsilon_3 in budgets:
                key = (reliability, epsilon_2, epsilon_3)
                if key in completed:
                    continue
                frames.append(_run_parallel(affected, config, data_dir, (0.0, epsilon_2, epsilon_3), 1.0 if epsilon_2 + epsilon_3 > 0 else 0.0, list(range(3)), [8, 9, 10, 11], communication_reliability=reliability))
                pd.concat(frames, ignore_index=True).to_csv(partial_sweep_path, index=False)
    sweep = pd.concat(frames, ignore_index=True)
    means = sweep.groupby(["communication_reliability", "epsilon_2", "epsilon_3"], as_index=False).agg(survival=("survival_total", "mean"), detours=("detour_count", "mean"))
    selected = means.sort_values(["communication_reliability", "survival", "epsilon_2", "epsilon_3"], ascending=[False, False, True, True]).groupby("communication_reliability", as_index=False).first().set_index("communication_reliability")
    positive = means[(means.epsilon_2 > 0) | (means.epsilon_3 > 0)].sort_values(["communication_reliability", "survival", "epsilon_2", "epsilon_3"], ascending=[False, False, True, True]).groupby("communication_reliability", as_index=False).first().set_index("communication_reliability")

    partial_eval_path = results_dir / "stage7_full_communication_evaluation.partial.csv"
    evaluations = [pd.read_csv(partial_eval_path)] if partial_eval_path.exists() else []
    evaluated = set()
    if evaluations:
        evaluated = set(zip(evaluations[0]["communication_reliability"], evaluations[0]["method"]))
    for reliability in reliabilities:
        selected_row = selected.loc[reliability]
        positive_row = positive.loc[reliability]
        arms = [
            ("belief_aware", (0.0, 0.0, 0.0), 0.0, False),
            ("perfect_information", (0.0, 0.0, 0.0), 0.0, True),
            ("best_positive_ration", (0.0, float(positive_row.epsilon_2), float(positive_row.epsilon_3)), 1.0, False),
        ]
        for method, arm_budgets, weight, perfect in arms:
            if (reliability, method) in evaluated:
                continue
            frame = _run_parallel(affected, config, data_dir, arm_budgets, weight, list(range(30)), [8, 9, 10, 11, 12], communication_reliability=reliability, perfect_information=perfect).assign(method=method)
            evaluations.append(frame)
            pd.concat(evaluations, ignore_index=True).to_csv(partial_eval_path, index=False)
    evaluation = pd.concat(evaluations, ignore_index=True)
    sweep_path = results_dir / "stage7_full_communication_budget_sweep.csv"
    evaluation_path = results_dir / "stage7_full_communication_evaluation.csv"
    sweep.to_csv(sweep_path, index=False)
    evaluation.to_csv(evaluation_path, index=False)
    return sweep_path, evaluation_path


def run_stage7_communication_replication(
    affected: pd.DataFrame,
    config: dict[str, Any],
    data_dir: Path,
    results_dir: Path,
) -> Path:
    """Run the pre-specified fresh-seed zero-communication replication."""
    partial_path = results_dir / "stage7_communication_replication.partial.csv"
    final_path = results_dir / "stage7_communication_replication.csv"
    frames = [pd.read_csv(partial_path)] if partial_path.exists() else []
    completed: set[tuple[str, int]] = set()
    if frames:
        completed = set(zip(frames[0]["method"], frames[0]["seed"]))
    arms = [
        ("belief_aware", (0.0, 0.0, 0.0), 0.0),
        ("positive_ration", (0.0, 0.05, 0.0), 1.0),
    ]
    for method, budgets, information_weight in arms:
        pending = [seed for seed in range(30, 90) if (method, seed) not in completed]
        for start in range(0, len(pending), 8):
            batch = pending[start : start + 8]
            frame = _run_parallel(
                affected,
                config,
                data_dir,
                budgets,
                information_weight,
                batch,
                [8, 9, 10, 11],
                communication_reliability=0.0,
            ).assign(method=method)
            frames.append(frame)
            pd.concat(frames, ignore_index=True).to_csv(partial_path, index=False)
    result = pd.concat(frames, ignore_index=True)
    expected = {(method, seed) for method, _, _ in arms for seed in range(30, 90)}
    actual = set(zip(result["method"], result["seed"]))
    if actual != expected:
        raise AssertionError("Replication output does not contain the pre-specified arms and seeds")
    result.to_csv(final_path, index=False)
    return final_path
