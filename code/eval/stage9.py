"""Run full-instance Stage 9 baseline comparisons and produce T7 and F14."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
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


def _configuration(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    network = config["network"]
    demand = config["demand"]
    environment = dict(config["environment"])
    posture = config["policy_training"]
    environment.update(
        {
            "scenario_alpha": float(posture["scenario_alpha"]),
            "scenario_gamma": float(posture["scenario_gamma"]),
            "phi_min": float(posture["phi_min"]),
            "fleet_by_class": {"A": 4, "B": 4, "C": 4},
            "operational_minutes": float(demand["operational_minutes"]),
            "exploration_enabled": False,
            "exploration_budgets": (0.0, 0.0, 0.0),
        }
    )
    return network, demand, environment


def _one_seed(args: tuple[Any, ...]) -> list[dict[str, Any]]:
    affected, config, data_dir, seed, methods, tuned = args
    network, demand, environment = _configuration(config)
    graph, hospitals, lookup = build_runtime_network(
        Path(data_dir) / "synthetic_road_features.geojson",
        Path(data_dir) / "synthetic_fire_fronts.geojson",
        network,
    )
    assert graph.number_of_edges() - len(hospitals) == 1297
    pds = affected.query("region == 'PDS'").set_index("day")
    union = float(pds["Total"].iloc[0])
    age = config["validation"]["expected_population_65_plus"] / config["validation"]["expected_population"]
    nodes = np.asarray([n for n, a in graph.nodes(data=True) if not a.get("hospital")])
    vehicle_count = sum(environment["fleet_by_class"].values())
    rows: list[dict[str, Any]] = []
    evidence_by_method: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        method: [] for method in methods
    }
    for day in range(8, 13):
        patient_seed = int(config["environment"]["seed"]) + 300000 + seed * 100 + day
        patients = generate_patients(day, float(pds.loc[day, "affected_population"]), union, nodes, age, demand, patient_seed)
        for method in methods:
            local_env = dict(environment)
            if method == "reliability_blind_myopic":
                local_env["belief_blind"] = True
            if method == "protocol_myopic" or method == "reliability_blind_myopic":
                policies = [ProtocolMyopic() for _ in range(vehicle_count)]
            elif method in ("ga", "alns"):
                policies = make_policies(method, vehicle_count, np.asarray(tuned[method], dtype=float))
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
                local_env,
                float(pds.loc[day, "affected_population"]),
                communication_reliability=1.0,
                seed=patient_seed,
                belief_evidence=evidence_by_method[method],
            )
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "day": day,
                    "survival_total": float(summary["survival_total"]),
                    "patients": int(summary["patients"]),
                    "served": int(summary["served"]),
                    "instance": "full_1297_arc",
                    "fleet": 12,
                    "fleet_A": 4,
                    "fleet_B": 4,
                    "fleet_C": 4,
                    "alpha": environment["scenario_alpha"],
                    "gamma": environment["scenario_gamma"],
                    "phi_min": environment["phi_min"],
                    "communication_reliability": 1.0,
                }
            )
    return rows


def _objective(frame: pd.DataFrame) -> float:
    return float(frame.groupby(["seed", "method"])["survival_total"].sum().mean())


def _evaluate_weights(
    affected: pd.DataFrame,
    config: dict[str, Any],
    data_dir: Path,
    method: str,
    weights: np.ndarray,
    seeds: list[int],
) -> float:
    tuned = {method: weights.tolist()}
    tasks = [(affected, config, data_dir, seed, (method,), tuned) for seed in seeds]
    with ProcessPoolExecutor(max_workers=min(4, len(tasks))) as pool:
        rows = [row for batch in pool.map(_one_seed, tasks) for row in batch]
    return _objective(pd.DataFrame(rows))


def _tune(affected: pd.DataFrame, config: dict[str, Any], data_dir: Path, results_dir: Path) -> dict[str, list[float]]:
    """Tune GA and ALNS score vectors on three calibration seeds."""
    rng = np.random.default_rng(202609)
    seeds = [90, 91, 92]
    base = np.asarray([1.0, 0.25, 0.25, 0.10, 0.0])
    records: list[dict[str, Any]] = []
    tuned: dict[str, list[float]] = {}
    for method in ("ga", "alns"):
        incumbent = base.copy()
        incumbent_value = _evaluate_weights(affected, config, data_dir, method, incumbent, seeds)
        records.append({"method": method, "iteration": 0, "objective": incumbent_value, **{f"w_{i}": value for i, value in enumerate(incumbent)}})
        for iteration in range(1, 9):
            if method == "ga":
                candidate = np.clip(incumbent + rng.normal(0.0, 0.35, len(base)), -2.0, 2.0)
            else:
                destroy = rng.choice(len(base), size=2, replace=False)
                candidate = incumbent.copy()
                candidate[destroy] = rng.uniform(-1.0, 1.5, len(destroy))
            value = _evaluate_weights(affected, config, data_dir, method, candidate, seeds)
            records.append({"method": method, "iteration": iteration, "objective": value, **{f"w_{i}": item for i, item in enumerate(candidate)}})
            if value > incumbent_value:
                incumbent, incumbent_value = candidate, value
        tuned[method] = incumbent.tolist()
    pd.DataFrame(records).to_csv(results_dir / "stage9_metaheuristic_tuning.csv", index=False)
    return tuned


def _paired_summary(raw: pd.DataFrame) -> pd.DataFrame:
    totals = raw.groupby(["method", "seed"], as_index=False)["survival_total"].sum().rename(columns={"survival_total": "five_day"})
    day12 = raw.query("day == 12")[["method", "seed", "survival_total"]].rename(columns={"survival_total": "day12"})
    values = totals.merge(day12, on=["method", "seed"])
    reference = values.query("method == 'protocol_myopic'").set_index("seed")
    rows = []
    for method, group in values.groupby("method", sort=False):
        group = group.set_index("seed").sort_index()
        for endpoint in ("five_day", "day12"):
            difference = group[endpoint] - reference.loc[group.index, endpoint]
            n = len(difference)
            mean = float(difference.mean())
            se = float(stats.sem(difference)) if n > 1 else 0.0
            critical = float(stats.t.ppf(0.975, n - 1)) if n > 1 else 0.0
            p = float(stats.ttest_1samp(difference, 0.0).pvalue) if n > 1 and difference.std(ddof=1) > 0 else (1.0 if mean == 0.0 else 0.0)
            rows.append(
                {
                    "method": method,
                    "endpoint": endpoint,
                    "survival_mean": float(group[endpoint].mean()),
                    "paired_difference": mean,
                    "ci_lower": mean - critical * se,
                    "ci_upper": mean + critical * se,
                    "p_value": p,
                    "n": n,
                    "instance": "full_1297_arc",
                    "fleet": "A:4 B:4 C:4",
                    "posture": "alpha=0.75 gamma=0.75 phi_min=0.70",
                    "communication_reliability": 1.0,
                }
            )
    return pd.DataFrame(rows)


def _figure(summary: pd.DataFrame, output: Path) -> None:
    order = [method for method in METHODS if method != "protocol_myopic"]
    labels = ["Blind", "Yan", "Ahmadi", "Peng", "GA", "ALNS"]
    colors = ["#D55E00", "#0072B2", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.5), sharey=True)
    for axis, endpoint, title in zip(axes, ("five_day", "day12"), ("Five-day", "Day 12")):
        table = summary.query("endpoint == @endpoint").set_index("method").loc[order]
        y = np.arange(len(order))
        means = table["paired_difference"].to_numpy()
        errors = np.vstack((means - table["ci_lower"].to_numpy(), table["ci_upper"].to_numpy() - means))
        axis.errorbar(means, y, xerr=errors, fmt="none", ecolor="#444444", capsize=2, linewidth=1)
        axis.scatter(means, y, c=colors, marker="o", s=28, edgecolor="black", linewidth=0.4, zorder=3)
        axis.axvline(0.0, color="black", linestyle="--", linewidth=0.8)
        axis.set_title(title, fontsize=9)
        axis.set_xlabel("Paired survival difference", fontsize=9)
        axis.tick_params(labelsize=8)
        axis.grid(axis="x", color="#dddddd", linewidth=0.5)
    axes[0].set_yticks(np.arange(len(labels)), labels, fontsize=8)
    axes[0].text(-0.16, 1.03, "(a)", transform=axes[0].transAxes, fontsize=9)
    axes[1].text(-0.16, 1.03, "(b)", transform=axes[1].transAxes, fontsize=9)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def run_stage9(project_root: Path) -> tuple[Path, Path, Path]:
    config = yaml.safe_load((project_root / "code" / "config" / "default.yaml").read_text(encoding="utf-8"))
    affected = pd.read_csv(project_root / "data" / "daily_affected_population_clean.csv")
    data_dir = project_root / "data"
    results_dir = project_root / "results"
    tuned = _tune(affected, config, data_dir, results_dir)
    partial = results_dir / "stage9_baselines.partial.csv"
    existing = pd.read_csv(partial) if partial.exists() else pd.DataFrame()
    completed = set(existing["seed"].unique()) if not existing.empty else set()
    frames = [existing] if not existing.empty else []
    pending = [seed for seed in range(30) if seed not in completed]
    for start in range(0, len(pending), 4):
        batch = pending[start : start + 4]
        tasks = [(affected, config, data_dir, seed, METHODS, tuned) for seed in batch]
        with ProcessPoolExecutor(max_workers=len(tasks)) as pool:
            rows = [row for result in pool.map(_one_seed, tasks) for row in result]
        frames.append(pd.DataFrame(rows))
        pd.concat(frames, ignore_index=True).to_csv(partial, index=False)
    raw = pd.concat(frames, ignore_index=True).sort_values(["method", "seed", "day"])
    raw_path = results_dir / "stage9_baselines.csv"
    table_path = results_dir / "T7_stage9_baselines.csv"
    figure_path = project_root / "figures" / "F14_baseline_comparison"
    raw.to_csv(raw_path, index=False)
    summary = _paired_summary(raw)
    summary.to_csv(table_path, index=False)
    _figure(summary, figure_path)
    return raw_path, table_path, figure_path.with_suffix(".pdf")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    print(*(str(path) for path in run_stage9(root)), sep="\n")


if __name__ == "__main__":
    main()
