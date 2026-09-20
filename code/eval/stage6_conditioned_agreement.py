"""Re-audit Stage 6 agreement conditional on a genuine action choice."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from code.demand.patients import generate_patients
from code.env.dispatch import build_runtime_network, run_episode
from code.env.vehicle_centric import run_vehicle_episode
from code.eval.stage6_vehicle import _setup
from code.policy.mappo import PPOController
from code.policy.model import ExploitationActorCritic
from code.policy.vehicle_centric import (
    NeuralVehicleController,
    ResidualVehiclePolicy,
    VehicleActorQCritic,
)


def _summary(variant: str, counts: list[int], agreements: list[bool]) -> dict[str, Any]:
    nontrivial = [agreement for count, agreement in zip(counts, agreements) if count >= 2]
    distribution = Counter(counts)
    return {
        "variant": variant,
        "decision_epochs": len(counts),
        "forced_epochs": int(distribution.get(1, 0)),
        "forced_epoch_fraction": float(distribution.get(1, 0) / len(counts)),
        "all_epoch_argmax_agreement": float(np.mean(agreements)),
        "nontrivial_epochs": len(nontrivial),
        "nontrivial_argmax_agreement": float(np.mean(nontrivial)),
        "minimum_feasible_set_size": min(counts),
        "maximum_feasible_set_size": max(counts),
    }


def _from_scratch(
    affected: pd.DataFrame, config: dict[str, Any], data_dir: Path, results_dir: Path
) -> tuple[dict[str, Any], Counter[int]]:
    pc, nc, dc = config["policy_training"], config["reduced_network"], config["reduced_demand"]
    graph, hospitals, lookup = build_runtime_network(
        data_dir / "reduced/synthetic_road_features.geojson",
        data_dir / "reduced/synthetic_fire_fronts.geojson",
        nc,
    )
    road_classes = sorted({data["fclass"] for _, _, data in graph.edges(data=True)})
    model = ExploitationActorCritic(3, 3 + len(road_classes), 6, 6, int(pc["hidden_dim"]))
    saved = torch.load(results_dir / "stage6_exploitation_only.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    pds = affected.query("region == 'PDS'").set_index("day")
    union = float(pds["Total"].iloc[0])
    age = config["validation"]["expected_population_65_plus"] / config["validation"]["expected_population"]
    env = dict(config["environment"])
    env.update({
        "scenario_alpha": float(pc["scenario_alpha"]),
        "scenario_gamma": float(pc["scenario_gamma"]),
        "phi_min": float(pc["phi_min"]),
        "scenario_alpha_concentration": pc["scenario_alpha_concentration"],
        "operational_minutes": float(dc["operational_minutes"]),
        "fleet_by_class": dict(pc["fleet_by_class"]),
    })
    nodes = np.asarray([n for n, a in graph.nodes(data=True) if not a.get("hospital")])
    counts: list[int] = []
    agreements: list[bool] = []
    for seed in range(int(pc["evaluation_seeds"])):
        for day in range(8, 13):
            patient_seed = int(pc["seed"]) + 100000 + 100 * seed + day
            patients = generate_patients(
                day, float(pds.loc[day, "affected_population"]), union, nodes, age, dc,
                patient_seed,
            )
            controller = PPOController(model, lookup, road_classes, training=False, seed=int(pc["seed"]) + seed)
            run_episode(
                graph, hospitals, lookup, patients, day, "belief_aware", nc,
                config["belief"], env, float(pds.loc[day, "affected_population"]),
                controller=controller,
            )
            counts.extend(int(row["feasible_action_count"]) for row in controller.decision_records)
            agreements.extend(bool(row["agrees_with_myopic"]) for row in controller.decision_records)
    return _summary("From scratch", counts, agreements), Counter(counts)


def _vehicle_variants(
    affected: pd.DataFrame, config: dict[str, Any], data_dir: Path, results_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, Counter[int]]]:
    pc, nc, dc, graph, hospitals, lookup, pds, union, age, env, nodes = _setup(
        affected, config, data_dir
    )
    fleet_count = sum(int(value) for value in pc["fleet_by_class"].values())
    edge_count = graph.number_of_edges()
    clone_saved = torch.load(results_dir / "stage6_vehicle_clone.pt", map_location="cpu", weights_only=False)
    clone = VehicleActorQCritic(edge_count, int(pc["hidden_dim"]), fleet_count)
    clone.load_state_dict(clone_saved["model_state_dict"])
    clone.eval()
    specifications: list[tuple[str, torch.nn.Module]] = [("Supervised clone", clone)]
    for label, name in [
        ("Residual warm start, scalar critic", "scalar_matched"),
        ("Residual warm start, counterfactual critic", "counterfactual"),
        ("Softened clone, counterfactual critic", "counterfactual_softened"),
    ]:
        base = VehicleActorQCritic(edge_count, int(pc["hidden_dim"]), fleet_count)
        base.load_state_dict(clone_saved["model_state_dict"])
        model = ResidualVehiclePolicy(base)
        saved = torch.load(results_dir / f"stage6_vehicle_{name}.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model_state_dict"])
        model.eval()
        specifications.append((label, model))

    rows: list[dict[str, Any]] = []
    distributions: dict[str, Counter[int]] = {}
    for label, model in specifications:
        counts: list[int] = []
        agreements: list[bool] = []
        for seed in range(int(pc["evaluation_seeds"])):
            for day in range(8, 13):
                patient_seed = int(pc["seed"]) + 300000 + seed * 100 + day
                patients = generate_patients(
                    day, float(pds.loc[day, "affected_population"]), union, nodes, age, dc,
                    patient_seed,
                )
                controls = [
                    NeuralVehicleController(model, edge_count, greedy=True)
                    for _ in range(fleet_count)
                ]
                run_vehicle_episode(
                    graph, hospitals, lookup, patients, day, controls, nc,
                    config["belief"], env, float(pds.loc[day, "affected_population"]),
                    seed=patient_seed,
                )
                counts.extend(count for control in controls for count in control.allowed_action_counts)
                agreements.extend(value for control in controls for value in control.myopic_agreements)
        rows.append(_summary(label, counts, agreements))
        distributions[label] = Counter(counts)
    return rows, distributions


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load((root / "code/config/default.yaml").read_text(encoding="utf-8"))
    affected = pd.read_csv(root / "data/daily_affected_population_clean.csv")
    scratch, scratch_distribution = _from_scratch(affected, config, root / "data", root / "results")
    vehicle_rows, vehicle_distributions = _vehicle_variants(
        affected, config, root / "data", root / "results"
    )
    rows = [scratch, *vehicle_rows]
    pd.DataFrame(rows).to_csv(root / "results/stage6_conditioned_agreement.csv", index=False)
    distributions = {"From scratch": scratch_distribution, **vehicle_distributions}
    distribution_rows = [
        {"variant": variant, "feasible_set_size": size, "epochs": epochs}
        for variant, distribution in distributions.items()
        for size, epochs in sorted(distribution.items())
    ]
    pd.DataFrame(distribution_rows).to_csv(
        root / "results/stage6_feasible_set_size_distribution.csv", index=False
    )


if __name__ == "__main__":
    main()
