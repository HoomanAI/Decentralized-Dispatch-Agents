"""Relate vehicle-centric feasible action counts to locally feasible queue size."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import spearmanr

from code.demand.patients import generate_patients
from code.env.vehicle_centric import run_vehicle_episode
from code.eval.stage6_vehicle import _setup
from code.policy.vehicle_centric import NeuralVehicleController, VehicleActorQCritic


class QueueAuditController(NeuralVehicleController):
    """Record the patient choices represented in each allowed action set."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.queue_records: list[dict[str, int]] = []

    def propose(self, context: dict[str, object]) -> tuple[int, float]:
        allowed = list(context["allowed_indices"])
        candidates = context["candidates"]
        patient_ids = {
            candidates[index].patient_id
            for index in allowed
            if candidates[index].patient_id is not None
        }
        self.queue_records.append(
            {
                "locally_feasible_patients": len(patient_ids),
                "feasible_action_count": len(allowed),
            }
        )
        return super().propose(context)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load((root / "code/config/default.yaml").read_text(encoding="utf-8"))
    affected = pd.read_csv(root / "data/daily_affected_population_clean.csv")
    pc, nc, dc, graph, hospitals, lookup, pds, union, age, env, nodes = _setup(
        affected, config, root / "data"
    )
    fleet_count = sum(int(value) for value in pc["fleet_by_class"].values())
    edge_count = graph.number_of_edges()
    checkpoint = torch.load(root / "results/stage6_vehicle_clone.pt", map_location="cpu", weights_only=False)
    model = VehicleActorQCritic(edge_count, int(pc["hidden_dim"]), fleet_count)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    rows: list[dict[str, int]] = []
    audit_seeds = min(3, int(pc["evaluation_seeds"]))
    for seed in range(audit_seeds):
        for day in range(8, 13):
            patient_seed = int(pc["seed"]) + 300000 + seed * 100 + day
            patients = generate_patients(
                day,
                float(pds.loc[day, "affected_population"]),
                union,
                nodes,
                age,
                dc,
                patient_seed,
            )
            controls = [
                QueueAuditController(model, edge_count, greedy=True)
                for _ in range(fleet_count)
            ]
            run_vehicle_episode(
                graph,
                hospitals,
                lookup,
                patients,
                day,
                controls,
                nc,
                config["belief"],
                env,
                float(pds.loc[day, "affected_population"]),
                seed=patient_seed,
            )
            for control in controls:
                rows.extend(
                    {"seed": seed, "day": day, **record}
                    for record in control.queue_records
                )
    detail = pd.DataFrame(rows)
    detail.to_csv(root / "results/forced_epoch_queue_audit.csv", index=False)
    binned = (
        detail.groupby("locally_feasible_patients", as_index=False)
        .agg(
            epochs=("feasible_action_count", "size"),
            mean_feasible_action_count=("feasible_action_count", "mean"),
            median_feasible_action_count=("feasible_action_count", "median"),
            forced_epoch_fraction=("feasible_action_count", lambda values: float(np.mean(values == 1))),
        )
    )
    binned.to_csv(root / "results/forced_epoch_queue_binned.csv", index=False)
    correlation = spearmanr(
        detail["locally_feasible_patients"], detail["feasible_action_count"]
    )
    pd.DataFrame(
        [
            {
                "variant": "Supervised clone",
                "queue_measure": "distinct locally feasible patients in allowed action set",
                "epochs": len(detail),
                "seeds": audit_seeds,
                "spearman_rho": float(correlation.statistic),
                "p_value": float(correlation.pvalue),
                "from_scratch_comparable": False,
                "from_scratch_reason": "From-scratch actions select vehicles for one current patient, while vehicle-centric actions select patients for one vehicle.",
            }
        ]
    ).to_csv(root / "results/forced_epoch_queue_summary.csv", index=False)


if __name__ == "__main__":
    main()
