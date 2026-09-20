"""Tests 8 and 9: policy-interface parity and supervised myopic cloning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional

from code.demand.patients import generate_patients
from code.env.dispatch import build_runtime_network, run_episode
from code.eval.stage6 import _perturb_exposure
from code.policy.mappo import PPOController
from code.policy.model import ExploitationActorCritic


class PipelineMyopicController:
    """Pass the myopic action through the controller interface."""

    def select(self, **context: Any) -> int:
        return min(
            range(len(context["options"])),
            key=lambda index: context["options"][index][0],
        )

    def observe(self, reward: float) -> None:
        del reward

    def finish_episode(self) -> None:
        return None


class MyopicTeacher(PPOController):
    """Collect actor states labeled with the myopic action."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.samples: list[tuple[dict[str, torch.Tensor], int]] = []

    def select(self, **context: Any) -> int:
        state = self._state(**context)
        action = min(
            range(len(context["options"])),
            key=lambda index: context["options"][index][0],
        )
        self.samples.append((state, action))
        return action

    def observe(self, reward: float) -> None:
        del reward

    def finish_episode(self) -> None:
        return None


def run_stage6_clone(
    affected_population: pd.DataFrame,
    config: dict[str, Any],
    data_dir: Path,
    results_dir: Path,
    training_episodes: int = 1000,
) -> tuple[Path, Path, Path]:
    """Run interface parity and train a supervised clone of myopic dispatch."""
    pc = config["policy_training"]
    nc = config["reduced_network"]
    dc = config["reduced_demand"]
    graph, hospitals, edge_lookup = build_runtime_network(
        data_dir / "reduced" / "synthetic_road_features.geojson",
        data_dir / "reduced" / "synthetic_fire_fronts.geojson",
        nc,
    )
    road_classes = sorted({data["fclass"] for _, _, data in graph.edges(data=True)})
    torch.manual_seed(int(pc["seed"]) + 900000)
    model = ExploitationActorCritic(3, 3 + len(road_classes), 6, 6, int(pc["hidden_dim"]))
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    pds = affected_population.query("region == 'PDS'").set_index("day")
    union_population = float(pds["Total"].iloc[0])
    age_share = config["validation"]["expected_population_65_plus"] / config["validation"]["expected_population"]
    environment = dict(config["environment"])
    environment.update({
        "scenario_alpha": float(pc["scenario_alpha"]),
        "scenario_gamma": float(pc["scenario_gamma"]),
        "phi_min": float(pc["phi_min"]),
        "scenario_alpha_concentration": pc["scenario_alpha_concentration"],
        "operational_minutes": float(dc["operational_minutes"]),
        "fleet_by_class": dict(pc["fleet_by_class"]),
    })
    nodes = np.array([n for n, a in graph.nodes(data=True) if not a.get("hospital")])
    modes = list(pc["augmentation_modes"])
    augmented = {mode: _perturb_exposure(graph, data_dir / "reduced" / "synthetic_fire_fronts.geojson", mode, float(pc["augmentation_buffer_degrees"]), float(pc["augmentation_translation_degrees"])) for mode in set(modes)}
    log_rows = []
    for episode in range(training_episodes):
        day = int(pc["train_days"][episode % len(pc["train_days"])])
        mode = modes[episode % len(modes)]
        train_graph = augmented[mode]
        train_lookup = {int(a["edge_id"]): i for i, (_, _, a) in enumerate(train_graph.edges(data=True))}
        patients = generate_patients(day, float(pds.loc[day, "affected_population"]), union_population, nodes, age_share, dc, int(pc["seed"]) + 200000 + episode)
        teacher = MyopicTeacher(model, train_lookup, road_classes, training=False, seed=episode)
        run_episode(train_graph, hospitals, train_lookup, patients, day, "belief_aware", nc, config["belief"], environment, float(pds.loc[day, "affected_population"]), controller=teacher)
        if not teacher.samples:
            continue
        losses = []
        correct = 0
        entropies = []
        for state, label in teacher.samples:
            logits, _ = model(**state)
            target = torch.tensor([label], dtype=torch.long)
            losses.append(functional.cross_entropy(logits.unsqueeze(0), target))
            correct += int(torch.argmax(logits).item() == label)
            entropies.append(float(torch.distributions.Categorical(logits=logits).entropy().item()))
        loss = torch.stack(losses).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        log_rows.append({"episode": episode + 1, "examples": len(teacher.samples), "cross_entropy": float(loss.item()), "training_agreement": correct / len(teacher.samples), "entropy": float(np.mean(entropies))})

    evaluation_rows = []
    for seed in range(int(pc["evaluation_seeds"])):
        totals = {"direct_myopic": 0.0, "pipeline_myopic": 0.0, "clone": 0.0}
        agreements = 0
        decisions = 0
        entropies = []
        for day in range(8, 13):
            patient_seed = int(pc["seed"]) + 100000 + 100 * seed + day
            patients = generate_patients(day, float(pds.loc[day, "affected_population"]), union_population, nodes, age_share, dc, patient_seed)
            direct, _ = run_episode(graph, hospitals, edge_lookup, patients, day, "belief_aware", nc, config["belief"], environment, float(pds.loc[day, "affected_population"]))
            pipeline, _ = run_episode(graph, hospitals, edge_lookup, patients, day, "belief_aware", nc, config["belief"], environment, float(pds.loc[day, "affected_population"]), controller=PipelineMyopicController())
            clone = PPOController(model, edge_lookup, road_classes, training=False, seed=patient_seed)
            cloned, _ = run_episode(graph, hospitals, edge_lookup, patients, day, "belief_aware", nc, config["belief"], environment, float(pds.loc[day, "affected_population"]), controller=clone)
            totals["direct_myopic"] += float(direct["survival_total"])
            totals["pipeline_myopic"] += float(pipeline["survival_total"])
            totals["clone"] += float(cloned["survival_total"])
            agreements += sum(clone.myopic_agreements)
            decisions += len(clone.myopic_agreements)
            entropies.extend(clone.decision_normalized_entropies)
        evaluation_rows.append({"synthetic_input": True, "evaluation_seed": seed, **totals, "pipeline_minus_direct": totals["pipeline_myopic"] - totals["direct_myopic"], "clone_minus_direct": totals["clone"] - totals["direct_myopic"], "clone_agreement_rate": agreements / decisions, "clone_normalized_entropy": float(np.mean(entropies))})
    log_path = results_dir / "stage6_clone_training_log.csv"
    evaluation_path = results_dir / "stage6_clone_evaluation.csv"
    checkpoint_path = results_dir / "stage6_myopic_clone.pt"
    pd.DataFrame(log_rows).to_csv(log_path, index=False)
    pd.DataFrame(evaluation_rows).to_csv(evaluation_path, index=False)
    torch.save({"model_state_dict": model.state_dict(), "training_episodes": training_episodes, "synthetic_input": True}, checkpoint_path)
    return log_path, evaluation_path, checkpoint_path
