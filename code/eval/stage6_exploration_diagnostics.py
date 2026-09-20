"""Diagnostics for exploration around the vehicle-centric cloned policy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.distributions import Categorical

from code.demand.patients import generate_patients
from code.env.vehicle_centric import run_vehicle_episode
from code.eval.stage6_vehicle import _setup
from code.policy.vehicle_centric import NeuralVehicleController, VehicleActorQCritic, vehicle_state


@dataclass
class DeviationClock:
    target: int
    counter: int = 0
    applied: bool = False


class SingleDeviationController(NeuralVehicleController):
    """Follow the clone greedily except for one shared randomly selected proposal."""

    def __init__(self, model, edge_count: int, clock: DeviationClock, rng: np.random.Generator) -> None:
        super().__init__(model, edge_count, greedy=True)
        self.clock = clock
        self.rng = rng

    def propose(self, context: dict[str, Any]) -> tuple[int, float]:
        state = vehicle_state(context, self.edge_count)
        allowed = list(context["allowed_indices"])
        logits, q = self.model(state)
        masked = torch.full_like(logits, -torch.inf)
        masked[allowed] = logits[allowed]
        distribution = Categorical(logits=masked)
        greedy_action = int(torch.argmax(masked).item())
        action = greedy_action
        if self.clock.counter == self.clock.target and len(allowed) > 1:
            alternatives = [item for item in allowed if item != greedy_action]
            action = int(self.rng.choice(alternatives))
            self.clock.applied = True
        self.clock.counter += 1
        chosen = torch.tensor(action)
        self.pending = (state, action, distribution.log_prob(chosen).detach(), q.detach())
        return action, float(distribution.probs[action].item())


class FlexibleIndexController(NeuralVehicleController):
    """Record shared proposal indices at which a non-greedy action exists."""

    def __init__(self, model, edge_count: int, clock: DeviationClock, flexible: list[int]) -> None:
        super().__init__(model, edge_count, greedy=True)
        self.clock = clock
        self.flexible = flexible

    def propose(self, context: dict[str, Any]) -> tuple[int, float]:
        if len(context["allowed_indices"]) > 1:
            self.flexible.append(self.clock.counter)
        self.clock.counter += 1
        return super().propose(context)


def _clone(config: dict[str, Any], results_dir: Path, edge_count: int, fleet_count: int) -> VehicleActorQCritic:
    pc = config["policy_training"]
    saved = torch.load(results_dir / "stage6_vehicle_clone.pt", map_location="cpu", weights_only=False)
    model = VehicleActorQCritic(edge_count, int(pc["hidden_dim"]), fleet_count)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    return model


def run_forced_deviation_test(
    affected: pd.DataFrame,
    config: dict[str, Any],
    data_dir: Path,
    results_dir: Path,
    deviations: int = 300,
) -> Path:
    """Measure one forced non-greedy proposal against its paired clone day."""
    pc, nc, dc, graph, hospitals, lookup, pds, union, age, env, nodes = _setup(affected, config, data_dir)
    fleet_count = sum(int(value) for value in pc["fleet_by_class"].values())
    edge_count = graph.number_of_edges()
    model = _clone(config, results_dir, edge_count, fleet_count)
    rng = np.random.default_rng(int(pc["seed"]) + 880000)
    baseline: dict[tuple[int, int], tuple[float, list[int]]] = {}
    for seed in range(int(pc["evaluation_seeds"])):
        for day in range(8, 13):
            patient_seed = int(pc["seed"]) + 300000 + seed * 100 + day
            patients = generate_patients(day, float(pds.loc[day, "affected_population"]), union, nodes, age, dc, patient_seed)
            clock = DeviationClock(-1)
            flexible: list[int] = []
            controls = [FlexibleIndexController(model, edge_count, clock, flexible) for _ in range(fleet_count)]
            summary, _ = run_vehicle_episode(graph, hospitals, lookup, patients, day, controls, nc, config["belief"], env, float(pds.loc[day, "affected_population"]), seed=patient_seed)
            baseline[(seed, day)] = (float(summary["survival_total"]), flexible)
    eligible = [key for key, (_, flexible) in baseline.items() if flexible]
    rows = []
    while len(rows) < deviations:
        seed, day = eligible[int(rng.integers(len(eligible)))]
        base_survival, flexible = baseline[(seed, day)]
        target = int(rng.choice(flexible))
        patient_seed = int(pc["seed"]) + 300000 + seed * 100 + day
        patients = generate_patients(day, float(pds.loc[day, "affected_population"]), union, nodes, age, dc, patient_seed)
        clock = DeviationClock(target)
        controls = [SingleDeviationController(model, edge_count, clock, rng) for _ in range(fleet_count)]
        summary, _ = run_vehicle_episode(graph, hospitals, lookup, patients, day, controls, nc, config["belief"], env, float(pds.loc[day,"affected_population"]), seed=patient_seed)
        if not clock.applied:
            raise AssertionError("Recorded flexible proposal was not reached in paired replay")
        delta = float(summary["survival_total"]) - base_survival
        rows.append({"synthetic_input": True, "replicate": len(rows), "seed": seed, "day": day, "proposal_index": target, "baseline_day_survival": base_survival, "deviation_day_survival": float(summary["survival_total"]), "delta_five_day_survival": delta})
    path = results_dir / "stage6_vehicle_forced_deviations.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def run_temperature_sweep(
    affected: pd.DataFrame,
    config: dict[str, Any],
    data_dir: Path,
    results_dir: Path,
    temperatures: list[float],
    calibration_seeds: int = 5,
) -> Path:
    """Evaluate sampled softened-clone behavior and its normalized entropy."""
    pc, nc, dc, graph, hospitals, lookup, pds, union, age, env, nodes = _setup(affected, config, data_dir)
    fleet_count = sum(int(value) for value in pc["fleet_by_class"].values())
    edge_count = graph.number_of_edges()
    model = _clone(config, results_dir, edge_count, fleet_count)
    rows = []
    for temperature in temperatures:
        for seed in range(calibration_seeds):
            total = 0.0
            entropies: list[float] = []
            for day in range(8, 13):
                patient_seed = int(pc["seed"]) + 300000 + seed * 100 + day
                torch.manual_seed(patient_seed + int(round(temperature * 10000)))
                patients = generate_patients(day, float(pds.loc[day,"affected_population"]), union, nodes, age, dc, patient_seed)
                controls = [NeuralVehicleController(model, edge_count, greedy=False, temperature=temperature) for _ in range(fleet_count)]
                summary, _ = run_vehicle_episode(graph, hospitals, lookup, patients, day, controls, nc, config["belief"], env, float(pds.loc[day,"affected_population"]), seed=patient_seed)
                total += float(summary["survival_total"])
                entropies.extend(value for control in controls for value in control.normalized_entropies)
            nontrivial = [value for value in entropies if value > 0.0]
            rows.append({"synthetic_input": True, "temperature": temperature, "seed": seed, "survival": total, "normalized_entropy": float(np.mean(entropies)), "nontrivial_normalized_entropy": float(np.mean(nontrivial)) if nontrivial else 0.0, "nontrivial_decision_share": len(nontrivial) / max(len(entropies), 1)})
    path = results_dir / "stage6_vehicle_temperature_sweep.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
