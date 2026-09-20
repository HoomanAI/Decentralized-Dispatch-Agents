"""Train and evaluate Stage 11 Component B route exploration."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import stats

from code.demand.patients import generate_patients
from code.env.vehicle_centric import run_vehicle_episode
from code.eval.stage11 import _full_setup
from code.exploration.triage import ExplorationLedger, InformationRoutePolicy
from code.policy.vehicle_centric import (
    NeuralRouteController,
    RouteActorQCritic,
    route_state,
)


FIXED_BUDGETS = (0.0, 0.05, 0.0)
TRAINING_BUDGETS = (float("inf"), float("inf"), float("inf"))
ROLLOUT_WORKERS = 8
UPDATES_PER_BATCH = 6
CONVERGENCE_WINDOW = 50
CONVERGENCE_TOLERANCE = 0.02

_ROLLOUT_CONTEXT: tuple[Any, ...] | None = None


class FixedRationDemonstrator:
    """Record route-choice states while executing the fixed ration exactly."""

    def __init__(
        self,
        ledger: ExplorationLedger,
        edge_lookup: dict[int, int],
    ) -> None:
        self.policy = InformationRoutePolicy(ledger, 1.0)
        self.edge_lookup = edge_lookup
        self.records: list[dict[str, Any]] = []

    def propose(self, context: dict[str, Any]) -> tuple[int, float]:
        choice, score = self.policy.propose(context)
        candidate = context["candidates"][choice]
        if candidate.patient_id is not None:
            route_indices = [
                index
                for index in context["allowed_indices"]
                if context["candidates"][index].patient_id == candidate.patient_id
            ]
            self.records.append(
                {
                    "state": route_state(context, route_indices, self.edge_lookup),
                    "action": route_indices.index(choice),
                    "detour": int(candidate.route_rank > 0),
                }
            )
        return choice, score

    def observe(self, reward: float, terminal: bool = False) -> None:
        self.policy.observe(reward, terminal)

    def set_other_actions(self, summary: np.ndarray) -> None:
        self.policy.set_other_actions(summary)


def _route_update(
    model: RouteActorQCritic,
    optimizer: torch.optim.Optimizer,
    transitions: list[dict[str, Any]],
    clip_ratio: float,
    value_weight: float,
    minibatch_size: int,
    entropy_weight: float,
) -> dict[str, float]:
    chosen = np.random.choice(
        len(transitions), size=min(minibatch_size, len(transitions)), replace=False
    )
    losses, actors, critics, entropies = [], [], [], []
    for raw_index in chosen:
        item = transitions[int(raw_index)]
        logits, q_values = model(item["state"])
        distribution = torch.distributions.Categorical(logits=logits)
        action = torch.tensor(item["action"])
        ratio = torch.exp(distribution.log_prob(action) - item["old_logp"])
        target = torch.tensor(item["target"], dtype=torch.float32)
        advantage = (target - torch.sum(distribution.probs * q_values)).detach()
        actor = -torch.minimum(
            ratio * advantage,
            torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantage,
        )
        critic = (q_values[action] - target) ** 2
        normalized_entropy = distribution.entropy() / max(float(np.log(len(logits))), 1.0)
        losses.append(actor + value_weight * critic - entropy_weight * normalized_entropy)
        actors.append(actor)
        critics.append(critic)
        entropies.append(normalized_entropy)
    loss = torch.stack(losses).mean()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return {
        "loss": float(loss.item()),
        "actor_loss": float(torch.stack(actors).mean().item()),
        "critic_loss": float(torch.stack(critics).mean().item()),
        "normalized_entropy_loss_sample": float(torch.stack(entropies).mean().item()),
    }


def _training_setup(
    affected: pd.DataFrame,
    config: dict[str, Any],
    data_dir: Path,
    posture: tuple[float, float, float] | None = None,
) -> tuple[Any, ...]:
    setup = _full_setup(affected, config, data_dir)
    pc, nc, dc, graph, hospitals, lookup, pds, union, age, env, nodes = setup
    env.update(
        {
            "exploration_enabled": True,
            "exploration_budgets": TRAINING_BUDGETS,
            "exploration_k": 3,
            "exploration_detour_minutes": 20.0,
        }
    )
    if posture is not None:
        env.update(
            {
                "scenario_alpha": float(posture[0]),
                "scenario_gamma": float(posture[1]),
                "phi_min": float(posture[2]),
            }
        )
    fleet_count = sum(int(value) for value in env["fleet_by_class"].values())
    assert fleet_count == 12
    return setup


def _initialize_rollout_worker(
    affected: pd.DataFrame,
    config: dict[str, Any],
    data_dir: Path,
    posture: tuple[float, float, float] | None,
) -> None:
    """Build immutable simulation state once in each rollout worker."""
    global _ROLLOUT_CONTEXT
    _ROLLOUT_CONTEXT = (*_training_setup(affected, config, data_dir, posture), config)


def _collect_rollout(
    task: tuple[int, dict[str, torch.Tensor]],
) -> dict[str, Any]:
    """Collect one independently seeded five-day rollout from a frozen policy."""
    if _ROLLOUT_CONTEXT is None:
        raise RuntimeError("Component B rollout worker was not initialized")
    episode, model_state = task
    pc, nc, dc, graph, hospitals, lookup, pds, union, age, env, nodes, config = (
        _ROLLOUT_CONTEXT
    )
    fleet_count = sum(int(value) for value in env["fleet_by_class"].values())
    torch.manual_seed(int(pc["seed"]) + 1300000 + episode)
    np.random.seed(int(pc["seed"]) + 1300000 + episode)
    model = RouteActorQCritic(int(pc["hidden_dim"]), fleet_count)
    model.load_state_dict(model_state)
    model.eval()
    controls = [NeuralRouteController(model, lookup, greedy=False) for _ in range(fleet_count)]
    evidence: list[tuple[np.ndarray, np.ndarray]] = []
    isolation: list[int] = []
    horizon_survival = 0.0
    daily_costs = np.zeros((5, 3), dtype=float)
    for day_offset, day in enumerate(range(8, 13)):
        patient_seed = int(pc["seed"]) + 1300000 + episode * 10 + day
        patients = generate_patients(
            day,
            float(pds.loc[day, "affected_population"]),
            union,
            nodes,
            age,
            dc,
            patient_seed,
        )
        before = [len(control.transitions) for control in controls]
        summary, _ = run_vehicle_episode(
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
            communication_reliability=0.0,
            seed=patient_seed,
            belief_evidence=evidence,
            isolation_epochs_state=isolation,
            mark_terminal=day == 12,
        )
        horizon_survival += float(summary["survival_total"])
        for control, start in zip(controls, before):
            for item in control.transitions[start:]:
                daily_costs[day_offset, int(item["triage"]) - 1] += float(
                    item["exploration_cost"]
                )
    return {
        "transition_groups": [control.transitions for control in controls],
        "environment_steps": sum(len(control.transitions) for control in controls),
        "horizon_survival": horizon_survival,
        "mean_cost": daily_costs.mean(axis=0),
        "route_sizes": [size for control in controls for size in control.route_set_sizes],
        "route_entropies": [
            value for control in controls for value in control.normalized_entropies
        ],
        "detours": sum(
            int(choice) for control in controls for choice in control.detour_choices
        ),
    }


def _collect_fixed_ration_demonstration(episode: int) -> dict[str, Any]:
    """Collect one five-day fixed-ration imitation trajectory."""
    if _ROLLOUT_CONTEXT is None:
        raise RuntimeError("Component B rollout worker was not initialized")
    pc, nc, dc, graph, hospitals, lookup, pds, union, age, env, nodes, config = (
        _ROLLOUT_CONTEXT
    )
    fleet_count = sum(int(value) for value in env["fleet_by_class"].values())
    evidence: list[tuple[np.ndarray, np.ndarray]] = []
    isolation: list[int] = []
    records: list[dict[str, Any]] = []
    detours = 0
    decisions = 0
    for day in range(8, 13):
        patient_seed = int(pc["seed"]) + 2300000 + episode * 10 + day
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
        controls = [
            FixedRationDemonstrator(ledger, lookup) for _ in range(fleet_count)
        ]
        _, trace = run_vehicle_episode(
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
            communication_reliability=0.0,
            seed=patient_seed,
            belief_evidence=evidence,
            isolation_epochs_state=isolation,
        )
        records.extend(record for control in controls for record in control.records)
        detours += sum(int(item["route_rank"] > 0) for item in trace)
        decisions += len(trace)
    return {"records": records, "detours": detours, "decisions": decisions}


def _fit_fixed_ration_warm_start(
    model: RouteActorQCritic,
    demonstrations: list[dict[str, Any]],
) -> dict[str, float]:
    """Imitate the fixed ration before survival-only policy optimization."""
    if not demonstrations:
        raise RuntimeError("Fixed-ration warm start collected no demonstrations")
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    rng = np.random.default_rng(2300000)
    for _ in range(250):
        chosen = rng.choice(
            len(demonstrations), size=min(256, len(demonstrations)), replace=False
        )
        losses = []
        for raw_index in chosen:
            record = demonstrations[int(raw_index)]
            logits, _ = model(record["state"])
            target = torch.tensor(int(record["action"]), dtype=torch.long)
            weight = 5.0 if int(record["detour"]) else 1.0
            losses.append(weight * torch.nn.functional.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0)))
        loss = torch.stack(losses).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    correct = 0
    predicted_detours = 0
    target_detours = 0
    entropies = []
    with torch.no_grad():
        for record in demonstrations:
            logits, _ = model(record["state"])
            prediction = int(torch.argmax(logits).item())
            correct += int(prediction == int(record["action"]))
            predicted_detours += int(prediction > 0)
            target_detours += int(record["detour"])
            if len(logits) > 1:
                distribution = torch.distributions.Categorical(logits=logits)
                entropies.append(float(distribution.entropy() / np.log(len(logits))))
    return {
        "demonstrations": len(demonstrations),
        "agreement": correct / len(demonstrations),
        "target_detour_fraction": target_detours / len(demonstrations),
        "predicted_detour_fraction": predicted_detours / len(demonstrations),
        "normalized_entropy": float(np.mean(entropies)) if entropies else 0.0,
    }


def _component_b_converged(logs: list[dict[str, Any]]) -> bool:
    """Apply the pre-specified 2 percent stability rule over 50 updates."""
    if len(logs) < 100 or len(logs) < CONVERGENCE_WINDOW:
        return False
    window = logs[-CONVERGENCE_WINDOW:]
    metrics = [
        "mu_1",
        "mu_2",
        "mu_3",
        "cost_1_per_day",
        "cost_2_per_day",
        "cost_3_per_day",
    ]
    for metric in metrics:
        first = float(window[0][metric])
        last = float(window[-1][metric])
        scale = max(abs(first), abs(last), 1.0e-8)
        if abs(last - first) > CONVERGENCE_TOLERANCE * scale:
            return False
    return True


def train_component_b(
    affected: pd.DataFrame,
    config: dict[str, Any],
    data_dir: Path,
    results_dir: Path,
    updates: int = 200,
    minimum_environment_steps: int = 150000,
    max_workers: int = ROLLOUT_WORKERS,
    artifact_stem: str = "stage11_component_b",
    fixed_ration_warm_start: bool = False,
    posture: tuple[float, float, float] | None = None,
) -> Path:
    """Train route choice at zero communication with dual budget control."""
    pc, nc, dc, graph, hospitals, lookup, pds, union, age, env, nodes = _training_setup(
        affected, config, data_dir, posture
    )
    fleet_count = sum(int(value) for value in env["fleet_by_class"].values())
    torch.manual_seed(int(pc["seed"]) + 1300000)
    np.random.seed(int(pc["seed"]) + 1300000)
    model = RouteActorQCritic(int(pc["hidden_dim"]), fleet_count)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(pc["learning_rate"]))
    dual = np.zeros(3, dtype=float)
    dual_learning_rate = 0.5
    entropy_weight = 0.01
    progress_path = results_dir / f"{artifact_stem}_progress.pt"
    partial_path = results_dir / f"{artifact_stem}_training.partial.csv"
    logs: list[dict[str, Any]] = []
    episode = 0
    environment_steps = 0
    if progress_path.exists():
        saved = torch.load(progress_path, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        dual = np.asarray(saved["dual"], dtype=float)
        episode = int(saved["episode"])
        environment_steps = int(saved["environment_steps"])
        if partial_path.exists():
            logs = pd.read_csv(partial_path).to_dict("records")
    stop_reason = "update_ceiling"
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_initialize_rollout_worker,
        initargs=(affected, config, data_dir, posture),
    ) as executor:
        if fixed_ration_warm_start and not logs:
            demo_rollouts = list(
                executor.map(_collect_fixed_ration_demonstration, range(max_workers))
            )
            demonstrations = [
                record for rollout in demo_rollouts for record in rollout["records"]
            ]
            warm_start_metrics = _fit_fixed_ration_warm_start(model, demonstrations)
            warm_start_metrics.update(
                {
                    "episodes": len(demo_rollouts),
                    "target_detours_per_day": sum(
                        int(rollout["detours"]) for rollout in demo_rollouts
                    )
                    / (5.0 * len(demo_rollouts)),
                    "target_decisions": sum(
                        int(rollout["decisions"]) for rollout in demo_rollouts
                    ),
                }
            )
            pd.DataFrame([warm_start_metrics]).to_csv(
                results_dir / f"{artifact_stem}_warm_start.csv", index=False
            )
        while len(logs) < updates and environment_steps < minimum_environment_steps:
            model_state = {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            }
            tasks = [(episode + offset, model_state) for offset in range(max_workers)]
            rollouts = list(executor.map(_collect_rollout, tasks))
            episode += len(rollouts)
            transitions = [
                item
                for rollout in rollouts
                for group in rollout["transition_groups"]
                for item in group
            ]
            transition_groups = [
                group for rollout in rollouts for group in rollout["transition_groups"]
            ]
            if not transitions:
                raise RuntimeError("Component B collected no route decisions")
            mean_cost = np.mean([rollout["mean_cost"] for rollout in rollouts], axis=0)
            dual = np.maximum(
                0.0,
                dual + dual_learning_rate * (mean_cost - np.asarray(FIXED_BUDGETS)),
            )
            for group in transition_groups:
                running = 0.0
                for item in reversed(group):
                    if item["terminal"]:
                        running = 0.0
                    adjusted = float(item["reward"]) - dual[int(item["triage"]) - 1] * float(
                        item["exploration_cost"]
                    )
                    running = adjusted + running
                    item["target"] = running
            environment_steps += sum(int(r["environment_steps"]) for r in rollouts)
            route_sizes = [size for rollout in rollouts for size in rollout["route_sizes"]]
            route_entropies = [
                value for rollout in rollouts for value in rollout["route_entropies"]
            ]
            detours = sum(int(rollout["detours"]) for rollout in rollouts)
            horizon_survival = float(np.mean([r["horizon_survival"] for r in rollouts]))
            batch_updates = min(UPDATES_PER_BATCH, updates - len(logs))
            for _ in range(batch_updates):
                metrics = _route_update(
                    model, optimizer, transitions, float(pc["clip_ratio"]),
                    float(pc["value_weight"]), int(pc["minibatch_size"]), entropy_weight,
                )
                logs.append(
                    {
                        "update": len(logs) + 1,
                        "episode": episode,
                        "environment_steps": environment_steps,
                        "horizon_survival": horizon_survival,
                        "normalized_route_entropy": float(np.mean(route_entropies)) if route_entropies else 0.0,
                        "mean_route_set_size": float(np.mean(route_sizes)),
                        "maximum_route_set_size": int(max(route_sizes)),
                        "detour_choices": detours,
                        "cost_1_per_day": mean_cost[0],
                        "cost_2_per_day": mean_cost[1],
                        "cost_3_per_day": mean_cost[2],
                        "epsilon_1": FIXED_BUDGETS[0],
                        "epsilon_2": FIXED_BUDGETS[1],
                        "epsilon_3": FIXED_BUDGETS[2],
                        "mu_1": dual[0],
                        "mu_2": dual[1],
                        "mu_3": dual[2],
                        "rollout_workers": max_workers,
                        "communication_reliability": 0.0,
                        "instance": "full_1297_arc",
                        "fleet": "A4_B4_C4",
                        "posture": (
                            f"alpha{float(env['scenario_alpha']):g}_"
                            f"gamma{float(env['scenario_gamma']):g}_"
                            f"phi{float(env['phi_min']):g}"
                        ),
                        **metrics,
                    }
                )
            pd.DataFrame(logs).to_csv(partial_path, index=False)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "dual": dual,
                    "episode": episode,
                    "environment_steps": environment_steps,
                    "updates": len(logs),
                    "synthetic_input": True,
                },
                progress_path,
            )
            if _component_b_converged(logs):
                stop_reason = "converged_2pct_over_50_updates"
                break
            if environment_steps >= minimum_environment_steps:
                stop_reason = "decision_target"
                break
    final_path = results_dir / f"{artifact_stem}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "dual": dual,
            "updates": len(logs),
            "environment_steps": environment_steps,
            "stop_reason": stop_reason,
            "rollout_workers": max_workers,
            "budgets": FIXED_BUDGETS,
            "synthetic_input": True,
        },
        final_path,
    )
    pd.DataFrame(logs).to_csv(results_dir / f"{artifact_stem}_training.csv", index=False)
    return final_path


def _safe_summary(values: pd.Series) -> tuple[float, float, float, float]:
    point = float(values.mean())
    if len(values) < 2 or float(values.std(ddof=1)) == 0.0:
        return point, point, point, 1.0 if point == 0.0 else 0.0
    lower, upper = stats.t.interval(
        0.95, len(values) - 1, loc=point, scale=stats.sem(values)
    )
    return point, float(lower), float(upper), float(stats.ttest_1samp(values, 0.0).pvalue)


def evaluate_component_b(
    affected: pd.DataFrame,
    config: dict[str, Any],
    data_dir: Path,
    results_dir: Path,
    artifact_stem: str = "stage11_component_b",
    posture: tuple[float, float, float] | None = None,
) -> Path:
    """Evaluate learned route selection against the fixed ration at q equal to zero."""
    pc, nc, dc, graph, hospitals, lookup, pds, union, age, env, nodes = _training_setup(
        affected, config, data_dir, posture
    )
    fleet_count = sum(int(value) for value in env["fleet_by_class"].values())
    saved = torch.load(results_dir / f"{artifact_stem}.pt", map_location="cpu", weights_only=False)
    model = RouteActorQCritic(int(pc["hidden_dim"]), fleet_count)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    rows: list[dict[str, Any]] = []
    for seed in range(int(pc["evaluation_seeds"])):
        learned_evidence: list[tuple[np.ndarray, np.ndarray]] = []
        fixed_evidence: list[tuple[np.ndarray, np.ndarray]] = []
        learned_isolation: list[int] = []
        fixed_isolation: list[int] = []
        learned_controls = [
            NeuralRouteController(model, lookup, greedy=True) for _ in range(fleet_count)
        ]
        for day in range(8, 13):
            patient_seed = int(config["environment"]["seed"]) + 300000 + seed * 100 + day
            patients = generate_patients(
                day, float(pds.loc[day, "affected_population"]), union, nodes, age, dc,
                patient_seed,
            )
            learned, learned_trace = run_vehicle_episode(
                graph, hospitals, lookup, patients, day, learned_controls, nc,
                config["belief"], env, float(pds.loc[day, "affected_population"]),
                communication_reliability=0.0, seed=patient_seed,
                belief_evidence=learned_evidence,
                isolation_epochs_state=learned_isolation,
            )
            ledger = ExplorationLedger(FIXED_BUDGETS)
            fixed_controls = [
                InformationRoutePolicy(ledger, 1.0) for _ in range(fleet_count)
            ]
            fixed, fixed_trace = run_vehicle_episode(
                graph, hospitals, lookup, patients, day, fixed_controls, nc,
                config["belief"], env, float(pds.loc[day, "affected_population"]),
                communication_reliability=0.0, seed=patient_seed,
                belief_evidence=fixed_evidence,
                isolation_epochs_state=fixed_isolation,
            )
            learned_detours = sum(int(item["route_rank"] > 0) for item in learned_trace)
            fixed_detours = sum(int(item["route_rank"] > 0) for item in fixed_trace)
            learned_costs = {
                triage: sum(
                    float(item["exploration_cost"])
                    for item in learned_trace
                    if int(item["triage"]) == triage
                )
                for triage in (1, 2, 3)
            }
            rows.append(
                {
                    "synthetic_input": True,
                    "seed": seed,
                    "day": day,
                    "communication_reliability": 0.0,
                    "scenario_alpha": float(env["scenario_alpha"]),
                    "scenario_gamma": float(env["scenario_gamma"]),
                    "phi_min": float(env["phi_min"]),
                    "fleet": "A4_B4_C4",
                    "learned_survival": learned["survival_total"],
                    "fixed_ration_survival": fixed["survival_total"],
                    "difference": learned["survival_total"] - fixed["survival_total"],
                    "learned_detours": learned_detours,
                    "fixed_ration_detours": fixed_detours,
                    "learned_cost_1": learned_costs[1],
                    "learned_cost_2": learned_costs[2],
                    "learned_cost_3": learned_costs[3],
                    "fixed_ration_cost_1": ledger.costs[1],
                    "fixed_ration_cost_2": ledger.costs[2],
                    "fixed_ration_cost_3": ledger.costs[3],
                    "learned_duplicate_rate": learned["duplicate_assignment_rate"],
                    "fixed_ration_duplicate_rate": fixed["duplicate_assignment_rate"],
                }
            )
    frame = pd.DataFrame(rows)
    evaluation_path = results_dir / f"{artifact_stem}_evaluation.csv"
    frame.to_csv(evaluation_path, index=False)
    summaries: list[dict[str, Any]] = []
    endpoints = {
        "five_day": frame.groupby("seed").difference.sum(),
        "days_9_11": frame[frame.day.isin([9, 10, 11])].groupby("seed").difference.sum(),
        "day_12": frame[frame.day == 12].set_index("seed").difference,
    }
    for endpoint, values in endpoints.items():
        point, lower, upper, p_value = _safe_summary(values)
        summaries.append(
            {
                "endpoint": endpoint,
                "point": point,
                "lower": lower,
                "upper": upper,
                "p": p_value,
                "n": len(values),
                "gate_primary": endpoint == "days_9_11",
                "gate_pass": bool(endpoint == "days_9_11" and lower > 0.0),
                "communication_reliability": 0.0,
                "epsilon_1": FIXED_BUDGETS[0],
                "epsilon_2": FIXED_BUDGETS[1],
                "epsilon_3": FIXED_BUDGETS[2],
                "mu_1": float(saved["dual"][0]),
                "mu_2": float(saved["dual"][1]),
                "mu_3": float(saved["dual"][2]),
                "learned_cost_1_per_day": float(frame.learned_cost_1.mean()),
                "learned_cost_2_per_day": float(frame.learned_cost_2.mean()),
                "learned_cost_3_per_day": float(frame.learned_cost_3.mean()),
                "fixed_epsilon_1": FIXED_BUDGETS[0],
                "fixed_epsilon_2": FIXED_BUDGETS[1],
                "fixed_epsilon_3": FIXED_BUDGETS[2],
                "learned_detours_per_day": float(frame.learned_detours.mean()),
                "fixed_ration_detours_per_day": float(frame.fixed_ration_detours.mean()),
            }
        )
    pd.DataFrame(summaries).to_csv(results_dir / f"{artifact_stem}_gate.csv", index=False)
    return evaluation_path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load((root / "code/config/default.yaml").read_text(encoding="utf-8"))
    affected = pd.read_csv(root / "data/daily_affected_population_clean.csv")
    train_component_b(affected, config, root / "data", root / "results")
    evaluate_component_b(affected, config, root / "data", root / "results")


if __name__ == "__main__":
    main()
