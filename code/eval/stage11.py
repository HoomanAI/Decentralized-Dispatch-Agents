"""Stage 11 staged per-agent learning over unchanged decentralized beliefs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import stats

from code.demand.patients import generate_patients
from code.env.dispatch import build_runtime_network
from code.env.vehicle_centric import LOOKAHEAD_FEATURE_NAMES, ProtocolMyopic, run_vehicle_episode
from code.eval.stage6 import _perturb_exposure
from code.eval.stage6_vehicle import _setup
from code.policy.vehicle_centric import LookaheadResidualPolicy, NeuralVehicleController


def _update(
    model: LookaheadResidualPolicy,
    optimizer: torch.optim.Optimizer,
    transitions: list[dict[str, Any]],
    clip_ratio: float,
    value_weight: float,
    minibatch_size: int,
    entropy_coefficient: float,
) -> dict[str, float]:
    chosen = np.random.choice(
        len(transitions), size=min(minibatch_size, len(transitions)), replace=False
    )
    losses, actors, critics, entropies = [], [], [], []
    for raw_index in chosen:
        item = transitions[int(raw_index)]
        logits, q_values = model(item["state"])
        allowed = item["state"]["allowed_mask"]
        masked = torch.where(allowed, logits, torch.full_like(logits, -1.0e9))
        temperature = item["state"]["behavior_temperature"]
        distribution = torch.distributions.Categorical(logits=masked / temperature)
        action = torch.tensor(item["action"])
        ratio = torch.exp(distribution.log_prob(action) - item["old_logp"])
        target = torch.tensor(item["target"], dtype=torch.float32)
        advantage = (target - torch.sum(distribution.probs * q_values)).detach()
        actor = -torch.minimum(
            ratio * advantage,
            torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantage,
        )
        critic = (q_values[action] - target) ** 2
        allowed_count = int(allowed.sum().item())
        normalized_entropy = (
            distribution.entropy() / float(np.log(allowed_count))
            if allowed_count > 1 else distribution.entropy() * 0.0
        )
        losses.append(actor + value_weight * critic - entropy_coefficient * normalized_entropy)
        actors.append(actor)
        critics.append(critic)
        entropies.append(distribution.entropy())
    loss = torch.stack(losses).mean()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return {
        "loss": float(loss.item()),
        "actor_loss": float(torch.stack(actors).mean().item()),
        "critic_loss": float(torch.stack(critics).mean().item()),
        "entropy": float(torch.stack(entropies).mean().item()),
        "entropy_coefficient": float(entropy_coefficient),
    }


def _calibrate_greedy_logits(
    model: LookaheadResidualPolicy,
    transitions: list[dict[str, Any]],
    target_entropy: float = 0.175,
) -> dict[str, float]:
    """Scale fixed greedy logits without changing their argmax ordering."""
    states = [item["state"] for item in transitions if int(item["state"]["allowed_mask"].sum()) > 1]
    if not states:
        raise RuntimeError("clone calibration found no nontrivial decisions")

    def diagnostics(scale: float) -> tuple[float, float]:
        entropies, margins = [], []
        for state in states:
            allowed = state["allowed_mask"]
            logits = state["arrival_bias"][allowed] * scale
            entropies.append(
                float(torch.distributions.Categorical(logits=logits).entropy().item())
                / float(np.log(len(logits)))
            )
            ordered = torch.sort(logits, descending=True).values
            margins.append(float((ordered[0] - ordered[1]).item()))
        return float(np.mean(entropies)), float(np.max(margins))

    before_entropy, before_margin = diagnostics(1.0)
    low, high = 1.0e-3, 256.0
    for _ in range(50):
        middle = (low + high) / 2.0
        entropy, _ = diagnostics(middle)
        if entropy > target_entropy:
            low = middle
        else:
            high = middle
    scale = (low + high) / 2.0
    after_entropy, after_margin = diagnostics(scale)
    model.base_logit_scale.fill_(scale)
    return {
        "target_nontrivial_normalized_entropy": target_entropy,
        "base_logit_scale": scale,
        "before_nontrivial_normalized_entropy": before_entropy,
        "after_nontrivial_normalized_entropy": after_entropy,
        "before_maximum_logit_margin": before_margin,
        "after_maximum_logit_margin": after_margin,
        "argmax_agreement": 1.0,
        "nontrivial_decisions": len(states),
    }


def train_component_a(
    affected: pd.DataFrame,
    config: dict[str, Any],
    data_dir: Path,
    results_dir: Path,
    updates: int = 625,
) -> Path:
    """Train the lookahead dispatch residual while route exploration is frozen off."""
    pc, nc, dc, graph, hospitals, lookup, pds, union, age, env, nodes = _full_setup(
        affected, config, data_dir
    )
    fleet_count = sum(int(value) for value in env["fleet_by_class"].values())
    assert fleet_count == 12
    torch.manual_seed(int(pc["seed"]) + 1100000)
    np.random.seed(int(pc["seed"]) + 1100000)
    model = LookaheadResidualPolicy(int(pc["hidden_dim"]), fleet_count)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(pc["learning_rate"]))
    modes = list(pc["augmentation_modes"])
    augmented = {
        mode: _perturb_exposure(
            graph,
            data_dir / "synthetic_fire_fronts.geojson",
            mode,
            float(pc["augmentation_buffer_degrees"]),
            float(pc["augmentation_translation_degrees"]),
        )
        for mode in set(modes)
    }
    progress_path = results_dir / "stage11_component_a_progress.pt"
    partial_path = results_dir / "stage11_component_a_training.partial.csv"
    calibration_path = results_dir / "stage11_component_a_clone_calibration.csv"
    logs: list[dict[str, Any]] = []
    episode = 0
    steps = 0
    entropy_coefficient = 0.0
    entropy_dual_learning_rate = 0.5
    checkpoint_seeds = (0, 1, 2)
    checkpoint_q = 1.0

    def checkpoint_rollout(policy_model: LookaheadResidualPolicy | None) -> tuple[float, float]:
        survivals: list[float] = []
        entropies: list[float] = []
        for heldout_seed in checkpoint_seeds:
            evidence: list[tuple[np.ndarray, np.ndarray]] = []
            controls = (
                [ProtocolMyopic() for _ in range(fleet_count)]
                if policy_model is None
                else [
                    NeuralVehicleController(
                        policy_model, graph.number_of_edges(), greedy=True
                    )
                    for _ in range(fleet_count)
                ]
            )
            total = 0.0
            for day in range(8, 13):
                patient_seed = (
                    int(pc["seed"]) + 1210000 + heldout_seed * 100 + day
                )
                patients = generate_patients(
                    day, float(pds.loc[day, "affected_population"]), union, nodes, age,
                    dc, patient_seed,
                )
                with torch.no_grad():
                    summary, _ = run_vehicle_episode(
                        graph, hospitals, lookup, patients, day, controls, nc,
                        config["belief"], env,
                        float(pds.loc[day, "affected_population"]),
                        communication_reliability=checkpoint_q,
                        seed=patient_seed,
                        belief_evidence=evidence,
                    )
                total += float(summary["survival_total"])
            survivals.append(total)
            if policy_model is not None:
                entropies.extend(
                    value
                    for control in controls
                    for value in control.nontrivial_normalized_entropies
                )
        return float(np.mean(survivals)), (
            float(np.mean(entropies)) if entropies else float("nan")
        )

    protocol_checkpoint_survival, _ = checkpoint_rollout(None)
    if progress_path.exists():
        saved = torch.load(progress_path, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        episode = int(saved["episode"])
        steps = int(saved["environment_steps"])
        entropy_coefficient = float(saved["entropy_coefficient"])
        if partial_path.exists():
            logs = pd.read_csv(partial_path).to_dict("records")
    else:
        calibration_graph = augmented[modes[0]]
        calibration_lookup = {
            int(attributes["edge_id"]): index
            for index, (_, _, attributes) in enumerate(calibration_graph.edges(data=True))
        }
        calibration_controls = [
            NeuralVehicleController(model, calibration_graph.number_of_edges(), greedy=True)
            for _ in range(fleet_count)
        ]
        calibration_evidence: list[tuple[np.ndarray, np.ndarray]] = []
        for day in range(8, 13):
            patient_seed = int(pc["seed"]) + 1190000 + day
            patients = generate_patients(
                day, float(pds.loc[day, "affected_population"]), union, nodes, age, dc,
                patient_seed,
            )
            run_vehicle_episode(
                calibration_graph, hospitals, calibration_lookup, patients, day,
                calibration_controls, nc, config["belief"], env,
                float(pds.loc[day, "affected_population"]),
                communication_reliability=1.0, seed=patient_seed,
                belief_evidence=calibration_evidence, mark_terminal=day == 12,
            )
        calibration_transitions = [
            item for control in calibration_controls for item in control.transitions
        ]
        pd.DataFrame([
            _calibrate_greedy_logits(model, calibration_transitions)
        ]).to_csv(calibration_path, index=False)
    while len(logs) < updates:
        mode = modes[episode % len(modes)]
        train_graph = augmented[mode]
        train_lookup = {
            int(attributes["edge_id"]): index
            for index, (_, _, attributes) in enumerate(train_graph.edges(data=True))
        }
        controls = [
            NeuralVehicleController(model, train_graph.number_of_edges(), greedy=False)
            for _ in range(fleet_count)
        ]
        evidence: list[tuple[np.ndarray, np.ndarray]] = []
        horizon_survival = 0.0
        for day in range(8, 13):
            patient_seed = int(pc["seed"]) + 1100000 + episode * 10 + day
            patients = generate_patients(
                day, float(pds.loc[day, "affected_population"]), union, nodes, age, dc,
                patient_seed,
            )
            summary, _ = run_vehicle_episode(
                train_graph, hospitals, train_lookup, patients, day, controls, nc,
                config["belief"], env, float(pds.loc[day, "affected_population"]),
                communication_reliability=float(episode % 2), seed=patient_seed,
                belief_evidence=evidence, mark_terminal=day == 12,
            )
            horizon_survival += float(summary["survival_total"])
        transitions = [item for control in controls for item in control.transitions]
        for control in controls:
            running = 0.0
            for item in reversed(control.transitions):
                if item["terminal"]:
                    running = 0.0
                running = float(item["reward"]) + running
                item["target"] = running
        if len(transitions) >= int(pc["minibatch_size"]):
            aggregate_entropy = float(
                np.mean([value for control in controls for value in control.normalized_entropies])
            )
            nontrivial_entropy = float(
                np.mean([
                    value for control in controls
                    for value in control.nontrivial_normalized_entropies
                ])
            )
            nontrivial_decision_share = float(
                np.mean([
                    count > 1 for control in controls for count in control.allowed_action_counts
                ])
            )
            maximum_logit_margin = float(
                np.max([value for control in controls for value in control.logit_margins])
            )
            steps += len(transitions)
            # Two minibatch updates use the collected horizon once each.  At roughly
            # 1,600 decisions per horizon this preserves about 500k environment steps
            # over 625 updates instead of discarding nearly all collected experience.
            for _ in range(min(2, updates - len(logs))):
                fraction = len(logs) / max(updates - 1, 1)
                target_entropy = 0.25 if fraction <= 2.0 / 3.0 else (
                    0.25 - 0.16 * (fraction - 2.0 / 3.0) / (1.0 / 3.0)
                )
                entropy_coefficient = float(
                    max(
                        0.0,
                        entropy_coefficient
                        + entropy_dual_learning_rate * (target_entropy - nontrivial_entropy),
                    )
                )
                metrics = _update(
                    model, optimizer, transitions, float(pc["clip_ratio"]),
                    float(pc["value_weight"]), int(pc["minibatch_size"]),
                    entropy_coefficient,
                )
                is_checkpoint = (len(logs) + 1) % 10 == 0 or (len(logs) + 1) == updates
                if is_checkpoint:
                    greedy_checkpoint_survival, greedy_checkpoint_entropy = checkpoint_rollout(
                        model
                    )
                else:
                    greedy_checkpoint_survival = float("nan")
                    greedy_checkpoint_entropy = float("nan")
                logs.append(
                    {
                        "update": len(logs) + 1,
                        "episode": episode + 1,
                        "environment_steps": steps,
                        "horizon_survival": horizon_survival,
                        "target_normalized_entropy": target_entropy,
                        "mean_normalized_entropy": aggregate_entropy,
                        "nontrivial_normalized_entropy": nontrivial_entropy,
                        "nontrivial_decision_share": nontrivial_decision_share,
                        "maximum_logit_margin": maximum_logit_margin,
                        "entropy_dual_learning_rate": entropy_dual_learning_rate,
                        "checkpoint_greedy_survival": greedy_checkpoint_survival,
                        "checkpoint_protocol_myopic_survival": protocol_checkpoint_survival,
                        "checkpoint_greedy_nontrivial_entropy": greedy_checkpoint_entropy,
                        "checkpoint_heldout_seed_count": len(checkpoint_seeds),
                        "checkpoint_communication_reliability": checkpoint_q,
                        "checkpoint_instance": "full_1297_arc",
                        "checkpoint_fleet": "A4_B4_C4",
                        "checkpoint_posture": "alpha0.75_gamma0.75_phi0.70",
                        **metrics,
                    }
                )
                if len(logs) % 10 == 0 or len(logs) == updates:
                    pd.DataFrame(logs).to_csv(partial_path, index=False)
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "episode": episode + 1,
                            "environment_steps": steps,
                            "entropy_coefficient": entropy_coefficient,
                            "feature_names": LOOKAHEAD_FEATURE_NAMES,
                            "synthetic_input": True,
                        },
                        progress_path,
                    )
        episode += 1
    final_path = results_dir / "stage11_component_a.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "updates": updates,
            "environment_steps": steps,
            "feature_names": LOOKAHEAD_FEATURE_NAMES,
            "synthetic_input": True,
        },
        final_path,
    )
    pd.DataFrame(logs).to_csv(results_dir / "stage11_component_a_training.csv", index=False)
    return final_path


def _full_setup(affected: pd.DataFrame, config: dict[str, Any], data_dir: Path):
    pc, nc, dc = config["policy_training"], config["network"], config["demand"]
    graph, hospitals, lookup = build_runtime_network(
        data_dir / "synthetic_road_features.geojson",
        data_dir / "synthetic_fire_fronts.geojson",
        nc,
    )
    pds = affected.query("region == 'PDS'").set_index("day")
    union = float(pds["Total"].iloc[0])
    age = config["validation"]["expected_population_65_plus"] / config["validation"]["expected_population"]
    env = dict(config["environment"])
    env.update(
        {
            "scenario_alpha": float(pc["scenario_alpha"]),
            "scenario_gamma": float(pc["scenario_gamma"]),
            "phi_min": float(pc["phi_min"]),
            "scenario_alpha_concentration": pc["scenario_alpha_concentration"],
            "fleet_by_class": {"A": 4, "B": 4, "C": 4},
            "operational_minutes": float(dc["operational_minutes"]),
        }
    )
    nodes = np.asarray([node for node, attributes in graph.nodes(data=True) if not attributes.get("hospital")])
    return pc, nc, dc, graph, hospitals, lookup, pds, union, age, env, nodes


def evaluate_component_a(
    affected: pd.DataFrame,
    config: dict[str, Any],
    data_dir: Path,
    results_dir: Path,
) -> Path:
    """Evaluate Component A at full and absent communication on 30 matched seeds."""
    pc, nc, dc, graph, hospitals, lookup, pds, union, age, env, nodes = _full_setup(
        affected, config, data_dir
    )
    fleet_count = sum(int(value) for value in env["fleet_by_class"].values())
    saved = torch.load(results_dir / "stage11_component_a.pt", map_location="cpu", weights_only=False)
    model = LookaheadResidualPolicy(int(pc["hidden_dim"]), fleet_count)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    rows = []
    for q in [1.0, 0.0]:
        for seed in range(int(pc["evaluation_seeds"])):
            policy_controls = [NeuralVehicleController(model, graph.number_of_edges(), greedy=True) for _ in range(fleet_count)]
            protocol_controls = [ProtocolMyopic() for _ in range(fleet_count)]
            policy_evidence: list[tuple[np.ndarray, np.ndarray]] = []
            protocol_evidence: list[tuple[np.ndarray, np.ndarray]] = []
            for day in range(8, 13):
                patient_seed = int(config["environment"]["seed"]) + 300000 + seed * 100 + day
                patients = generate_patients(
                    day, float(pds.loc[day, "affected_population"]), union, nodes, age, dc,
                    patient_seed,
                )
                protocol, _ = run_vehicle_episode(
                    graph, hospitals, lookup, patients, day, protocol_controls, nc,
                    config["belief"], env, float(pds.loc[day, "affected_population"]),
                    communication_reliability=q, seed=patient_seed,
                    belief_evidence=protocol_evidence,
                )
                policy, _ = run_vehicle_episode(
                    graph, hospitals, lookup, patients, day, policy_controls, nc,
                    config["belief"], env, float(pds.loc[day, "affected_population"]),
                    communication_reliability=q, seed=patient_seed,
                    belief_evidence=policy_evidence,
                )
                rows.append(
                    {
                        "synthetic_input": True,
                        "seed": seed,
                        "day": day,
                        "communication_reliability": q,
                        "protocol_myopic": protocol["survival_total"],
                        "component_a": policy["survival_total"],
                        "difference": policy["survival_total"] - protocol["survival_total"],
                    }
                )
            rows[-1]["argmax_agreement"] = float(
                np.mean([value for control in policy_controls for value in control.myopic_agreements])
            )
            nontrivial_agreements = [
                agreement
                for control in policy_controls
                for count, agreement in zip(
                    control.allowed_action_counts, control.myopic_agreements
                )
                if count >= 2
            ]
            rows[-1]["nontrivial_argmax_agreement"] = (
                float(np.mean(nontrivial_agreements))
                if nontrivial_agreements else float("nan")
            )
            rows[-1]["forced_epoch_fraction"] = float(
                np.mean([
                    count == 1
                    for control in policy_controls
                    for count in control.allowed_action_counts
                ])
            )
            rows[-1]["normalized_entropy"] = float(
                np.mean([value for control in policy_controls for value in control.normalized_entropies])
            )
    frame = pd.DataFrame(rows)
    path = results_dir / "stage11_component_a_evaluation.csv"
    frame.to_csv(path, index=False)
    summaries = []
    for q in [1.0, 0.0]:
        level = frame[frame.communication_reliability == q]
        for endpoint, values in {
            "five_day": level.groupby("seed").difference.sum(),
            "days_9_11": level[level.day.isin([9, 10, 11])].groupby("seed").difference.sum(),
            "day_12": level[level.day == 12].set_index("seed").difference,
        }.items():
            lower, upper = stats.t.interval(
                0.95, len(values) - 1, loc=values.mean(), scale=stats.sem(values)
            )
            summaries.append(
                {
                    "communication_reliability": q,
                    "endpoint": endpoint,
                    "point": float(values.mean()),
                    "lower": float(lower),
                    "upper": float(upper),
                    "p": float(stats.ttest_1samp(values, 0.0).pvalue),
                    "n": len(values),
                    "argmax_agreement": float(level.argmax_agreement.dropna().mean()),
                    "nontrivial_argmax_agreement": float(
                        level.nontrivial_argmax_agreement.dropna().mean()
                    ),
                    "forced_epoch_fraction": float(
                        level.forced_epoch_fraction.dropna().mean()
                    ),
                    "normalized_entropy": float(level.normalized_entropy.dropna().mean()),
                }
            )
    pd.DataFrame(summaries).to_csv(results_dir / "stage11_component_a_gate.csv", index=False)
    return path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load((root / "code" / "config" / "default.yaml").read_text(encoding="utf-8"))
    affected = pd.read_csv(root / "data" / "daily_affected_population_clean.csv")
    train_component_a(affected, config, root / "data", root / "results")
    evaluate_component_a(affected, config, root / "data", root / "results")


if __name__ == "__main__":
    main()
