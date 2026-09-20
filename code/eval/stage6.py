"""Train and evaluate the Stage 6 exploitation-only shared policy."""

from __future__ import annotations

import copy
from collections import deque
import logging
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from scipy.stats import t as student_t
from scipy.stats import ttest_rel

from code.demand.patients import generate_patients
from code.env.dispatch import build_runtime_network, run_episode
from code.policy.mappo import PPOController, ppo_update
from code.policy.model import ExploitationActorCritic

LOGGER = logging.getLogger(__name__)


def _perturb_exposure(
    graph: nx.Graph,
    fronts_path: Path,
    mode: str,
    buffer_degrees: float,
    translation_degrees: float,
) -> nx.Graph:
    """Spatially perturb the calibrated exposed-edge mask for training only."""
    del fronts_path
    augmented = copy.deepcopy(graph)
    routable_edges = [
        (u, v, data)
        for u, v, data in augmented.edges(data=True)
        if data["fclass"] != "hospital_access"
    ]
    midpoint = np.array(
        [
            [
                (augmented.nodes[u]["x"] + augmented.nodes[v]["x"]) / 2.0,
                (augmented.nodes[u]["y"] + augmented.nodes[v]["y"]) / 2.0,
            ]
            for u, v, _ in routable_edges
        ]
    )
    for day in range(7, 13):
        exposed = np.array(
            [
                bool(augmented.graph["edge_day_flags"][int(data["edge_id"])][str(day)])
                for _, _, data in routable_edges
            ]
        )
        if mode == "dilation":
            if exposed.any():
                distance, _ = cKDTree(midpoint[exposed]).query(midpoint)
                transformed = exposed | (distance <= buffer_degrees)
            else:
                transformed = exposed
        elif mode == "erosion":
            if (~exposed).any():
                distance, _ = cKDTree(midpoint[~exposed]).query(midpoint)
                transformed = exposed & (distance > buffer_degrees)
            else:
                transformed = exposed
        elif mode == "translation":
            transformed = np.zeros_like(exposed)
            if exposed.any():
                shifted = midpoint[exposed] + np.array(
                    [translation_degrees, -translation_degrees]
                )
                distance, _ = cKDTree(shifted).query(midpoint)
                selected = np.argsort(distance)[: int(np.count_nonzero(exposed))]
                transformed[selected] = True
        else:
            raise ValueError(f"Unknown perimeter augmentation mode {mode}")
        for index, (_, _, attributes) in enumerate(routable_edges):
            edge_id = int(attributes["edge_id"])
            augmented.graph["edge_day_flags"][edge_id][str(day)] = int(
                transformed[index]
            )
    return augmented


def run_stage6(
    affected_population: pd.DataFrame,
    config: dict[str, Any],
    data_dir: Path,
    results_dir: Path,
) -> tuple[Path, Path, Path]:
    """Train to a step and convergence gate, then run paired-seed evaluation."""
    policy_config = config["policy_training"]
    assert all(float(value) == 0.0 for value in policy_config["exploration_budget"])
    torch.manual_seed(int(policy_config["seed"]))
    np.random.seed(int(policy_config["seed"]))
    reduced_data_dir = data_dir / "reduced"
    network_config = config["reduced_network"]
    demand_config = config["reduced_demand"]
    graph, hospitals, edge_lookup = build_runtime_network(
        reduced_data_dir / "synthetic_road_features.geojson",
        reduced_data_dir / "synthetic_fire_fronts.geojson",
        network_config,
    )
    road_classes = sorted({data["fclass"] for _, _, data in graph.edges(data=True)})
    model = ExploitationActorCritic(
        node_dim=3,
        edge_dim=3 + len(road_classes),
        vehicle_dim=6,
        patient_dim=6,
        hidden_dim=int(policy_config["hidden_dim"]),
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(policy_config["learning_rate"])
    )
    pds = affected_population.query("region == 'PDS'").set_index("day")
    union_population = float(pds["Total"].iloc[0])
    age_share = config["validation"]["expected_population_65_plus"] / config["validation"][
        "expected_population"
    ]
    environment = dict(config["environment"])
    environment.update(
        {
            "scenario_alpha": float(policy_config["scenario_alpha"]),
            "scenario_gamma": float(policy_config["scenario_gamma"]),
            "phi_min": float(policy_config["phi_min"]),
            "scenario_alpha_concentration": policy_config[
                "scenario_alpha_concentration"
            ],
            "operational_minutes": float(demand_config["operational_minutes"]),
            "fleet_by_class": dict(policy_config["fleet_by_class"]),
        }
    )
    training_log = []
    learning_curve = []
    modes = list(policy_config["augmentation_modes"])
    train_days = list(policy_config["train_days"])
    augmented_graphs = {
        mode: _perturb_exposure(
            graph,
            reduced_data_dir / "synthetic_fire_fronts.geojson",
            mode,
            float(policy_config["augmentation_buffer_degrees"]),
            float(policy_config["augmentation_translation_degrees"]),
        )
        for mode in set(modes)
    }
    environment_steps = 0
    gradient_updates = 0
    episode_index = 0
    transition_buffer = []
    recent_returns: deque[float] = deque(maxlen=200)
    next_update_step = int(policy_config["steps_per_update"])
    next_curve_step = int(policy_config["learning_curve_interval_steps"])
    plateau_reached = False
    maximum_steps = int(policy_config["maximum_environment_steps"])
    results_dir.mkdir(parents=True, exist_ok=True)
    while environment_steps < maximum_steps:
        day = int(train_days[episode_index % len(train_days)])
        mode = str(modes[episode_index % len(modes)])
        training_graph = augmented_graphs[mode]
        training_edge_lookup = {
            int(data["edge_id"]): index
            for index, (_, _, data) in enumerate(training_graph.edges(data=True))
        }
        nodes = np.array(
            [
                node
                for node, attributes in training_graph.nodes(data=True)
                if not attributes.get("hospital")
            ]
        )
        patients = generate_patients(
            day,
            float(pds.loc[day, "affected_population"]),
            union_population,
            nodes,
            age_share,
            demand_config,
            int(policy_config["seed"]) + episode_index,
        )
        controller = PPOController(
            model,
            training_edge_lookup,
            road_classes,
            training=True,
            seed=int(policy_config["seed"]) + episode_index,
        )
        summary, _ = run_episode(
            training_graph,
            hospitals,
            training_edge_lookup,
            patients,
            day,
            "belief_aware",
            network_config,
            config["belief"],
            environment,
            float(pds.loc[day, "affected_population"]),
            controller=controller,
        )
        episode_index += 1
        environment_steps += len(patients)
        transition_buffer.extend(controller.transitions)
        recent_returns.append(float(summary["survival_total"]))
        if environment_steps >= next_update_step and transition_buffer:
            anneal_updates = max(
                1,
                int(
                    float(policy_config["entropy_anneal_fraction"])
                    * int(policy_config["minimum_gradient_updates"])
                ),
            )
            anneal_progress = min(gradient_updates / anneal_updates, 1.0)
            entropy_weight = float(policy_config["entropy_weight"]) + anneal_progress * (
                float(policy_config["entropy_weight_final"])
                - float(policy_config["entropy_weight"])
            )
            metrics = ppo_update(
                model,
                optimizer,
                transition_buffer,
                float(policy_config["discount"]),
                float(policy_config["gae_lambda"]),
                float(policy_config["clip_ratio"]),
                float(policy_config["value_weight"]),
                entropy_weight,
                int(policy_config["ppo_epochs"]),
                int(policy_config["minibatch_size"]),
            )
            gradient_updates += int(bool(metrics["optimizer_step"]))
            training_log.append(
                {
                    "gradient_update": gradient_updates,
                    "environment_steps": environment_steps,
                    "episodes": episode_index,
                    "buffer_decisions": len(transition_buffer),
                    "latest_day": day,
                    "latest_augmentation": mode,
                    "recent_mean_return": float(np.mean(recent_returns)),
                    "entropy_weight": entropy_weight,
                    **metrics,
                }
            )
            transition_buffer = []
            next_update_step += int(policy_config["steps_per_update"])
        if environment_steps >= next_curve_step:
            learning_curve.append(
                {
                    "environment_steps": environment_steps,
                    "gradient_updates": gradient_updates,
                    "episodes": episode_index,
                    "mean_episode_return": float(np.mean(recent_returns)),
                    "std_episode_return": float(np.std(recent_returns)),
                }
            )
            next_curve_step += int(policy_config["learning_curve_interval_steps"])
            window = int(policy_config["plateau_window_points"])
            if len(learning_curve) >= window:
                values = np.array(
                    [item["mean_episode_return"] for item in learning_curve[-window:]],
                    dtype=float,
                )
                slope = float(np.polyfit(np.arange(window), values, 1)[0])
                relative_slope = abs(slope) / max(abs(float(values.mean())), 1.0)
                learning_curve[-1]["plateau_relative_slope"] = relative_slope
                plateau_reached = (
                    relative_slope
                    <= float(policy_config["plateau_relative_slope"])
                )
            else:
                learning_curve[-1]["plateau_relative_slope"] = float("nan")
            pd.DataFrame(learning_curve).to_csv(
                results_dir / "stage6_learning_curve.csv", index=False
            )
            pd.DataFrame(training_log).to_csv(
                results_dir / "stage6_training_log.csv", index=False
            )
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "environment_steps": environment_steps,
                    "gradient_updates": gradient_updates,
                    "episodes": episode_index,
                    "synthetic_input": True,
                },
                results_dir / "stage6_training_progress.pt",
            )
            LOGGER.info(
                "Stage 6 progress: %d environment steps, %d updates, mean return %.3f",
                environment_steps,
                gradient_updates,
                learning_curve[-1]["mean_episode_return"],
            )
        if (
            environment_steps >= int(policy_config["minimum_environment_steps"])
            and gradient_updates >= int(policy_config["minimum_gradient_updates"])
            and plateau_reached
        ):
            break

    evaluation_rows = []
    paired_rows = []
    evaluation_nodes = np.array(
        [node for node, attributes in graph.nodes(data=True) if not attributes.get("hospital")]
    )
    model.eval()
    for evaluation_seed in range(int(policy_config["evaluation_seeds"])):
        policy_total = 0.0
        myopic_total = 0.0
        for day in range(8, 13):
            patients = generate_patients(
                day,
                float(pds.loc[day, "affected_population"]),
                union_population,
                evaluation_nodes,
                age_share,
                demand_config,
                int(policy_config["seed"]) + 100000 + 100 * evaluation_seed + day,
            )
            controller = PPOController(
                model,
                edge_lookup,
                road_classes,
                training=False,
                seed=int(policy_config["seed"]) + evaluation_seed,
            )
            policy_summary, _ = run_episode(
                graph,
                hospitals,
                edge_lookup,
                patients,
                day,
                "belief_aware",
                network_config,
                config["belief"],
                environment,
                float(pds.loc[day, "affected_population"]),
                controller=controller,
            )
            myopic_summary, _ = run_episode(
                graph,
                hospitals,
                edge_lookup,
                patients,
                day,
                "belief_aware",
                network_config,
                config["belief"],
                environment,
                float(pds.loc[day, "affected_population"]),
            )
            policy_total += float(policy_summary["survival_total"])
            myopic_total += float(myopic_summary["survival_total"])
            evaluation_rows.append(
                {
                    "synthetic_input": True,
                    "evaluation_seed": evaluation_seed,
                    "method": "exploitation_only",
                    **policy_summary,
                }
            )
            evaluation_rows.append(
                {
                    "synthetic_input": True,
                    "evaluation_seed": evaluation_seed,
                    "method": "belief_aware_myopic",
                    **myopic_summary,
                }
            )
        paired_rows.append(
            {
                "evaluation_seed": evaluation_seed,
                "policy_total": policy_total,
                "myopic_total": myopic_total,
                "paired_difference": policy_total - myopic_total,
            }
        )
    training_path = results_dir / "stage6_training_log.csv"
    evaluation_path = results_dir / "stage6_exploitation_only_summary.csv"
    paired_path = results_dir / "stage6_paired_seed_test.csv"
    curve_path = results_dir / "stage6_learning_curve.csv"
    checkpoint_path = results_dir / "stage6_exploitation_only.pt"
    pd.DataFrame(training_log).to_csv(training_path, index=False)
    pd.DataFrame(evaluation_rows).to_csv(evaluation_path, index=False)
    paired_frame = pd.DataFrame(paired_rows)
    differences = paired_frame["paired_difference"].to_numpy(dtype=float)
    test = ttest_rel(paired_frame["policy_total"], paired_frame["myopic_total"])
    standard_error = float(differences.std(ddof=1) / np.sqrt(len(differences)))
    critical = float(student_t.ppf(0.975, len(differences) - 1))
    paired_frame["mean_difference"] = float(differences.mean())
    paired_frame["ci_95_lower"] = float(differences.mean() - critical * standard_error)
    paired_frame["ci_95_upper"] = float(differences.mean() + critical * standard_error)
    paired_frame["paired_t_statistic"] = float(test.statistic)
    paired_frame["paired_p_value"] = float(test.pvalue)
    paired_frame.to_csv(paired_path, index=False)
    pd.DataFrame(learning_curve).to_csv(curve_path, index=False)
    import matplotlib.pyplot as plt

    curve_frame = pd.DataFrame(learning_curve)
    figure, axis = plt.subplots(figsize=(7.0, 4.0))
    axis.plot(
        curve_frame["environment_steps"],
        curve_frame["mean_episode_return"],
        color="#1f4e79",
        linewidth=2.0,
    )
    axis.set_xlabel("Environment steps")
    axis.set_ylabel("Rolling mean episode return")
    axis.set_title("SYNTHETIC INPUT: Stage 6 learning curve")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(results_dir / "stage6_learning_curve.png", dpi=180)
    plt.close(figure)
    training_frame = pd.DataFrame(training_log)
    figure, axes = plt.subplots(2, 1, figsize=(7.0, 6.5), sharex=True)
    axes[0].plot(
        curve_frame["environment_steps"],
        curve_frame["mean_episode_return"],
        color="#1f4e79",
        linewidth=2.0,
    )
    axes[0].set_ylabel("Rolling mean return")
    axes[0].grid(alpha=0.25)
    axes[1].plot(
        training_frame["environment_steps"],
        training_frame["entropy"],
        color="#8b4513",
        label="Policy entropy",
    )
    axes[1].plot(
        training_frame["environment_steps"],
        training_frame["entropy_weight"],
        color="#555555",
        label="Entropy coefficient",
    )
    axes[1].set_xlabel("Environment steps")
    axes[1].set_ylabel("Entropy / coefficient")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    figure.suptitle("SYNTHETIC INPUT: Stage 6 learning and entropy")
    figure.tight_layout()
    figure.savefig(results_dir / "stage6_learning_and_entropy.png", dpi=180)
    plt.close(figure)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "policy_config": policy_config,
            "road_classes": road_classes,
            "synthetic_input": True,
            "environment_steps": environment_steps,
            "gradient_updates": gradient_updates,
            "plateau_reached": plateau_reached,
        },
        checkpoint_path,
    )
    return training_path, evaluation_path, checkpoint_path
