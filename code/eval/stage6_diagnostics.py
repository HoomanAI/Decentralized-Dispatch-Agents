"""Post-training diagnostics for the Stage 6 exploitation-only policy."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_rel

from code.demand.patients import generate_patients
from code.env.dispatch import build_runtime_network, run_episode
from code.policy.mappo import PPOController
from code.policy.model import ExploitationActorCritic


def run_stage6_diagnostics(
    affected_population: pd.DataFrame,
    config: dict[str, Any],
    data_dir: Path,
    results_dir: Path,
) -> tuple[Path, Path]:
    """Compare greedy and sampled evaluation and record decision diagnostics."""
    policy_config = config["policy_training"]
    network_config = config["reduced_network"]
    demand_config = config["reduced_demand"]
    graph, hospitals, edge_lookup = build_runtime_network(
        data_dir / "reduced" / "synthetic_road_features.geojson",
        data_dir / "reduced" / "synthetic_fire_fronts.geojson",
        network_config,
    )
    checkpoint = torch.load(
        results_dir / "stage6_exploitation_only.pt",
        map_location="cpu",
        weights_only=False,
    )
    road_classes = list(checkpoint["road_classes"])
    model = ExploitationActorCritic(
        node_dim=3,
        edge_dim=3 + len(road_classes),
        vehicle_dim=6,
        patient_dim=6,
        hidden_dim=int(policy_config["hidden_dim"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    pds = affected_population.query("region == 'PDS'").set_index("day")
    union_population = float(pds["Total"].iloc[0])
    age_share = config["validation"]["expected_population_65_plus"] / config[
        "validation"
    ]["expected_population"]
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
    nodes = np.array(
        [node for node, attributes in graph.nodes(data=True) if not attributes.get("hospital")]
    )
    rows: list[dict[str, Any]] = []
    discount = float(policy_config["discount"])
    for evaluation_seed in range(int(policy_config["evaluation_seeds"])):
        seed_rows: dict[str, dict[str, float]] = {}
        myopic_survival_total = 0.0
        myopic_discounted_total = 0.0
        agreement_delta = 0.0
        disagreement_delta = 0.0
        no_policy_decision_delta = 0.0
        disagreement_margins: list[float] = []
        for mode in ("greedy", "sample"):
            survival_total = 0.0
            discounted_total = 0.0
            decisions = 0
            agreements = 0
            entropies: list[float] = []
            normalized_entropies: list[float] = []
            for day in range(8, 13):
                patient_seed = (
                    int(policy_config["seed"])
                    + 100000
                    + 100 * evaluation_seed
                    + day
                )
                patients = generate_patients(
                    day,
                    float(pds.loc[day, "affected_population"]),
                    union_population,
                    nodes,
                    age_share,
                    demand_config,
                    patient_seed,
                )
                torch.manual_seed(patient_seed)
                controller = PPOController(
                    model,
                    edge_lookup,
                    road_classes,
                    training=False,
                    seed=patient_seed,
                    evaluation_mode=mode,
                )
                summary, policy_trace = run_episode(
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
                rewards = np.asarray(
                    [transition.reward for transition in controller.transitions], dtype=float
                )
                survival_total += float(summary["survival_total"])
                discounted_total += float(
                    np.sum(rewards * discount ** np.arange(len(rewards)))
                )
                decisions += len(controller.myopic_agreements)
                agreements += int(sum(controller.myopic_agreements))
                entropies.extend(controller.decision_entropies)
                normalized_entropies.extend(controller.decision_normalized_entropies)
                if mode == "greedy":
                    myopic_summary, myopic_trace = run_episode(
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
                    myopic_rewards = np.asarray(
                        [
                            item["patient_reward"]
                            for item in myopic_trace
                            if item["assignment"] is not None
                        ],
                        dtype=float,
                    )
                    myopic_survival_total += float(myopic_summary["survival_total"])
                    myopic_discounted_total += float(
                        np.sum(myopic_rewards * discount ** np.arange(len(myopic_rewards)))
                    )
                    policy_reward = {
                        item["patient_id"]: float(item.get("reward", 0.0))
                        for item in controller.decision_records
                    }
                    myopic_reward = {
                        item["patient_id"]: float(item["patient_reward"])
                        for item in myopic_trace
                    }
                    records = {
                        item["patient_id"]: item for item in controller.decision_records
                    }
                    patient_ids = set(policy_reward) | set(myopic_reward)
                    for patient_id in patient_ids:
                        delta = policy_reward.get(patient_id, 0.0) - myopic_reward.get(
                            patient_id, 0.0
                        )
                        if patient_id not in records:
                            no_policy_decision_delta += delta
                        elif records[patient_id]["agrees_with_myopic"]:
                            agreement_delta += delta
                        else:
                            disagreement_delta += delta
                            disagreement_margins.append(
                                float(records[patient_id]["top1_top2_margin"])
                            )
            seed_rows[mode] = {
                "survival_total": survival_total,
                "discounted_reward_total": discounted_total,
                "decisions": decisions,
                "myopic_agreements": agreements,
                "myopic_agreement_rate": agreements / decisions if decisions else np.nan,
                "mean_action_entropy": float(np.mean(entropies)),
                "mean_normalized_action_entropy": float(np.mean(normalized_entropies)),
            }
        rows.append(
            {
                "synthetic_input": True,
                "evaluation_seed": evaluation_seed,
                **{f"greedy_{key}": value for key, value in seed_rows["greedy"].items()},
                **{f"sampled_{key}": value for key, value in seed_rows["sample"].items()},
                "sampled_minus_greedy": seed_rows["sample"]["survival_total"]
                - seed_rows["greedy"]["survival_total"],
                "myopic_survival_total": myopic_survival_total,
                "myopic_discounted_reward_total": myopic_discounted_total,
                "greedy_minus_myopic_discounted": seed_rows["greedy"][
                    "discounted_reward_total"
                ]
                - myopic_discounted_total,
                "agreement_decision_reward_delta": agreement_delta,
                "disagreement_decision_reward_delta": disagreement_delta,
                "no_policy_decision_reward_delta": no_policy_decision_delta,
                "disagreement_margin_mean": float(np.mean(disagreement_margins)),
                "disagreement_margin_median": float(np.median(disagreement_margins)),
                "disagreement_margin_below_0_05": float(
                    np.mean(np.asarray(disagreement_margins) < 0.05)
                ),
            }
        )
    frame = pd.DataFrame(rows)
    test = ttest_rel(frame["sampled_survival_total"], frame["greedy_survival_total"])
    discounted_test = ttest_rel(
        frame["greedy_discounted_reward_total"],
        frame["myopic_discounted_reward_total"],
    )
    summary = pd.DataFrame(
        [
            {
                "synthetic_input": True,
                "evaluation_seeds": len(frame),
                "greedy_mean_survival": frame["greedy_survival_total"].mean(),
                "sampled_mean_survival": frame["sampled_survival_total"].mean(),
                "sampled_minus_greedy_mean": frame["sampled_minus_greedy"].mean(),
                "sampled_vs_greedy_p_value": float(test.pvalue),
                "greedy_myopic_agreement_rate": frame["greedy_myopic_agreements"].sum()
                / frame["greedy_decisions"].sum(),
                "sampled_myopic_agreement_rate": frame["sampled_myopic_agreements"].sum()
                / frame["sampled_decisions"].sum(),
                "greedy_mean_action_entropy": np.average(
                    frame["greedy_mean_action_entropy"], weights=frame["greedy_decisions"]
                ),
                "greedy_mean_normalized_action_entropy": np.average(
                    frame["greedy_mean_normalized_action_entropy"],
                    weights=frame["greedy_decisions"],
                ),
                "greedy_mean_discounted_reward": frame[
                    "greedy_discounted_reward_total"
                ].mean(),
                "greedy_mean_undiscounted_reward": frame[
                    "greedy_survival_total"
                ].mean(),
                "myopic_mean_discounted_reward": frame[
                    "myopic_discounted_reward_total"
                ].mean(),
                "greedy_minus_myopic_discounted_mean": frame[
                    "greedy_minus_myopic_discounted"
                ].mean(),
                "greedy_minus_myopic_discounted_p_value": float(
                    discounted_test.pvalue
                ),
                "agreement_decision_reward_delta_mean": frame[
                    "agreement_decision_reward_delta"
                ].mean(),
                "disagreement_decision_reward_delta_mean": frame[
                    "disagreement_decision_reward_delta"
                ].mean(),
                "no_policy_decision_reward_delta_mean": frame[
                    "no_policy_decision_reward_delta"
                ].mean(),
                "disagreement_margin_mean": frame[
                    "disagreement_margin_mean"
                ].mean(),
                "disagreement_margin_median": frame[
                    "disagreement_margin_median"
                ].median(),
                "disagreement_margin_below_0_05": frame[
                    "disagreement_margin_below_0_05"
                ].mean(),
            }
        ]
    )
    detail_path = results_dir / "stage6_diagnostic_seed_results.csv"
    summary_path = results_dir / "stage6_diagnostic_summary.csv"
    frame.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)

    training = pd.read_csv(results_dir / "stage6_training_log.csv")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.0, 4.0))
    axis.plot(training["environment_steps"], training["entropy"], color="#8b4513")
    axis.set_xlabel("Environment steps")
    axis.set_ylabel("Policy entropy")
    axis.set_title("SYNTHETIC INPUT: Stage 6 training entropy")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(results_dir / "stage6_entropy_trace.png", dpi=180)
    plt.close(figure)
    return detail_path, summary_path
