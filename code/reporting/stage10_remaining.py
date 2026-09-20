"""Build Stage 10 outputs supported by completed Stages 1 through 7."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]

from code.reliability.model import compute_damage
from code.reporting.stage10_load_bearing import COLORS, configure_style, save_figure
from code.demand.patients import generate_patients
from code.env.vehicle_centric import ProtocolMyopic, run_vehicle_episode
from code.eval.stage6 import _perturb_exposure
from code.eval.stage6_vehicle import _setup


RESULTS = ROOT / "results"
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"
DATA = ROOT / "data"


def build_t2() -> pd.DataFrame:
    """Compare the full and reduced synthetic instances."""
    with (DATA / "synthetic_network_validation.json").open(encoding="utf-8") as stream:
        full = json.load(stream)
    with (DATA / "reduced" / "synthetic_network_validation.json").open(encoding="utf-8") as stream:
        reduced = json.load(stream)
    full_eval = pd.read_csv(RESULTS / "stage7_full_communication_evaluation.csv")
    full_eval = full_eval[
        (full_eval.communication_reliability == 1.0)
        & (full_eval.method == "belief_aware")
    ]
    reduced_eval = pd.read_csv(RESULTS / "stage7_communication_selected_evaluation.csv")
    reduced_eval = reduced_eval[
        (reduced_eval.communication_reliability == 1.0)
        & (reduced_eval.method == "belief_aware")
    ]

    def row(label: str, validation: dict, evaluation: pd.DataFrame) -> dict:
        totals = evaluation.groupby("seed").survival_total.sum()
        day12 = float(evaluation[evaluation.day == 12].survival_total.mean())
        five_day = float(totals.mean())
        return {
            "instance": label,
            "nodes": validation["node_count"],
            "routable_arcs": validation["routable_edge_count"],
            "mean_degree": validation["mean_node_degree"],
            "dead_end_fraction": validation["dead_end_share"],
            "top5_betweenness_share": validation["top_5_percent_edge_betweenness_share"],
            "arterial_arcs": validation["arterial_count"],
            "local_arcs": validation["local_count"],
            "five_day_survival": five_day,
            "day12_survival": day12,
            "day12_share_percent": 100.0 * day12 / five_day,
        }

    result = pd.DataFrame([row("Full", full, full_eval), row("Reduced", reduced, reduced_eval)])
    result.to_csv(TABLES / "T2_instance_characteristics.csv", index=False)
    result.to_latex(
        TABLES / "T2_instance_characteristics.tex",
        index=False,
        float_format="%.4f",
        caption="Full and reduced synthetic instance characteristics and matched belief-aware dispatch outcomes.",
        label="tab:instance-characteristics",
    )
    return result


def build_t3() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the scenario grid and risk-posture memory panel."""
    grid = pd.read_csv(RESULTS / "reliability_scenario_grid.csv")
    reversal = pd.read_csv(RESULTS / "stage4_reversal_robustness.csv")
    reversal = reversal[~reversal.heterogeneous_alpha].copy()
    grid = grid.merge(
        reversal[["alpha", "gamma", "recovery_holds", "relapse_holds", "reversal_holds"]],
        on=["alpha", "gamma"],
        how="left",
    )
    grid["parameter_status"] = "scenario input, not estimate"
    grid.to_csv(TABLES / "T3a_reliability_scenario_grid.csv", index=False)

    memory = pd.read_csv(RESULTS / "stage4_memory_phi_sweep.csv")
    posture = memory[
        (memory.alpha == 0.75)
        & (memory.gamma == 0.75)
        & (~memory.heterogeneous_alpha)
    ].copy()
    columns = [
        "phi_min",
        "day_7_blocked_not_exposed",
        "day_8_blocked_not_exposed",
        "day_9_blocked_not_exposed",
        "day_10_blocked_not_exposed",
        "day_11_blocked_not_exposed",
        "day_12_blocked_not_exposed",
        "memory_effect_total",
        "memory_active",
    ]
    posture = posture[columns].sort_values("phi_min")
    posture.to_csv(TABLES / "T3b_phi_min_memory_effect.csv", index=False)

    grid_tex = grid.to_latex(index=False, float_format="%.2f", escape=True)
    posture_tex = posture.to_latex(index=False, float_format="%.2f", escape=True)
    content = (
        "% SYNTHETIC INPUT\n"
        "\\begin{table*}\n\\centering\n"
        "\\caption{Scenario reliability grid and blocked-but-unexposed memory effect by dispatcher risk posture.}\n"
        "\\label{tab:reliability-region}\n"
        "\\textit{Panel A: scenario grid (parameters are not estimates).}\\par\n"
        f"{grid_tex}\n"
        "\\textit{Panel B: risk-posture sweep at $\\alpha=\\gamma=0.75$.}\\par\n"
        f"{posture_tex}\n" + "\\end{table*}\n"
    )
    (TABLES / "T3_reliability_scenario_region.tex").write_text(content, encoding="utf-8")
    return grid, posture


def build_perfect_information_audit() -> pd.DataFrame:
    """Document the distinct estimands behind the two day-12 contrasts."""
    stage4_aware = pd.read_csv(RESULTS / "stage4_myopic_summary.csv")
    stage4_aware = float(
        stage4_aware.loc[
            (stage4_aware.day == 12) & (stage4_aware.baseline == "belief_aware"),
            "survival_total",
        ].iloc[0]
    )
    stage4_perfect = float(
        pd.read_csv(RESULTS / "perfect_information_myopic.csv")
        .loc[lambda x: x.day == 12, "survival_total"]
        .iloc[0]
    )
    bridge = pd.read_csv(RESULTS / "stage7_stage4_seed_bridge.csv").set_index("method")
    evaluation = pd.read_csv(RESULTS / "stage7_full_communication_evaluation.csv")
    evaluation = evaluation[
        (evaluation.communication_reliability == 1.0) & (evaluation.day == 12)
    ]
    means = evaluation.groupby("method").survival_total.mean()
    result = pd.DataFrame(
        [
            {
                "estimand": "Stage 4 fixed demand, patient-centric dispatcher",
                "seed_count": 1,
                "belief_aware": stage4_aware,
                "perfect_information": stage4_perfect,
                "contrast": stage4_perfect - stage4_aware,
            },
            {
                "estimand": "Stage 4 demand seed, vehicle-centric protocol",
                "seed_count": 1,
                "belief_aware": bridge.loc["belief_aware", "survival_total"],
                "perfect_information": bridge.loc["perfect_information", "survival_total"],
                "contrast": bridge.loc["perfect_information", "survival_total"]
                - bridge.loc["belief_aware", "survival_total"],
            },
            {
                "estimand": "Stage 7 matched vehicle-centric evaluation",
                "seed_count": 30,
                "belief_aware": means.belief_aware,
                "perfect_information": means.perfect_information,
                "contrast": means.perfect_information - means.belief_aware,
            },
        ]
    )
    result.to_csv(TABLES / "T4_perfect_information_estimand_audit.csv", index=False)
    return result


def build_connectivity_audit() -> pd.DataFrame:
    """Summarize whether q=0.75 ever partitions the fleet."""
    audit = pd.read_csv(RESULTS / "stage7_q0_75_partition_audit.csv")
    epochs = int(audit.communication_epochs.sum())
    partitioned = int(audit.partitioned_epochs.sum())
    result = pd.DataFrame(
        [
            {
                "communication_reliability": 0.75,
                "communication_epochs": epochs,
                "partitioned_epochs": partitioned,
                "partition_fraction": partitioned / epochs,
                "maximum_components": int(audit.maximum_component_count.max()),
                "partition_seed": int(audit.loc[audit.partitioned_epochs > 0, "seed"].iloc[0]),
                "partition_day": int(audit.loc[audit.partitioned_epochs > 0, "day"].iloc[0]),
            }
        ]
    )
    result.to_csv(TABLES / "T4_connectivity_partition_audit.csv", index=False)
    return result


def build_f4(grid: pd.DataFrame) -> None:
    """Plot damage envelopes for representative arcs and clearance half-life."""
    days = np.arange(7, 13)
    arterial_exposure = np.array([[1, 1, 0, 0, 0, 1]], dtype=float)
    local_exposure = np.array([[1, 0, 0, 0, 0, 1]], dtype=float)
    trajectories = {"Arterial": [], "Local": []}
    for item in grid.itertuples():
        for label, exposure in (("Arterial", arterial_exposure), ("Local", local_exposure)):
            damage = compute_damage(
                exposure,
                np.array([0]),
                np.array([item.alpha]),
                np.array([item.gamma]),
            )[0]
            trajectories[label].append(damage)

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.15))
    ax = axes[0]
    for label, key, exposure, marker in (
        ("Arterial", "belief_aware", arterial_exposure, "o"),
        ("Local", "best_positive_ration", local_exposure, "s"),
    ):
        values = np.vstack(trajectories[label])
        ax.fill_between(days, values.min(axis=0), values.max(axis=0), color=COLORS[key], alpha=0.18)
        operating = compute_damage(
            exposure, np.array([0]), np.array([0.75]), np.array([0.75])
        )[0]
        ax.plot(days, operating, color=COLORS[key], marker=marker, label=f"{label}, operating point")
    ax.set_xlabel("Day")
    ax.set_ylabel("Latent damage")
    ax.set_xticks(days)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(color="0.88", linewidth=0.5)
    ax.legend(frameon=False, loc="upper right")
    ax.text(-0.17, 1.04, "(a)", transform=ax.transAxes, fontsize=7)

    ax = axes[1]
    half_life = grid[["gamma", "half_life_days"]].drop_duplicates().sort_values("gamma")
    ax.plot(
        half_life.gamma,
        half_life.half_life_days,
        color=COLORS["perfect_information"],
        marker="^",
    )
    ax.set_xlabel("Daily clearance rate, gamma")
    ax.set_ylabel("Clearance half-life (days)")
    ax.grid(color="0.88", linewidth=0.5)
    ax.text(-0.17, 1.04, "(b)", transform=ax.transAxes, fontsize=7)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.18, top=0.95, wspace=0.30)
    save_figure(fig, "F4_damage_recovery")


def build_f5() -> None:
    """Plot blocked arcs and unreachable patients as percentages of their denominators.

    The full network carries 1,297 routable arcs against 350 in the reduced one, so raw
    counts are not comparable across instances.
    """
    full = pd.read_csv(RESULTS / "stage4_myopic_summary.csv")
    full = full[full.baseline == "belief_aware"].set_index("day")
    reduced = pd.read_csv(RESULTS / "stage6_exploitation_only_summary.csv")
    reduced = reduced[reduced.method == "belief_aware_myopic"].groupby("day").mean(numeric_only=True)
    days = np.arange(8, 13)
    arcs_full, arcs_reduced = 1297.0, 350.0

    full_arc_pct = full.loc[days, "blocked_arc_count"].to_numpy() / arcs_full * 100.0
    red_arc_pct = reduced.loc[days, "blocked_arc_count"].to_numpy() / arcs_reduced * 100.0
    full_pat_pct = (full.loc[days, "unreachable"].to_numpy()
                    / full.loc[days, "patients"].to_numpy() * 100.0)
    red_pat_pct = (reduced.loc[days, "unreachable"].to_numpy()
                   / reduced.loc[days, "patients"].to_numpy() * 100.0)

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.9))
    for ax, (a, b), label in (
        (axes[0], (full_arc_pct, red_arc_pct), "Blocked arcs (% of routable arcs)"),
        (axes[1], (full_pat_pct, red_pat_pct), "Unreachable patients (% of demand)"),
    ):
        ax.plot(days, a, color=COLORS["belief_aware"], marker="o", label="Full (1,297 arcs)")
        ax.plot(days, b, color=COLORS["best_positive_ration"], marker="s",
                linestyle="--", label="Reduced (350 arcs)")
        ax.set_xlabel("Day")
        ax.set_ylabel(label)
        ax.set_xticks(days)
        ax.set_ylim(bottom=0)
        ax.grid(color="0.88", linewidth=0.5)
    axes[0].text(-0.15, 1.04, "(a)", transform=axes[0].transAxes, fontsize=7)
    axes[1].text(-0.15, 1.04, "(b)", transform=axes[1].transAxes, fontsize=7)
    axes[0].legend(frameon=False, fontsize=7)
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.19, top=0.93, wspace=0.28)
    save_figure(fig, "F5_network_state")


def build_f8() -> None:
    """Plot matched from-scratch and warm-started training traces."""
    scratch = pd.read_csv(RESULTS / "stage6_training_log.csv")
    warm = pd.read_csv(RESULTS / "stage6_vehicle_counterfactual_training.csv")
    reference_path = RESULTS / "stage6_protocol_training_reference.csv"
    if reference_path.exists():
        reference = pd.read_csv(reference_path)
    else:
        config = yaml.safe_load((ROOT / "code" / "config" / "default.yaml").read_text(encoding="utf-8"))
        affected = pd.read_csv(DATA / "daily_affected_population_clean.csv")
        pc, nc, dc, graph, hospitals, _, pds, union, age, env, nodes = _setup(
            affected, config, DATA
        )
        modes = list(pc["augmentation_modes"])
        augmented = {
            mode: _perturb_exposure(
                graph,
                DATA / "reduced" / "synthetic_fire_fronts.geojson",
                mode,
                float(pc["augmentation_buffer_degrees"]),
                float(pc["augmentation_translation_degrees"]),
            )
            for mode in set(modes)
        }
        fleet_count = sum(int(value) for value in pc["fleet_by_class"].values())
        rows = []
        for episode in range(200):
            day = int(pc["train_days"][episode % len(pc["train_days"])])
            mode = modes[episode % len(modes)]
            training_graph = augmented[mode]
            lookup = {
                int(attributes["edge_id"]): index
                for index, (_, _, attributes) in enumerate(training_graph.edges(data=True))
            }
            patient_seed = int(pc["seed"]) + 500000 + episode
            patients = generate_patients(
                day,
                float(pds.loc[day, "affected_population"]),
                union,
                nodes,
                age,
                dc,
                patient_seed,
            )
            summary, _ = run_vehicle_episode(
                training_graph,
                hospitals,
                lookup,
                patients,
                day,
                [ProtocolMyopic() for _ in range(fleet_count)],
                nc,
                config["belief"],
                env,
                float(pds.loc[day, "affected_population"]),
                seed=patient_seed,
            )
            rows.append({"episode": episode, "day": day, "survival_total": summary["survival_total"]})
        reference = pd.DataFrame(rows)
        reference.to_csv(reference_path, index=False)
    protocol_training_reference = float(reference.survival_total.mean())
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.1))
    for frame, label, key, linestyle in (
        (scratch, "From scratch", "from_scratch", "-"),
        (warm, "Warm start", "counterfactual", "--"),
    ):
        survival_column = (
            "mean_episode_survival" if "mean_episode_survival" in frame else "recent_mean_return"
        )
        survival = frame[survival_column].rolling(25, min_periods=5).mean()
        entropy = frame.entropy.rolling(25, min_periods=5).mean()
        axes[0].plot(frame[frame.columns[0]], survival, color=COLORS[key], linestyle=linestyle, label=label)
        axes[1].plot(frame[frame.columns[0]], entropy, color=COLORS[key], linestyle=linestyle, label=label)
    axes[0].axhline(
        protocol_training_reference,
        color="0.35",
        linestyle=":",
        linewidth=1.0,
    )
    axes[0].annotate(
        f"Protocol myopic reference {protocol_training_reference:.2f}",
        (620, protocol_training_reference),
        xytext=(-4, 4),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=7,
    )
    axes[0].text(
        0.98,
        0.04,
        "Greedy evaluation: scratch -3.81; warm +0.004",
        transform=axes[0].transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
    )
    axes[0].set_xlabel("Optimizer update")
    axes[0].set_ylabel("Mean episode survival")
    axes[1].set_xlabel("Optimizer update")
    axes[1].set_ylabel("Mean policy entropy (nats)")
    scratch_terminal = scratch.entropy.rolling(25, min_periods=5).mean().iloc[-1]
    warm_terminal = warm.entropy.rolling(25, min_periods=5).mean().iloc[-1]
    axes[1].annotate(
        f"{scratch_terminal:.2f} nats; 81.12% evaluation",
        (scratch.iloc[-1, 0], scratch_terminal),
        xytext=(-5, 7),
        textcoords="offset points",
        ha="right",
        fontsize=7,
    )
    axes[1].annotate(
        f"{warm_terminal:.2f} nats; 9.37% evaluation",
        (warm.iloc[-1, 0], warm_terminal),
        xytext=(-5, 7),
        textcoords="offset points",
        ha="right",
        fontsize=7,
    )
    for index, ax in enumerate(axes):
        ax.grid(color="0.88", linewidth=0.5)
        ax.text(-0.17, 1.04, f"({chr(ord('a') + index)})", transform=ax.transAxes, fontsize=7)
    axes[0].legend(frameon=True, facecolor="white", edgecolor="none", loc="center right")
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.18, top=0.95, wspace=0.30)
    save_figure(fig, "F8_learning_curves")


def build_f11() -> None:
    """Plot detour use and its survival trade across connectivity."""
    evaluation = pd.read_csv(RESULTS / "stage7_full_communication_evaluation.csv")
    ration = evaluation[evaluation.method == "best_positive_ration"].copy()
    aware = evaluation[evaluation.method == "belief_aware"].copy()
    detours = ration.groupby("communication_reliability").detour_count.mean().sort_index()
    cost = ration.assign(cost=lambda x: x.exploration_cost_1 + x.exploration_cost_2 + x.exploration_cost_3)
    cost = cost.groupby(["communication_reliability", "seed"]).cost.sum().groupby(level=0).mean().sort_index()
    paired = evaluation.pivot_table(
        index=["communication_reliability", "seed", "day"], columns="method", values="survival_total"
    ).reset_index()
    paired["effect"] = paired.best_positive_ration - paired.belief_aware
    benefit = paired.groupby(["communication_reliability", "seed"]).effect.sum().groupby(level=0).mean().sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.1))
    axes[0].plot(detours.index, detours.values, color=COLORS["best_positive_ration"], marker="s")
    axes[0].set_xlabel("Communication reliability")
    axes[0].set_ylabel("Mean detours per day")
    axes[1].plot(benefit.index, benefit.values, color=COLORS["belief_aware"], marker="o", label="Net survival effect")
    axes[1].plot(cost.index, cost.values, color=COLORS["best_positive_ration"], marker="s", linestyle="--", label="Five-day total detour cost")
    axes[1].axhline(0, color="0.35", linewidth=0.7)
    axes[1].set_xlabel("Communication reliability")
    axes[1].set_ylabel("Five-day survival units")
    axes[1].legend(frameon=False)
    for index, ax in enumerate(axes):
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.grid(axis="y", color="0.88", linewidth=0.5)
        ax.text(-0.17, 1.04, f"({chr(ord('a') + index)})", transform=ax.transAxes, fontsize=7)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.18, top=0.95, wspace=0.30)
    save_figure(fig, "F11_detour_tradeoff")


def append_captions() -> None:
    path = FIGURES / "STAGE10_CAPTIONS.md"
    existing = path.read_text(encoding="utf-8")
    additions = """
F4. Latent damage envelopes and operating-point trajectories beside scenario clearance half-lives.

F5. Daily blocked arcs and unreachable patients on full and reduced synthetic instances.

F8. Smoothed training survival and raw entropy, with protocol reference and terminal normalized evaluation entropy.

F11. Detour frequency, survival cost, and net survival effect across communication reliability.
"""
    if "F4. Latent damage envelopes" not in existing:
        path.write_text(existing.rstrip() + "\n\n" + additions.lstrip(), encoding="utf-8")


def main() -> None:
    configure_style()
    TABLES.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    build_t2()
    grid, _ = build_t3()
    build_perfect_information_audit()
    build_connectivity_audit()
    build_f4(grid)
    build_f5()
    build_f8()
    build_f11()
    append_captions()


if __name__ == "__main__":
    main()
