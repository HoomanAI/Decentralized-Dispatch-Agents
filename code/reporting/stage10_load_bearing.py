"""Build the load-bearing Stage 10 tables and figures from completed results."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"

# Fixed Okabe-Ito mapping for every Stage 10 figure.
COLORS = {
    "belief_aware": "#0072B2",
    "best_positive_ration": "#E69F00",
    "perfect_information": "#009E73",
    "oracle": "#CC79A7",
    "reachable_union": "#56B4E9",
    "reachable_realization": "#D55E00",
    "from_scratch": "#000000",
    "clone": "#0072B2",
    "scalar": "#E69F00",
    "counterfactual": "#009E73",
    "softened": "#CC79A7",
}
MARKERS = {
    "belief_aware": "o",
    "best_positive_ration": "s",
    "perfect_information": "^",
    "oracle": "D",
    "reachable_union": "v",
    "reachable_realization": "P",
    "from_scratch": "X",
    "clone": "o",
    "scalar": "s",
    "counterfactual": "^",
    "softened": "D",
}


def configure_style() -> None:
    """Apply manuscript-wide typography and vector export settings."""
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.2,
            "lines.markersize": 4.5,
        }
    )


def paired_summary(values: Iterable[float]) -> tuple[float, float, float, float]:
    """Return mean, 95% t interval, and two-sided one-sample p-value."""
    array = np.asarray(list(values), dtype=float)
    mean = float(array.mean())
    lower, upper = stats.t.interval(
        0.95,
        len(array) - 1,
        loc=mean,
        scale=stats.sem(array),
    )
    p_value = float(stats.ttest_1samp(array, 0.0).pvalue)
    return mean, float(lower), float(upper), p_value


def save_figure(fig: plt.Figure, stem: str) -> None:
    """Write vector PDF and 600 dpi PNG."""
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.pdf")
    fig.savefig(FIGURES / f"{stem}.png", dpi=600)
    plt.close(fig)


def load_evaluation() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "stage7_full_communication_evaluation.csv")


def load_reachable() -> dict[float, pd.DataFrame]:
    frames = {}
    for path in RESULTS.glob("stage7_full_reachable_headroom_q*.csv"):
        frame = pd.read_csv(path)
        frames[float(frame["communication_reliability"].iloc[0])] = frame
    return frames


def build_t4() -> pd.DataFrame:
    """Build the connectivity-indexed information-contrast ladder."""
    evaluation = load_evaluation()
    reachable = load_reachable()
    oracle = pd.read_csv(RESULTS / "stage5_travel_capacity_check.csv")
    oracle_day12_gap = float(oracle.loc[oracle.day == 12, "reportable_gap"].iloc[0])
    rows = []
    for q in sorted(evaluation.communication_reliability.unique(), reverse=True):
        level = evaluation[evaluation.communication_reliability == q]
        daily = level.pivot_table(
            index=["seed", "day"], columns="method", values="survival_total"
        ).reset_index()
        totals = level.groupby(["seed", "method"]).survival_total.sum().unstack()
        day12 = daily[daily.day == 12]
        audit = reachable[float(q)]
        union = audit[audit.bound_type == "union"]
        per = audit[audit.bound_type != "union"].copy()
        per["day12_contrast"] = per.reachable_day12 - per.aware_day12
        per["five_day_contrast"] = per.reachable_five_day - per.aware_five_day
        per_mean = float(per.day12_contrast.mean())
        achieved_12 = float(
            (day12.best_positive_ration - day12.belief_aware).mean()
        )
        achieved_5 = float(
            (totals.best_positive_ration - totals.belief_aware).mean()
        )
        rows.append(
            {
                "communication": q,
                "exact_oracle_gap_day12_reference": oracle_day12_gap,
                "perfect_information_contrast_day12": float(
                    (day12.perfect_information - day12.belief_aware).mean()
                ),
                "perfect_information_contrast_five_day": float(
                    (totals.perfect_information - totals.belief_aware).mean()
                ),
                "reachable_union_contrast_day12": float(
                    (union.reachable_day12 - union.aware_day12).mean()
                ),
                "reachable_union_contrast_five_day": float(
                    (union.reachable_five_day - union.aware_five_day).mean()
                ),
                "reachable_per_realization_min_day12": float(
                    per.day12_contrast.min()
                ),
                "reachable_per_realization_mean_day12": per_mean,
                "reachable_per_realization_min_five_day": float(
                    per.five_day_contrast.min()
                ),
                "reachable_per_realization_mean_five_day": float(
                    per.five_day_contrast.mean()
                ),
                "break_even_percent": 100.0 * 0.25 / per_mean,
                "achieved_day12": achieved_12,
                "achieved_five_day": achieved_5,
            }
        )
    result = pd.DataFrame(rows)
    TABLES.mkdir(exist_ok=True)
    result.to_csv(TABLES / "T4_bounds_ladder.csv", index=False)
    result.to_latex(
        TABLES / "T4_bounds_ladder.tex",
        index=False,
        float_format="%.4f",
        na_rep="not available",
        caption=(
            "Information and optimization contrasts by communication level. "
            "All inputs are synthetic."
        ),
        label="tab:bounds-ladder",
    )
    q0 = evaluation[evaluation.communication_reliability == 0.0]
    q0_pivot = q0.pivot_table(
        index=["seed", "day"],
        columns="method",
        values=["survival_total", "served", "duplicate_assignment_rate"],
    ).reset_index()
    decomposition = []
    for day, frame in q0_pivot.groupby("day"):
        contrast = (
            frame[("survival_total", "perfect_information")]
            - frame[("survival_total", "belief_aware")]
        )
        mean, lower, upper, p_value = paired_summary(contrast)
        decomposition.append(
            {
                "day": int(day),
                "perfect_information_contrast": mean,
                "ci_95_lower": lower,
                "ci_95_upper": upper,
                "paired_p_value": p_value,
                "belief_aware_served": float(frame[("served", "belief_aware")].mean()),
                "perfect_information_served": float(frame[("served", "perfect_information")].mean()),
                "belief_aware_duplicate_rate": float(
                    frame[("duplicate_assignment_rate", "belief_aware")].mean()
                ),
                "perfect_information_duplicate_rate": float(
                    frame[("duplicate_assignment_rate", "perfect_information")].mean()
                ),
            }
        )
    pd.DataFrame(decomposition).to_csv(
        TABLES / "T4_q0_perfect_information_decomposition.csv", index=False
    )
    return result


def evaluation_row(
    label: str,
    frame: pd.DataFrame,
    difference: str,
    entropy: str,
    agreement: str,
    steps: int,
) -> dict[str, float | str | int]:
    mean, lower, upper, p_value = paired_summary(frame[difference])
    return {
        "variant": label,
        "normalized_entropy": float(frame[entropy].mean()),
        "argmax_agreement": float(frame[agreement].mean()),
        "paired_difference": mean,
        "ci_95_lower": lower,
        "ci_95_upper": upper,
        "paired_p_value": p_value,
        "environment_steps": steps,
    }


def build_t5() -> pd.DataFrame:
    """Build the policy-learning ablation table."""
    scratch = pd.read_csv(RESULTS / "stage6_paired_seed_test.csv")
    scratch_mean, scratch_lo, scratch_hi, scratch_p = paired_summary(
        scratch.paired_difference
    )
    rows: list[dict[str, float | str | int]] = [
        {
            "variant": "From scratch",
            "normalized_entropy": 0.8112,
            "argmax_agreement": 0.6680,
            "paired_difference": scratch_mean,
            "ci_95_lower": scratch_lo,
            "ci_95_upper": scratch_hi,
            "paired_p_value": scratch_p,
            "environment_steps": 500015,
        }
    ]
    clone = pd.read_csv(RESULTS / "stage6_vehicle_clone_evaluation.csv")
    rows.append(
        evaluation_row(
            "Supervised clone",
            clone,
            "difference",
            "normalized_entropy",
            "agreement",
            0,
        )
    )
    specifications = [
        (
            "Residual warm start, scalar critic",
            "stage6_vehicle_scalar_matched_evaluation.csv",
            "stage6_vehicle_scalar_matched_training.csv",
        ),
        (
            "Residual warm start, counterfactual critic",
            "stage6_vehicle_counterfactual_evaluation.csv",
            "stage6_vehicle_counterfactual_training.csv",
        ),
        (
            "Softened clone, counterfactual critic",
            "stage6_vehicle_counterfactual_softened_evaluation.csv",
            "stage6_vehicle_counterfactual_softened_training.csv",
        ),
    ]
    for label, evaluation_name, training_name in specifications:
        frame = pd.read_csv(RESULTS / evaluation_name)
        training = pd.read_csv(RESULTS / training_name)
        rows.append(
            evaluation_row(
                label,
                frame,
                "paired_difference",
                "normalized_entropy",
                "argmax_agreement",
                int(training.environment_steps.iloc[-1]),
            )
        )
    result = pd.DataFrame(rows)
    result.to_csv(TABLES / "T5_policy_learning_ablation.csv", index=False)
    result.to_latex(
        TABLES / "T5_policy_learning_ablation.tex",
        index=False,
        float_format="%.4f",
        caption=(
            "Policy-learning ablation against belief-aware protocol myopic dispatch "
            "on the reduced synthetic training instance."
        ),
        label="tab:policy-ablation",
    )
    return result


def build_t6() -> pd.DataFrame:
    """Build paired ration effects by connectivity."""
    evaluation = load_evaluation()
    boundary = RESULTS / "stage7_q0_10_evaluation.csv"
    if boundary.exists():
        evaluation = pd.concat([evaluation, pd.read_csv(boundary)], ignore_index=True)
    rows = []
    high_effects: dict[int, list[float]] = {}
    zero_effects: dict[int, float] = {}
    for q in sorted(evaluation.communication_reliability.unique(), reverse=True):
        level = evaluation[evaluation.communication_reliability == q]
        pivot = level.pivot_table(
            index=["seed", "day"], columns="method", values="survival_total"
        ).reset_index()
        pivot["effect"] = pivot.best_positive_ration - pivot.belief_aware
        day12 = pivot[pivot.day == 12].set_index("seed").effect
        days9_11 = pivot[pivot.day.isin([9, 10, 11])].groupby("seed").effect.sum()
        summary12 = paired_summary(day12)
        summary9 = paired_summary(days9_11)
        ration = level[level.method == "best_positive_ration"]
        detours = float(ration.detour_count.mean())
        rows.append(
            {
                "row": f"q={q:.2f}",
                "posture": "permissive_memoryless",
                "communication": q,
                "day12_difference": summary12[0],
                "day12_ci_lower": summary12[1],
                "day12_ci_upper": summary12[2],
                "day12_p": summary12[3],
                "days9_11_difference": summary9[0],
                "days9_11_ci_lower": summary9[1],
                "days9_11_ci_upper": summary9[2],
                "days9_11_p": summary9[3],
                "detours_per_day": detours,
                "n": len(days9_11),
            }
        )
        for seed, value in days9_11.items():
            if q in (0.5, 0.75, 1.0):
                high_effects.setdefault(int(seed), []).append(float(value))
            if q == 0.0:
                zero_effects[int(seed)] = float(value)
    interaction = np.array(
        [zero_effects[seed] - np.mean(high_effects[seed]) for seed in zero_effects]
    )
    interaction_summary = paired_summary(interaction)
    conservative_path = RESULTS / "stage7_conservative_posture.csv"
    if conservative_path.exists():
        conservative = pd.read_csv(conservative_path)
        for q in [1.0, 0.0]:
            level = conservative[conservative.communication_reliability == q]
            replication_path = RESULTS / "stage7_conservative_q0_replication.csv"
            if q == 0.0 and replication_path.exists():
                level = pd.concat([level, pd.read_csv(replication_path)], ignore_index=True)
            pivot = level.pivot_table(
                index=["seed", "day"], columns="method", values="survival_total"
            ).reset_index()
            pivot["effect"] = pivot.positive_ration - pivot.belief_aware
            summary12 = paired_summary(pivot[pivot.day == 12].set_index("seed").effect)
            summary9 = paired_summary(
                pivot[pivot.day.isin([9, 10, 11])].groupby("seed").effect.sum()
            )
            ration = level[level.method == "positive_ration"]
            rows.append(
                {
                    "row": f"q={q:.2f}, conservative active memory",
                    "posture": "conservative_active_memory",
                    "communication": q,
                    "day12_difference": summary12[0],
                    "day12_ci_lower": summary12[1],
                    "day12_ci_upper": summary12[2],
                    "day12_p": summary12[3],
                    "days9_11_difference": summary9[0],
                    "days9_11_ci_lower": summary9[1],
                    "days9_11_ci_upper": summary9[2],
                    "days9_11_p": summary9[3],
                    "detours_per_day": float(ration.detour_count.mean()),
                    "n": len(pivot.seed.unique()),
                }
            )
    rows.append(
        {
            "row": "q=0 minus mean(q=0.50,0.75,1.00)",
            "posture": "permissive_memoryless",
            "communication": np.nan,
            "day12_difference": np.nan,
            "day12_ci_lower": np.nan,
            "day12_ci_upper": np.nan,
            "day12_p": np.nan,
            "days9_11_difference": interaction_summary[0],
            "days9_11_ci_lower": interaction_summary[1],
            "days9_11_ci_upper": interaction_summary[2],
            "days9_11_p": interaction_summary[3],
            "detours_per_day": np.nan,
            "n": len(interaction),
        }
    )
    replication = pd.read_csv(RESULTS / "stage7_communication_replication.csv")
    replication_pivot = replication.pivot_table(
        index=["seed", "day"], columns="method", values="survival_total"
    ).reset_index()
    replication_pivot["effect"] = (
        replication_pivot.positive_ration - replication_pivot.belief_aware
    )
    replication_effect = (
        replication_pivot[replication_pivot.day.isin([9, 10, 11])]
        .groupby("seed")
        .effect.sum()
    )
    replication_summary = paired_summary(replication_effect)
    rows.append(
        {
            "row": "q=0 confirmatory, fresh seeds 30-89",
            "posture": "permissive_memoryless",
            "communication": np.nan,
            "day12_difference": np.nan,
            "day12_ci_lower": np.nan,
            "day12_ci_upper": np.nan,
            "day12_p": np.nan,
            "days9_11_difference": replication_summary[0],
            "days9_11_ci_lower": replication_summary[1],
            "days9_11_ci_upper": replication_summary[2],
            "days9_11_p": replication_summary[3],
            "detours_per_day": float(
                replication[replication.method == "positive_ration"].detour_count.mean()
            ),
            "n": len(replication_effect),
        }
    )
    result = pd.DataFrame(rows)
    result.to_csv(TABLES / "T6_ration_by_connectivity.csv", index=False)
    result.to_latex(
        TABLES / "T6_ration_by_connectivity.tex",
        index=False,
        float_format="%.4f",
        na_rep="not applicable",
        caption=(
            "Paired ration effects by communication reliability and reliability posture, "
            "including the fresh-seed confirmation under complete isolation."
        ),
        label="tab:ration-connectivity",
    )
    return result


def build_f6(t4: pd.DataFrame) -> None:
    """Plot the day-12 optimization and information contrast ladder."""
    fig, ax = plt.subplots(figsize=(7.16, 3.7))
    levels = t4.communication.to_numpy()
    y = np.arange(len(levels))
    series = [
        ("Perfect information", "perfect_information_contrast_day12", "perfect_information", "\\\\"),
        ("Reachable union", "reachable_union_contrast_day12", "reachable_union", "xx"),
        ("Reachable per-realization mean", "reachable_per_realization_mean_day12", "reachable_realization", ".."),
        ("Achieved ration", "achieved_day12", "best_positive_ration", "--"),
    ]
    height = 0.14
    offsets = np.linspace(-0.28, 0.28, len(series))
    for offset, (label, column, key, hatch) in zip(offsets, series):
        ax.barh(
            y + offset,
            t4[column],
            height=height,
            color=COLORS[key],
            edgecolor="black",
            linewidth=0.35,
            hatch=hatch,
            label=label,
        )
    ax.axvline(0, color="0.25", linewidth=0.7)
    achieved_offset = offsets[-1]
    for row, value in enumerate(t4.achieved_day12):
        ax.text(
            0.04,
            y[row] + achieved_offset,
            f"{value:+.3f}",
            ha="left",
            va="center",
            fontsize=7,
            color="black",
        )
    ax.set_yticks(y, [f"{value:.2f}" for value in levels])
    ax.set_ylabel("Communication reliability")
    ax.set_xlabel("Day-12 survival contrast against belief-aware dispatch")
    ax.grid(axis="x", color="0.85", linewidth=0.5)
    ax.set_xlim(-0.05, 6.0)
    ax.invert_yaxis()
    ax.legend(
        frameon=False,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.subplots_adjust(left=0.13, right=0.99, bottom=0.18, top=0.79)
    save_figure(fig, "F6_bounds_ladder")


def interval_by_group(frame: pd.DataFrame, group: list[str]) -> pd.DataFrame:
    rows = []
    for keys, values in frame.groupby(group):
        keys = keys if isinstance(keys, tuple) else (keys,)
        mean, lower, upper, _ = paired_summary(values.survival_total)
        rows.append({**dict(zip(group, keys)), "mean": mean, "lower": lower, "upper": upper})
    return pd.DataFrame(rows)


def build_f7() -> None:
    """Survival against communication availability, one column per day.

    Top row: absolute survival with descriptive 95 percent intervals. Bottom row: paired
    contrasts against belief-aware dispatch on one shared scale, which is the correct
    basis for comparing methods evaluated on common seeds.
    """
    evaluation = load_evaluation()
    summary = interval_by_group(
        evaluation, ["communication_reliability", "day", "method"]
    )
    days = sorted(summary.day.unique())
    methods = ["belief_aware", "best_positive_ration", "perfect_information"]
    labels = ["Belief-aware", "Calibrated ration", "Perfect information"]

    paired = evaluation.pivot_table(
        index=["communication_reliability", "day", "seed"],
        columns="method", values="survival_total",
    ).reset_index()
    contrasts = []
    for (q, day), frame in paired.groupby(["communication_reliability", "day"]):
        for method in methods[1:]:
            diff = (frame[method] - frame["belief_aware"]).to_numpy()
            mean, lower, upper, _ = paired_summary(pd.Series(diff))
            contrasts.append({"q": q, "day": day, "method": method,
                              "mean": mean, "lower": lower, "upper": upper})
    contrasts = pd.DataFrame(contrasts)

    fig, axes = plt.subplots(2, len(days), figsize=(7.16, 4.3),
                             sharex=True, sharey="row")
    for column, day in enumerate(days):
        top = axes[0, column]
        # Belief-aware is drawn last, dashed, because the ration tracks it closely on most
        # days and would otherwise hide it entirely.
        for method in ["best_positive_ration", "perfect_information", "belief_aware"]:
            frame = summary[(summary.day == day) & (summary.method == method)].sort_values(
                "communication_reliability"
            )
            x = frame.communication_reliability.to_numpy()
            dashed = method == "belief_aware"
            top.plot(x, frame["mean"], color=COLORS[method], marker=MARKERS[method],
                     markersize=3.2, linewidth=1.1,
                     linestyle="--" if dashed else "-",
                     markerfacecolor="none" if dashed else COLORS[method],
                     zorder=4 if dashed else 3)
            top.fill_between(x, frame["lower"], frame["upper"],
                             color=COLORS[method], alpha=0.16, linewidth=0)
        top.set_title(f"Day {day}", fontsize=8)
        top.grid(color="0.90", linewidth=0.45)
        top.text(0.03, 0.96, f"({chr(97 + column)})", transform=top.transAxes,
                 ha="left", va="top", fontsize=7)

        bottom = axes[1, column]
        for method in methods[1:]:
            frame = contrasts[(contrasts.day == day) & (contrasts.method == method)].sort_values("q")
            x = frame["q"].to_numpy()
            bottom.plot(x, frame["mean"], color=COLORS[method], marker=MARKERS[method],
                        markersize=3.2, linewidth=1.0)
            bottom.fill_between(x, frame["lower"], frame["upper"],
                                color=COLORS[method], alpha=0.16, linewidth=0)
        bottom.axhline(0, color="0.25", linewidth=0.7, linestyle="--")
        bottom.grid(color="0.90", linewidth=0.45)
        bottom.set_xticks([0.0, 0.5, 1.0])
        bottom.text(0.03, 0.96, f"({chr(97 + len(days) + column)})", transform=bottom.transAxes,
                    ha="left", va="top", fontsize=7)
        if day == 12:
            headline = contrasts[
                (contrasts.day == 12)
                & (contrasts.method == "perfect_information")
                & np.isclose(contrasts["q"], 0.0)
            ]["mean"].iloc[0]
            bottom.annotate(
                f"{headline:+.2f}", xy=(0.0, headline), xytext=(0.30, headline - 1.6),
                fontsize=6.8, color=COLORS["perfect_information"],
                arrowprops={"arrowstyle": "-", "linewidth": 0.6,
                            "color": COLORS["perfect_information"]},
            )
        if day == 10:
            window = contrasts[
                (contrasts.day.isin([9, 10, 11]))
                & (contrasts.method == "perfect_information")
                & np.isclose(contrasts["q"], 0.0)
            ]["mean"].sum()
            bottom.text(
                0.5, 0.06, f"days 9 to 11 sum: {window:+.2f}",
                transform=bottom.transAxes, ha="center", va="bottom", fontsize=6.5,
                color=COLORS["perfect_information"],
            )

    axes[0, 0].set_ylabel("Survival")
    axes[1, 0].set_ylabel("Paired contrast\nagainst belief-aware")
    handles = [
        mpl.lines.Line2D([], [], color=COLORS[m], marker=MARKERS[m], markersize=3.2,
                         linewidth=1.1, label=l,
                         linestyle="--" if m == "belief_aware" else "-",
                         markerfacecolor="none" if m == "belief_aware" else COLORS[m])
        for m, l in zip(methods, labels)
    ]
    fig.legend(handles=handles, frameon=False, ncol=3, loc="lower center",
               bbox_to_anchor=(0.55, 0.005), fontsize=7.5)
    fig.supxlabel("Communication availability $q$", y=0.085, fontsize=9)
    fig.subplots_adjust(left=0.105, right=0.995, bottom=0.175, top=0.93,
                        wspace=0.30, hspace=0.20)
    save_figure(fig, "F7_survival_series")


def build_f9(t5: pd.DataFrame) -> None:
    """Plot evaluation entropy against paired policy difference."""
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    scratch = t5.iloc[0]
    warm = t5.iloc[1:]
    warm_x = 100.0 * float(warm.normalized_entropy.mean())
    warm_y = float(warm.paired_difference.mean())
    ax.axvspan(12.0, 79.0, color="0.92", zorder=0)
    ax.text(
        45.5,
        -1.7,
        "Entropy region not reached\nby any trained configuration",
        ha="center",
        va="center",
        fontsize=7,
        color="0.25",
    )
    ax.scatter(
        100.0 * scratch.normalized_entropy,
        scratch.paired_difference,
        color=COLORS["from_scratch"],
        marker=MARKERS["from_scratch"],
        edgecolor="black",
        linewidth=0.35,
        s=30,
        zorder=3,
    )
    ax.scatter(
        warm_x,
        warm_y,
        color=COLORS["clone"],
        marker=MARKERS["clone"],
        edgecolor="black",
        linewidth=0.35,
        s=34,
        zorder=3,
    )
    ax.annotate("From scratch", (100 * scratch.normalized_entropy, scratch.paired_difference), xytext=(-5, 6), textcoords="offset points", ha="right", fontsize=7)
    ax.annotate(
        "Clone, scalar, counterfactual,\nand softened variants",
        (warm_x, warm_y),
        xytext=(8, -16),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=7,
    )
    ax.axhline(0, color="0.35", linewidth=0.7, linestyle="--")
    ax.set_xlabel("Normalised evaluation entropy (%)")
    ax.set_ylabel("Paired survival difference")
    ax.grid(color="0.88", linewidth=0.5)
    fig.subplots_adjust(left=0.2, right=0.98, bottom=0.19, top=0.95)
    save_figure(fig, "F9_entropy_band")


def build_f10(t6: pd.DataFrame) -> None:
    """Plot ration effects by nominal connectivity and realized isolation."""
    levels = t6[
        t6.communication.notna() & (t6.posture == "permissive_memoryless")
    ].sort_values("communication")
    partition = pd.read_csv(RESULTS / "stage7_partition_audit_summary.csv")
    partition_map = dict(
        zip(partition.communication_reliability, partition.partition_fraction)
    )
    levels = levels.assign(
        partition_fraction=levels.communication.map(partition_map)
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.1), sharey=True)
    specifications = [
        ("Day 12", "day12_difference", "day12_ci_lower", "day12_ci_upper", "belief_aware", "o", "-"),
        ("Days 9 to 11", "days9_11_difference", "days9_11_ci_lower", "days9_11_ci_upper", "best_positive_ration", "s", "--"),
    ]
    for ax, x_col in zip(axes, ["communication", "partition_fraction"]):
        plotted = levels.sort_values(x_col)
        for label, mean_col, low_col, high_col, key, marker, linestyle in specifications:
            means = plotted[mean_col].to_numpy()
            lower = plotted[low_col].to_numpy()
            upper = plotted[high_col].to_numpy()
            ax.errorbar(
                plotted[x_col],
                means,
                yerr=np.vstack([means - lower, upper - means]),
                color=COLORS[key],
                marker=marker,
                linestyle=linestyle,
                capsize=2.5,
                label=label,
            )
    confirmation = t6[t6.row == "q=0 confirmatory, fresh seeds 30-89"].iloc[0]
    for ax, x in zip(axes, [0.025, 0.975]):
        ax.errorbar(
            x,
            confirmation.days9_11_difference,
            yerr=[
                [confirmation.days9_11_difference - confirmation.days9_11_ci_lower],
                [confirmation.days9_11_ci_upper - confirmation.days9_11_difference],
            ],
            color=COLORS["best_positive_ration"],
            marker="D",
            markerfacecolor="white",
            linestyle="none",
            capsize=2.5,
            zorder=4,
        )
    axes[0].annotate(
        "Fresh-seed confirmation",
        (0.025, confirmation.days9_11_difference),
        xytext=(8, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=7,
    )
    for ax in axes:
        ax.axhline(0, color="0.3", linewidth=0.7)
        ax.axhline(0.25, color="0.4", linewidth=0.8, linestyle=":", label="Break-even benchmark")
        ax.set_xlim(-0.04, 1.04)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.grid(axis="y", color="0.88", linewidth=0.5)
    axes[0].set_xlabel("Communication reliability, q")
    axes[1].set_xlabel("Realized partition fraction")
    axes[0].set_ylabel("Paired survival difference")
    axes[0].text(0.01, 0.98, "(a)", transform=axes[0].transAxes, ha="left", va="top")
    axes[1].text(0.01, 0.98, "(b)", transform=axes[1].transAxes, ha="left", va="top")
    axes[1].legend(frameon=False, loc="upper left")
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.19, top=0.96, wspace=0.12)
    save_figure(fig, "F10_sign_reversal")


def write_captions() -> None:
    captions = """# Stage 10 load-bearing captions

F6. Day-12 greedy information contrasts by communication reliability; the reference exact-oracle gap is 21.011.

F7. Mean survival and 95% intervals by method, day, and communication reliability on the full synthetic network.

F9. Evaluation entropy against paired survival difference, highlighting the entropy region no trained configuration reached.

F10. Paired ration effects against nominal reliability and realized partition frequency; day-12 intervals are narrower than their markers.
"""
    (FIGURES / "STAGE10_CAPTIONS.md").write_text(captions, encoding="utf-8")


def main() -> None:
    """Build load-bearing tables and figures."""
    configure_style()
    TABLES.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    t4 = build_t4()
    t5 = build_t5()
    t6 = build_t6()
    build_f6(t4)
    build_f7()
    build_f9(t5)
    build_f10(t6)
    write_captions()


if __name__ == "__main__":
    main()
