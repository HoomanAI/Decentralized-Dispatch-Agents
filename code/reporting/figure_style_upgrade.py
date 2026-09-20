"""Build the Stage 10 figure-style upgrades from existing simulation artifacts.

No simulation or training is performed here. Every plotted value is aggregated from a
saved CSV. F14 is deliberately named ``F14_posture_comparison`` because Stage 9 also
reserves the F14 number for its baseline figure.
"""

from __future__ import annotations

from pathlib import Path
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

from code.reporting.stage10_load_bearing import COLORS, configure_style, save_figure


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"
DATA = ROOT / "data"


def calibration_effects() -> pd.DataFrame:
    """Return calibration effects relative to the matched zero-ration cell."""
    frames = [pd.read_csv(RESULTS / "stage7_full_communication_budget_sweep.csv")]
    boundary = RESULTS / "stage7_q0_10_calibration.csv"
    if boundary.exists():
        frames.append(pd.read_csv(boundary))
    data = pd.concat(frames, ignore_index=True)
    totals = (
        data.groupby(
            ["communication_reliability", "epsilon_2", "epsilon_3", "seed"],
            as_index=False,
        ).survival_total.sum()
    )
    baseline = (
        totals[(totals.epsilon_2 == 0) & (totals.epsilon_3 == 0)]
        .set_index(["communication_reliability", "seed"])
        .survival_total
    )
    totals["effect"] = [
        row.survival_total
        - baseline.loc[(row.communication_reliability, row.seed)]
        for row in totals.itertuples()
    ]
    return (
        totals.groupby(
            ["communication_reliability", "epsilon_2", "epsilon_3"],
            as_index=False,
        ).effect.mean()
    )


def evaluation_effects() -> pd.DataFrame:
    """Return seed-level paired effects for every evaluated connectivity level.

    The q=0.10 boundary cell lives in its own artifact and is concatenated here.
    """
    frames = [pd.read_csv(RESULTS / "stage7_full_communication_evaluation.csv")]
    boundary = RESULTS / "stage7_q0_10_evaluation.csv"
    if boundary.exists():
        frames.append(pd.read_csv(boundary))
    data = pd.concat(frames, ignore_index=True)
    pivot = data.pivot_table(
        index=["communication_reliability", "seed", "day"],
        columns="method",
        values="survival_total",
    ).reset_index()
    pivot["effect"] = pivot.best_positive_ration - pivot.belief_aware
    return pivot


def annotated_heatmap(
    ax: plt.Axes,
    values: np.ndarray,
    xlabels: list[str],
    ylabels: list[str],
    *,
    label_format: str = "+.2f",
    stars: np.ndarray | None = None,
) -> mpl.image.AxesImage:
    """Draw a zero-centred red-harm, green-benefit annotated heatmap."""
    finite = np.abs(values[np.isfinite(values)])
    limit = max(float(finite.max(initial=0.0)), 1e-9)
    image = ax.imshow(
        values,
        cmap="RdYlGn",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        aspect="auto",
    )
    ax.set_xticks(np.arange(len(xlabels)), xlabels)
    ax.set_yticks(np.arange(len(ylabels)), ylabels)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            text = "NA" if not np.isfinite(value) else format(value, label_format)
            if stars is not None and stars[row, column] and np.isfinite(value):
                text += "*"
            colour = "white" if np.isfinite(value) and abs(value) > 0.55 * limit else "black"
            ax.text(column, row, text, ha="center", va="center", fontsize=7, color=colour)
    return image


def build_f10() -> None:
    """Create calibration heatmap and held-out connectivity effect panel."""
    calibration = calibration_effects()
    levels = sorted(calibration.communication_reliability.unique(), reverse=True)
    budgets = sorted(
        {(float(row.epsilon_2), float(row.epsilon_3)) for row in calibration.itertuples()}
    )
    matrix = np.full((len(levels), len(budgets)), np.nan)
    for row, q in enumerate(levels):
        indexed = calibration[calibration.communication_reliability == q].set_index(
            ["epsilon_2", "epsilon_3"]
        )
        for column, budget in enumerate(budgets):
            matrix[row, column] = indexed.loc[budget, "effect"]

    effects = evaluation_effects()
    summaries = []
    for q, frame in effects.groupby("communication_reliability"):
        by_seed_12 = frame[frame.day == 12].set_index("seed").effect
        by_seed_9 = frame[frame.day.isin([9, 10, 11])].groupby("seed").effect.sum()
        summaries.append((q, by_seed_12.mean(), by_seed_9.mean()))
    summaries = np.asarray(sorted(summaries), dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.5), gridspec_kw={"width_ratios": [1.45, 1]})
    image = annotated_heatmap(
        axes[0],
        matrix,
        [f"{e2:.2f}, {e3:.2f}" for e2, e3 in budgets],
        [f"{q:.2f}" for q in levels],
    )
    axes[0].set_xlabel(r"Calibration budget $(\epsilon_2,\epsilon_3)$")
    axes[0].set_ylabel("Communication availability $q$")
    axes[0].tick_params(axis="x", rotation=55)
    cbar = fig.colorbar(image, ax=axes[0], fraction=0.045, pad=0.03)
    cbar.set_label("Days 8 to 11 paired effect")

    axes[1].plot(summaries[:, 0], summaries[:, 2], color=COLORS["best_positive_ration"], marker="s", label="Days 9 to 11")
    axes[1].plot(summaries[:, 0], summaries[:, 1], color=COLORS["belief_aware"], marker="o", linestyle="--", label="Day 12")
    axes[1].axhline(0, color="0.25", linewidth=0.7)
    axes[1].axhline(0.25, color="0.35", linewidth=0.8, linestyle=":", label="Break-even benchmark")
    axes[1].set_xlabel("Communication availability $q$")
    axes[1].set_ylabel("Held-out effect", labelpad=2)
    axes[1].set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axes[1].grid(axis="y", color="0.88", linewidth=0.5)
    axes[1].legend(frameon=False, fontsize=6.5)
    for index, ax in enumerate(axes):
        ax.text(-0.14, 1.03, f"({chr(97 + index)})", transform=ax.transAxes, fontsize=7)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.30, top=0.96, wspace=0.55)
    save_figure(fig, "F10_sign_reversal")


def build_f12() -> None:
    """Plot the calibration phase diagram at epsilon_3 equal to zero."""
    calibration = calibration_effects()
    calibration = calibration[calibration.epsilon_3 == 0].copy()
    levels = sorted(calibration.communication_reliability.unique())
    budgets = sorted(calibration.epsilon_2.unique())
    values = calibration.pivot(index="epsilon_2", columns="communication_reliability", values="effect").loc[budgets, levels].to_numpy()
    x, y = np.meshgrid(levels, budgets)
    limit = max(float(np.abs(values).max()), 1e-9)
    fig, ax = plt.subplots(figsize=(3.5, 3.35))
    field = ax.contourf(x, y, values, levels=np.linspace(-limit, limit, 17), cmap="RdYlGn", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), extend="both")
    ax.contour(x, y, values, levels=[0], colors="black", linewidths=1.0)
    if values.min() <= 0.25 <= values.max():
        ax.contour(x, y, values, levels=[0.25], colors="0.2", linestyles="--", linewidths=1.0)
    ax.scatter(x, y, s=12, facecolor="white", edgecolor="black", linewidth=0.4, zorder=3)
    ax.text(0.69, 0.086, "EXPLORATION HARMS", ha="center", va="center", fontsize=7, weight="bold")
    ax.text(0.075, 0.052, "PAYS", ha="center", va="center", fontsize=7, weight="bold", rotation=90)
    ax.text(0.53, 0.018, "NEUTRAL", ha="center", va="center", fontsize=7, weight="bold")
    ax.plot([], [], color="black", linewidth=1.0, label="Zero effect")
    ax.plot([], [], color="0.2", linewidth=1.0, linestyle="--", label="Break-even effect")
    ax.legend(frameon=False, fontsize=6.5, loc="upper right")
    ax.set_xlabel("Communication reliability")
    ax.set_ylabel(r"Class-2 exploration budget, $\epsilon_2$")
    ax.set_xticks(levels)
    ax.tick_params(axis="x", rotation=35)
    ax.set_yticks(budgets)
    cbar = fig.colorbar(field, ax=ax, pad=0.03)
    cbar.set_label("Days 8 to 11 paired effect")
    fig.subplots_adjust(left=0.18, right=0.94, bottom=0.17, top=0.96)
    save_figure(fig, "F12_exploration_phase")


def build_f13() -> None:
    """Show seed-level paired effects at the five prespecified connectivity levels."""
    data = evaluation_effects()
    levels = sorted(data.communication_reliability.unique(), reverse=True)
    fig, axes = plt.subplots(2, len(levels), figsize=(7.16, 4.0), sharey="row")
    for column, q in enumerate(levels):
        frame = data[data.communication_reliability == q]
        endpoints = [
            frame[frame.day == 12].set_index("seed").effect,
            frame[frame.day.isin([9, 10, 11])].groupby("seed").effect.sum(),
        ]
        for row, values in enumerate(endpoints):
            ax = axes[row, column]
            ax.boxplot(
                values.to_numpy(), widths=0.52, patch_artist=True, showfliers=True,
                boxprops={"facecolor": COLORS["best_positive_ration"], "alpha": 0.65, "linewidth": 0.7},
                medianprops={"color": "black", "linewidth": 1.0},
                whiskerprops={"linewidth": 0.7}, capprops={"linewidth": 0.7},
                flierprops={"marker": ".", "markersize": 2.5, "markerfacecolor": "0.25", "markeredgecolor": "0.25"},
            )
            ax.axhline(0, color="0.25", linewidth=0.7, linestyle="--")
            ax.set_xticks([])
            ax.grid(axis="y", color="0.9", linewidth=0.45)
            if row == 0:
                ax.set_title(f"q={q:.2f}")
            ax.text(0.03, 0.96, f"({chr(97 + row * len(levels) + column)})", transform=ax.transAxes, ha="left", va="top", fontsize=7)
    axes[0, 0].set_ylabel("Day-12 paired effect")
    axes[1, 0].set_ylabel("Days 9 to 11 paired effect")
    fig.supxlabel("Communication availability $q$", y=0.04, fontsize=9)
    fig.subplots_adjust(left=0.09, right=0.995, bottom=0.11, top=0.93, wspace=0.22, hspace=0.22)
    save_figure(fig, "F13_seed_distributions")


def build_f6() -> None:
    """Day-12 greedy information contrasts at the three reported connectivity levels.

    The hindsight reference is annotated rather than drawn as a bar: it is a different
    optimisation quantity and two orders of magnitude larger than the other contrasts.
    """
    table = pd.read_csv(TABLES / "T4_bounds_ladder.csv")
    levels = [1.00, 0.25, 0.00]
    series = [
        ("Perfect information", "perfect_information_contrast_day12", COLORS["perfect_information"]),
        ("Reachable, union", "reachable_union_contrast_day12", COLORS["reachable_union"]),
        ("Reachable, per-realisation mean", "reachable_per_realization_mean_day12", COLORS["reachable_realization"]),
        ("Achieved ration", "achieved_day12", COLORS["best_positive_ration"]),
    ]
    reference = float(table.loc[np.isclose(table.communication, 1.0), "exact_oracle_gap_day12_reference"].iloc[0])

    fig, ax = plt.subplots(figsize=(7.16, 3.0))
    height = 0.24
    y = np.arange(len(series))
    for offset, q in enumerate(levels):
        row = table.loc[np.isclose(table.communication, q)].iloc[0]
        values = [float(row[column]) for _, column, _ in series]
        shift = (offset - 1) * height
        bars = ax.barh(
            y + shift, values, height=height * 0.92,
            color=[colour for _, _, colour in series],
            edgecolor="black", linewidth=0.4,
            alpha=[1.0, 0.72, 0.45][offset],
        )
        for bar, value in zip(bars, values):
            ax.text(
                max(value, 0) + 0.07, bar.get_y() + bar.get_height() / 2,
                f"{value:+.3f}", va="center", fontsize=6.2,
            )
    ax.axvline(0, color="0.25", linewidth=0.7)
    ax.set_yticks(y, [name for name, _, _ in series])
    ax.invert_yaxis()
    ax.set_xlim(-0.35, 6.6)
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.set_xlabel("Day-12 survival contrast against belief-aware dispatch (linear scale)")
    ax.grid(axis="x", color="0.88", linewidth=0.5)

    shades = [
        plt.Rectangle((0, 0), 1, 1, facecolor="0.35", alpha=a, edgecolor="black", linewidth=0.4)
        for a in (1.0, 0.72, 0.45)
    ]
    ax.legend(
        shades, [f"q={q:.2f}" for q in levels],
        frameon=False, fontsize=7, loc="lower right", title="Availability",
        title_fontsize=7,
    )
    ax.annotate(
        f"Hindsight reference (pooled patient-slot relaxation): {reference:+.3f}\n"
        "a different optimisation quantity, not plotted to scale",
        xy=(0.995, 1.06), xycoords="axes fraction", ha="right", va="bottom", fontsize=7,
    )
    fig.subplots_adjust(left=0.275, right=0.995, bottom=0.17, top=0.86)
    save_figure(fig, "F6_bounds_ladder")


def posture_effect(frame: pd.DataFrame, positive_name: str) -> tuple[float, float]:
    pivot = frame.pivot_table(index=["seed", "day"], columns="method", values="survival_total").reset_index()
    effect = pivot[positive_name] - pivot.belief_aware
    days = effect[pivot.day.isin([9, 10, 11])].groupby(pivot.loc[pivot.day.isin([9, 10, 11]), "seed"]).sum()
    from scipy import stats
    return float(days.mean()), float(stats.ttest_1samp(days, 0).pvalue)


def build_f14_posture() -> None:
    """Build the two-posture heatmap without claiming unavailable intermediate cells."""
    permissive = pd.read_csv(RESULTS / "stage7_full_communication_evaluation.csv")
    conservative = pd.read_csv(RESULTS / "stage7_conservative_posture.csv")
    replication = pd.read_csv(RESULTS / "stage7_conservative_q0_replication.csv")
    values = np.full((2, 2), np.nan)
    stars = np.zeros((2, 2), dtype=bool)
    for column, q in enumerate([1.0, 0.0]):
        value, p = posture_effect(permissive[permissive.communication_reliability == q], "best_positive_ration")
        values[0, column], stars[0, column] = value, p < 0.05
        level = conservative[conservative.communication_reliability == q]
        if q == 0.0:
            level = pd.concat([level, replication], ignore_index=True)
        value, p = posture_effect(level, "positive_ration")
        values[1, column], stars[1, column] = value, p < 0.05
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    image = annotated_heatmap(ax, values, ["1.00", "0.00"], ["Permissive", "Conservative"], stars=stars)
    ax.set_xlabel("Communication reliability")
    ax.set_ylabel("Reliability posture")
    cbar = fig.colorbar(image, ax=ax, pad=0.04)
    cbar.set_label("Days 9 to 11 paired effect")
    ax.text(1.0, 1.03, "* paired p < 0.05", transform=ax.transAxes, fontsize=7, ha="right")
    fig.subplots_adjust(left=0.31, right=0.88, bottom=0.25, top=0.88)
    save_figure(fig, "F14_posture_comparison")


def build_f1() -> None:
    """Plot the generated planar network and six daily synthetic fire perimeters."""
    roads = json.loads((DATA / "synthetic_road_features.geojson").read_text(encoding="utf-8"))
    fronts = json.loads((DATA / "synthetic_fire_fronts.geojson").read_text(encoding="utf-8"))
    front_by_day = {int(item["properties"]["day"]): item for item in fronts["features"]}
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.8), sharex=True, sharey=True)
    for panel, day in enumerate(range(7, 13)):
        ax = axes.flat[panel]
        for feature in roads["features"]:
            coordinates = np.asarray(feature["geometry"]["coordinates"], dtype=float)
            exposed = float(feature["properties"].get(str(day), 0.0)) > 0
            ax.plot(
                coordinates[:, 0],
                coordinates[:, 1],
                color="#D55E00" if exposed else "0.76",
                linewidth=0.65 if exposed else 0.25,
                alpha=0.95 if exposed else 0.65,
                zorder=1,
            )
        polygon = np.asarray(front_by_day[day]["geometry"]["coordinates"][0], dtype=float)
        ax.fill(polygon[:, 0], polygon[:, 1], color="#E69F00", alpha=0.12, zorder=0)
        ax.plot(polygon[:, 0], polygon[:, 1], color="black", linewidth=0.85, zorder=2)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"Day {day}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(-0.03, 1.02, f"({chr(97 + panel)})", transform=ax.transAxes, fontsize=7)
    handles = [
        mpl.lines.Line2D([], [], color="0.65", linewidth=1.0, label="Generated network"),
        mpl.lines.Line2D([], [], color="#D55E00", linewidth=1.5, label="Exposed arc"),
        mpl.lines.Line2D([], [], color="black", linewidth=1.0, label="Synthetic perimeter"),
    ]
    fig.legend(handles=handles, ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.01))
    fig.subplots_adjust(left=0.015, right=0.995, bottom=0.09, top=0.95, wspace=0.04, hspace=0.12)
    save_figure(fig, "F1_synthetic_exposure_sequence")


def write_notes() -> None:
    text = """# Figure style upgrade notes

F6. Day-12 contrasts at full connectivity from the exact oracle through the achieved route ration.

F10. Calibration and held-out exploration effects vary sharply across communication reliability and ration budgets.

F12. Calibration phase diagram separates harmful, neutral, and beneficial class-2 exploration operating regions.

F13. Seed-level paired survival effects show endpoint distributions across five communication reliability levels.

F14 posture comparison. Reliability posture changes the magnitude but not the exploration-effect sign pattern.

F8 confidence bands were not added. The saved training artifacts contain one aggregate trace per run and no seed-indexed training traces, so a seed-level band cannot be computed without new runs.

F14 naming note. Stage 9 also reserves F14 for its baseline figure. The posture artifact is therefore named F14_posture_comparison pending final manuscript numbering.

F1. Generated planar road network and synthetic fire perimeters across the six-day exposure sequence.
"""
    (FIGURES / "FIGURE_STYLE_UPGRADE_NOTES.md").write_text(text, encoding="utf-8")


def main() -> None:
    configure_style()
    build_f12()
    build_f13()
    build_f10()
    build_f6()
    build_f14_posture()
    build_f1()
    write_notes()


if __name__ == "__main__":
    main()
