"""Build the quality-of-service panel from saved baseline artifacts.

Aggregates results/stage9_baselines.csv and the measured daily exposure fractions. No
simulation is run here.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from code.reporting.stage10_load_bearing import COLORS, configure_style, save_figure

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

AWARE = ["protocol_myopic", "ahmadi", "alns", "ga", "peng"]
EXPOSURE_DAYS = [7, 8, 9, 10, 11, 12]
EXPOSURE_PERCENT = [90.7, 27.8, 0.7, 0.3, 0.3, 20.5]

GROUPS = [
    ("Reliability-aware (five methods)", AWARE, COLORS["belief_aware"], "-", "o"),
    ("Reliability-blind myopic", ["reliability_blind_myopic"], COLORS["best_positive_ration"], "--", "s"),
    ("Yan, static plan", ["yan"], COLORS["reachable_realization"], ":", "^"),
]


def _band(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Mean and 95 percent t interval across seeds, per day."""
    rows = []
    for day, group in frame.groupby("day"):
        values = group[column].to_numpy(dtype=float)
        mean = float(values.mean())
        lower, upper = stats.t.interval(
            0.95, len(values) - 1, loc=mean, scale=stats.sem(values)
        )
        rows.append({"day": day, "mean": mean, "lower": float(lower), "upper": float(upper)})
    return pd.DataFrame(rows).sort_values("day")


def build_f3() -> None:
    data = pd.read_csv(RESULTS / "stage9_baselines.csv")
    data["service_rate"] = data.served / data.patients
    data["survival_per_patient"] = data.survival_total / data.patients

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.7))

    ax = axes[0]
    ax.bar(EXPOSURE_DAYS, EXPOSURE_PERCENT, color=COLORS["reachable_realization"],
           edgecolor="black", linewidth=0.4, width=0.62)
    for day, value in zip(EXPOSURE_DAYS, EXPOSURE_PERCENT):
        ax.text(day, value + 2.5, f"{value:.1f}", ha="center", fontsize=6.5)
    ax.set_ylim(0, 104)
    ax.set_xticks(EXPOSURE_DAYS)
    ax.set_xlabel("Day")
    ax.set_ylabel("Routable arcs intersecting\nthe perimeter (%)")
    ax.set_title("(a) Measured exposure", fontsize=8)
    ax.grid(axis="y", color="0.90", linewidth=0.45)

    for index, (ax, column, title, ylabel) in enumerate((
        (axes[1], "service_rate", "(b) Service rate", "Patients served / arrived"),
        (axes[2], "survival_per_patient", "(c) Survival yield", "Survival per patient"),
    )):
        for label, methods, colour, style, marker in GROUPS:
            band = _band(data[data.method.isin(methods)], column)
            ax.plot(band.day, band["mean"], color=colour, linestyle=style, marker=marker,
                    markersize=3.0, linewidth=1.1, label=label)
            ax.fill_between(band.day, band["lower"], band["upper"], color=colour,
                            alpha=0.18, linewidth=0)
        ax.axvspan(9.5, 11.5, color="0.92", zorder=0)
        ax.set_ylim(0, 1.0)
        ax.set_xticks([8, 9, 10, 11, 12])
        ax.set_xlabel("Day")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=8)
        ax.grid(color="0.90", linewidth=0.45)

    aware_day11 = _band(data[data.method.isin(AWARE)], "service_rate").set_index("day").loc[11, "mean"]
    blind_day11 = _band(data[data.method == "reliability_blind_myopic"], "service_rate").set_index("day").loc[11, "mean"]
    gap = (aware_day11 - blind_day11) * 100.0
    axes[1].annotate(
        "", xy=(11, aware_day11), xytext=(11, blind_day11),
        arrowprops={"arrowstyle": "<->", "linewidth": 0.8, "color": "0.25"},
    )
    axes[1].text(10.85, (aware_day11 + blind_day11) / 2,
                 f"{gap:.1f} percentage\npoints", fontsize=6.5, va="center", ha="right")

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="lower center",
               bbox_to_anchor=(0.55, 0.0), fontsize=7)
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.30, top=0.90, wspace=0.42)
    save_figure(fig, "F3_resilience_quality_of_service")


if __name__ == "__main__":
    configure_style()
    build_f3()
