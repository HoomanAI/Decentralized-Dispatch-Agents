"""Build Figures F16 and F17 for the resilience ablation (RESILIENCE_MINOR).

F16_resilience_curve: five-day survival vs communication reliability for three
information architectures, plus a normalised retention panel.

F17_resilience_mechanism: (a) observations incorporated by end of horizon vs q,
(b) per-day survival at q = 0 for the three architectures (identical by
construction, confirming the mechanism rather than the dispatch rule).

No simulation is performed here. Values come from:
  results/minor_resilience_ablation.csv  (seed-level, per-day)
  results/minor_resilience_ablation_summary.csv  (paired summary)
"""

from __future__ import annotations
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

COLORS = {
    "centralized_stale_global": "#D55E00",
    "decentralized_merge":       "#0072B2",
    "decentralized_no_merge":    "#CC79A7",
}
MARKERS = {
    "centralized_stale_global": "s",
    "decentralized_merge":       "o",
    "decentralized_no_merge":    "^",
}
LABELS = {
    "centralized_stale_global": "Centralised (stale global view)",
    "decentralized_merge":       "Decentralised with merge (proposed)",
    "decentralized_no_merge":    "Decentralised, no merge",
}

def configure_style() -> None:
    mpl.rcParams.update({
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
    })

def save_figure(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  saved {name}.pdf/.png")

def five_day_stats(df: pd.DataFrame) -> pd.DataFrame:
    totals = (
        df.groupby(["architecture", "communication_reliability", "seed"])
        .survival_total.sum()
        .reset_index()
    )
    rows = []
    for (arch, q), grp in totals.groupby(["architecture", "communication_reliability"]):
        vals = grp.survival_total.to_numpy()
        n = len(vals)
        mean = float(vals.mean())
        se = float(stats.sem(vals))
        lo, hi = stats.t.interval(0.95, n - 1, loc=mean, scale=se)
        rows.append({"architecture": arch, "q": float(q), "mean": mean,
                     "lo": float(lo), "hi": float(hi)})
    return pd.DataFrame(rows)

def build_f16(df: pd.DataFrame) -> None:
    fiveday = five_day_stats(df)
    q_vals = sorted(fiveday.q.unique())
    archs = ["centralized_stale_global", "decentralized_merge", "decentralized_no_merge"]

    ref = {}
    for arch in archs:
        row = fiveday[(fiveday.architecture == arch) & np.isclose(fiveday.q, 1.0)]
        ref[arch] = float(row["mean"].iloc[0])

    fig, (ax_abs, ax_norm) = plt.subplots(1, 2, figsize=(7.16, 3.2),
                                            gridspec_kw={"wspace": 0.42})
    for arch in archs:
        sub = fiveday[fiveday.architecture == arch].sort_values("q")
        qs    = sub.q.to_numpy()
        means = sub["mean"].to_numpy()
        lo    = sub["lo"].to_numpy()
        hi    = sub["hi"].to_numpy()
        c, m, lbl = COLORS[arch], MARKERS[arch], LABELS[arch]

        ax_abs.plot(qs, means, color=c, marker=m, label=lbl)
        ax_abs.fill_between(qs, lo, hi, color=c, alpha=0.13)

        ax_norm.plot(qs, means / ref[arch], color=c, marker=m, label=lbl)
        ax_norm.fill_between(qs, lo / ref[arch], hi / ref[arch], color=c, alpha=0.13)

    for ax in (ax_abs, ax_norm):
        ax.set_xlabel("Communication reliability")
        ax.set_xticks(q_vals)
        ax.grid(axis="y", color="0.88", linewidth=0.5)

    ax_abs.set_ylabel("Five-day survival")
    ax_norm.set_ylabel("Retention relative to own full-pooling baseline")
    ax_norm.yaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda x, _: f"{100*x:.0f}%")
    )
    ax_abs.legend(frameon=False, fontsize=6.5, loc="upper left")
    for idx, ax in enumerate((ax_abs, ax_norm)):
        ax.text(-0.13, 1.03, f"({chr(97 + idx)})", transform=ax.transAxes, fontsize=7)

    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.17, top=0.96)
    save_figure(fig, "F16_resilience_curve")

def build_f17(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    archs = ["centralized_stale_global", "decentralized_merge", "decentralized_no_merge"]
    obs_rows = (
        summary[summary.endpoint == "five_day"]
        [["communication_reliability", "architecture", "end_horizon_observations_incorporated"]]
        .rename(columns={"communication_reliability": "q"})
    )
    q_vals = sorted(obs_rows.q.unique())

    q0 = df[np.isclose(df.communication_reliability, 0.0)]
    day_means = (
        q0.groupby(["architecture", "day"])
        .survival_total.mean()
        .reset_index()
    )

    fig, (ax_obs, ax_day) = plt.subplots(1, 2, figsize=(7.16, 3.2),
                                          gridspec_kw={"wspace": 0.42})
    for arch in archs:
        c, m, lbl = COLORS[arch], MARKERS[arch], LABELS[arch]

        sub = obs_rows[obs_rows.architecture == arch].sort_values("q")
        ax_obs.plot(sub.q, sub.end_horizon_observations_incorporated,
                    color=c, marker=m, label=lbl)

        sub_d = day_means[day_means.architecture == arch].sort_values("day")
        ax_day.plot(sub_d.day, sub_d.survival_total, color=c, marker=m, label=lbl)

    ax_obs.set_xlabel("Communication reliability")
    ax_obs.set_ylabel("Traversal observations incorporated")
    ax_obs.set_xticks(q_vals)
    ax_obs.grid(axis="y", color="0.88", linewidth=0.5)
    ax_obs.yaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k" if x >= 1000 else f"{x:.0f}")
    )
    ax_obs.legend(frameon=False, fontsize=6.5)

    ax_day.set_xlabel("Day")
    ax_day.set_ylabel(r"Mean survival per day at $q = 0$")
    ax_day.set_xticks([8, 9, 10, 11, 12])
    ax_day.grid(axis="y", color="0.88", linewidth=0.5)
    ax_day.text(0.97, 0.97,
                "All three architectures\noverlap at $q = 0$",
                transform=ax_day.transAxes, ha="right", va="top",
                fontsize=6.5, color="0.45")

    for idx, ax in enumerate((ax_obs, ax_day)):
        ax.text(-0.13, 1.03, f"({chr(97 + idx)})", transform=ax.transAxes, fontsize=7)

    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.17, top=0.96)
    save_figure(fig, "F17_resilience_mechanism")

def main() -> None:
    configure_style()
    df = pd.read_csv(RESULTS / "minor_resilience_ablation.csv")
    summary = pd.read_csv(RESULTS / "minor_resilience_ablation_summary.csv")
    build_f16(df)
    build_f17(df, summary)
    print("Done.")

if __name__ == "__main__":
    main()
