"""Reachable perfect-information bound for route-level exploration."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from code.eval.stage7 import _run


def run_reachable_headroom_level(
    affected: pd.DataFrame,
    config: dict[str, Any],
    data_dir: Path,
    results_dir: Path,
    communication_reliability: float,
) -> Path:
    """Compute union and single-realization route-observable information ceilings."""
    alternative_records: list[dict[str, Any]] = []
    _run(affected, config, data_dir, (0.0, np.inf, np.inf), 1.0, list(range(3)), [8, 9, 10, 11, 12], communication_reliability=communication_reliability, reduced_instance=False, audit_records=alternative_records)
    alternative_by_realization = {
        int(item["seed"]): set(item["alternative_arcs"])
        for item in alternative_records
    }
    if set(alternative_by_realization) != {0, 1, 2}:
        raise AssertionError(
            "Reachable-headroom audit requires alternative-route sets for seeds 0, 1, and 2"
        )
    alternative_union = set().union(*alternative_by_realization.values())

    aware_records: list[dict[str, Any]] = []
    aware = _run(affected, config, data_dir, (0.0, 0.0, 0.0), 0.0, list(range(30)), [8, 9, 10, 11, 12], communication_reliability=communication_reliability, reduced_instance=False, audit_records=aware_records)
    perfect_records: list[dict[str, Any]] = []
    perfect = _run(affected, config, data_dir, (0.0, 0.0, 0.0), 0.0, list(range(30)), [8, 9, 10, 11, 12], communication_reliability=communication_reliability, perfect_information=True, reduced_instance=False, audit_records=perfect_records)
    perfect_by_seed = {item["seed"]: item for item in perfect_records}
    critical_by_seed = {}
    for item in aware_records:
        traffic = item["traffic_edge_ids"] | perfect_by_seed[item["seed"]]["traffic_edge_ids"]
        critical_by_seed[item["seed"]] = item["mismatch_edge_ids"] & traffic
    critical_union = set().union(*critical_by_seed.values())
    reachable_union = critical_union & alternative_union

    reveal_sets = {"union": (reachable_union, list(range(30)))}
    reveal_sets.update(
        {
            f"realization_{realization}": (
                critical_by_seed[realization] & alternative_arcs,
                [realization],
            )
            for realization, alternative_arcs in alternative_by_realization.items()
        }
    )
    reachable_by_bound = {
        bound_type: (
            evaluation_seeds,
            _run(
            affected,
            config,
            data_dir,
            (0.0, 0.0, 0.0),
            0.0,
            evaluation_seeds,
            [8, 9, 10, 11, 12],
            communication_reliability=communication_reliability,
            reduced_instance=False,
            day12_reveal_edges=reveal_edges,
            ),
        )
        for bound_type, (reveal_edges, evaluation_seeds) in reveal_sets.items()
    }
    rows = []
    for bound_type, (evaluation_seeds, reachable) in reachable_by_bound.items():
        realization = (
            int(bound_type.removeprefix("realization_"))
            if bound_type.startswith("realization_")
            else np.nan
        )
        alternative_arcs = (
            alternative_union
            if bound_type == "union"
            else alternative_by_realization[int(realization)]
        )
        for seed in evaluation_seeds:
            aware_seed = aware.loc[aware.seed == seed]
            perfect_seed = perfect.loc[perfect.seed == seed]
            reachable_seed = reachable.loc[reachable.seed == seed]
            critical = critical_by_seed[seed]
            reachable_critical = critical & alternative_arcs
            rows.append(
                {
                    "synthetic_input": True,
                    "communication_reliability": communication_reliability,
                    "bound_type": bound_type,
                    "alternative_realization": realization,
                    "seed": seed,
                    "critical_arc_count": len(critical),
                    "reachable_critical_arc_count": len(reachable_critical),
                    "reachable_fraction": (
                        len(reachable_critical) / len(critical) if critical else np.nan
                    ),
                    "alternative_arc_count": len(alternative_arcs),
                    "alternative_arc_union_count": len(alternative_union),
                    "aware_day12": float(
                        aware_seed.loc[aware_seed.day == 12, "survival_total"].iloc[0]
                    ),
                    "reachable_day12": float(
                        reachable_seed.loc[
                            reachable_seed.day == 12, "survival_total"
                        ].iloc[0]
                    ),
                    "perfect_day12": float(
                        perfect_seed.loc[
                            perfect_seed.day == 12, "survival_total"
                        ].iloc[0]
                    ),
                    "aware_five_day": float(aware_seed.survival_total.sum()),
                    "reachable_five_day": float(reachable_seed.survival_total.sum()),
                    "perfect_five_day": float(perfect_seed.survival_total.sum()),
                }
            )
    path = results_dir / f"stage7_full_reachable_headroom_q{str(communication_reliability).replace('.', '_')}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def main() -> None:
    """Run one resumable connectivity-level audit from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--communication-reliability", type=float, required=True)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load(
        (project_root / "code" / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    affected = pd.read_csv(project_root / "data" / "daily_affected_population_clean.csv")
    path = run_reachable_headroom_level(
        affected,
        config,
        project_root / "data",
        project_root / "results",
        args.communication_reliability,
    )
    print(path)


if __name__ == "__main__":
    main()
