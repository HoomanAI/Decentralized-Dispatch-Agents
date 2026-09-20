"""Reconstruct seeded communication traces and summarize disconnection run lengths."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from code.env.vehicle_centric import _components


def _close_run(
    rows: list[dict[str, object]],
    reliability: float,
    seed: int,
    vehicle: int,
    kind: str,
    length: int,
    censored: bool,
) -> None:
    if length:
        rows.append(
            {
                "synthetic_input": True,
                "communication_reliability": reliability,
                "seed": seed,
                "vehicle": vehicle,
                "isolation_definition": kind,
                "run_length_epochs": length,
                "right_censored": censored,
            }
        )


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load(
        (root / "code" / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    audit = pd.read_csv(root / "results" / "stage7_partition_audit.csv")
    fleet_count = 3 * int(config["environment"]["fleet_per_class"])
    seed_base = int(config["environment"]["seed"])
    rows: list[dict[str, object]] = []

    boundary = root / "results" / "stage7_q0_10_partition_audit.csv"
    if boundary.exists():
        audit = pd.concat([audit, pd.read_csv(boundary)], ignore_index=True)
    levels = [1.0, 0.75, 0.5, 0.25]
    if (audit.communication_reliability == 0.10).any():
        levels.append(0.10)
    levels.append(0.0)
    for reliability in levels:
        for seed in range(30):
            active = {
                kind: [0] * fleet_count
                for kind in ["not_connected_to_full_fleet", "singleton"]
            }
            seed_rows = audit[
                (audit.communication_reliability == reliability)
                & (audit.seed == seed)
            ].sort_values("day")
            if reliability == 0.0:
                # With no links, every vehicle is isolated for the full horizon.  The
                # exact count of decision epochs is policy-path dependent, so retain
                # this as a right-censored full-horizon run rather than inventing a
                # numeric duration from another reliability level.
                for vehicle in range(fleet_count):
                    for kind in active:
                        rows.append(
                            {
                                "synthetic_input": True,
                                "communication_reliability": reliability,
                                "seed": seed,
                                "vehicle": vehicle,
                                "isolation_definition": kind,
                                "run_length_epochs": np.nan,
                                "right_censored": True,
                            }
                        )
                continue

            for day_row in seed_rows.itertuples(index=False):
                patient_seed = seed_base + 300000 + seed * 100 + int(day_row.day)
                rng = np.random.default_rng(patient_seed)
                for _ in range(int(day_row.communication_epochs)):
                    components = _components(fleet_count, reliability, rng)
                    component_size = {
                        vehicle: len(component)
                        for component in components
                        for vehicle in component
                    }
                    states = {
                        "not_connected_to_full_fleet": [
                            component_size[v] < fleet_count for v in range(fleet_count)
                        ],
                        "singleton": [component_size[v] == 1 for v in range(fleet_count)],
                    }
                    for kind, vehicle_states in states.items():
                        for vehicle, disconnected in enumerate(vehicle_states):
                            if disconnected:
                                active[kind][vehicle] += 1
                            else:
                                _close_run(
                                    rows,
                                    reliability,
                                    seed,
                                    vehicle,
                                    kind,
                                    active[kind][vehicle],
                                    False,
                                )
                                active[kind][vehicle] = 0
            for kind in active:
                for vehicle in range(fleet_count):
                    _close_run(
                        rows,
                        reliability,
                        seed,
                        vehicle,
                        kind,
                        active[kind][vehicle],
                        True,
                    )

    detail = pd.DataFrame(rows)
    detail.to_csv(root / "results" / "stage7_isolation_duration_runs.csv", index=False)
    summary_rows: list[dict[str, object]] = []
    for reliability in levels:
        for kind in ["not_connected_to_full_fleet", "singleton"]:
            values = detail[
                (detail.communication_reliability == reliability)
                & (detail.isolation_definition == kind)
            ].run_length_epochs.dropna().to_numpy(dtype=float)
            all_horizon_censored = reliability == 0.0
            summary_rows.append(
                {
                    "synthetic_input": True,
                    "communication_reliability": reliability,
                    "isolation_definition": kind,
                    "run_count": len(values),
                    "mean_run_epochs": float(values.mean()) if len(values) else 0.0,
                    "median_run_epochs": float(np.median(values)) if len(values) else 0.0,
                    "p90_run_epochs": float(np.quantile(values, 0.90)) if len(values) else 0.0,
                    "p95_run_epochs": float(np.quantile(values, 0.95)) if len(values) else 0.0,
                    "maximum_run_epochs": int(values.max()) if len(values) else 0,
                    "share_one_epoch": float(np.mean(values == 1)) if len(values) else 0.0,
                    "all_horizon_censored": all_horizon_censored,
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(root / "results" / "stage7_isolation_duration_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
