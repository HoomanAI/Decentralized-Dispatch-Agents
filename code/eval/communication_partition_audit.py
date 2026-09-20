"""Audit realized communication partitions for every full-instance reliability level."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from code.eval.stage7 import _run


def _audit_seed(args: tuple[Any, ...]) -> list[dict[str, Any]]:
    affected, config, data_dir, seed, reliability = args
    records: list[dict[str, Any]] = []
    _run(
        affected,
        config,
        data_dir,
        (0.0, 0.0, 0.0),
        0.0,
        [seed],
        [8, 9, 10, 11, 12],
        communication_reliability=reliability,
        reduced_instance=False,
        communication_audit_records=records,
    )
    return records


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load(
        (project_root / "code" / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    affected = pd.read_csv(project_root / "data" / "daily_affected_population_clean.csv")
    output = project_root / "results" / "stage7_partition_audit.csv"
    existing = pd.read_csv(output) if output.exists() else pd.DataFrame()
    legacy = project_root / "results" / "stage7_q0_75_partition_audit.csv"
    if legacy.exists() and (
        existing.empty
        or not (existing.communication_reliability == 0.75).any()
    ):
        existing = pd.concat([existing, pd.read_csv(legacy)], ignore_index=True)
    completed = set()
    if not existing.empty:
        completed = set(zip(existing.communication_reliability, existing.seed))
    records = [] if existing.empty else existing.to_dict("records")
    # q=0 is deterministic: every communication graph has 12 singleton
    # components, so its partition fraction is exactly one.  Running full
    # dispatch rollouts cannot change that audit quantity.
    for reliability in [1.0, 0.75, 0.5, 0.25]:
        tasks = [
            (affected, config, project_root / "data", seed, reliability)
            for seed in range(30)
            if (reliability, seed) not in completed
        ]
        if tasks:
            with ProcessPoolExecutor(max_workers=8) as executor:
                nested = list(executor.map(_audit_seed, tasks))
            records.extend(record for seed_records in nested for record in seed_records)
            pd.DataFrame(records).to_csv(output, index=False)
    frame = pd.DataFrame(records).sort_values(
        ["communication_reliability", "seed", "day"], ascending=[False, True, True]
    )
    frame.to_csv(output, index=False)
    summary = frame.groupby("communication_reliability", as_index=False).agg(
        communication_epochs=("communication_epochs", "sum"),
        partitioned_epochs=("partitioned_epochs", "sum"),
        maximum_components=("maximum_component_count", "max"),
    )
    summary["partition_fraction"] = (
        summary.partitioned_epochs / summary.communication_epochs
    )
    summary = pd.concat(
        [
            summary,
            pd.DataFrame(
                [
                    {
                        "communication_reliability": 0.0,
                        "communication_epochs": pd.NA,
                        "partitioned_epochs": pd.NA,
                        "maximum_components": 12,
                        "partition_fraction": 1.0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    ).sort_values("communication_reliability", ascending=False)
    summary.to_csv(
        project_root / "results" / "stage7_partition_audit_summary.csv", index=False
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
