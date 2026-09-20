"""Command-line entry point for the Stage 7 confirmatory replication."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from code.eval.stage7 import run_stage7_communication_replication


def main() -> None:
    """Run or resume the pre-specified replication from its checkpoint."""
    project_root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load(
        (project_root / "code" / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    affected = pd.read_csv(project_root / "data" / "daily_affected_population_clean.csv")
    path = run_stage7_communication_replication(
        affected,
        config,
        project_root / "data",
        project_root / "results",
    )
    print(path)


if __name__ == "__main__":
    main()
