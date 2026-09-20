"""Command-line entry point for the resumable full-instance Stage 7 sweep."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from code.eval.stage7 import run_stage7_full_cross


def main() -> None:
    """Resume calibration and then run matched evaluation from saved checkpoints."""
    project_root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load(
        (project_root / "code" / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    affected = pd.read_csv(project_root / "data" / "daily_affected_population_clean.csv")
    paths = run_stage7_full_cross(
        affected,
        config,
        project_root / "data",
        project_root / "results",
    )
    print(*(str(path) for path in paths), sep="\n")


if __name__ == "__main__":
    main()
