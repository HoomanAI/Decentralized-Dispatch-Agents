"""Profile one full-instance five-day Stage 7 rollout."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from code.eval.stage7 import _run


def main() -> None:
    """Run one belief-aware seed over days 8 through 12."""
    project_root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load(
        (project_root / "code" / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    affected = pd.read_csv(project_root / "data" / "daily_affected_population_clean.csv")
    result = _run(
        affected,
        config,
        project_root / "data",
        (0.0, 0.0, 0.0),
        0.0,
        [0],
        [8, 9, 10, 11, 12],
        communication_reliability=1.0,
        reduced_instance=False,
    )
    print(result["survival_total"].sum())


if __name__ == "__main__":
    main()
