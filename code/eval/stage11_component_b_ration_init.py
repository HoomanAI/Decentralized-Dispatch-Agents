"""Run the single pre-specified fixed-ration-initialized Component B arm."""

from pathlib import Path

import pandas as pd
import yaml

from code.eval.stage11_component_b import evaluate_component_b, train_component_b


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load(
        (root / "code/config/default.yaml").read_text(encoding="utf-8")
    )
    affected = pd.read_csv(root / "data/daily_affected_population_clean.csv")
    stem = "stage11_component_b_ration_init"
    train_component_b(
        affected,
        config,
        root / "data",
        root / "results",
        artifact_stem=stem,
        fixed_ration_warm_start=True,
    )
    evaluate_component_b(
        affected,
        config,
        root / "data",
        root / "results",
        artifact_stem=stem,
    )


if __name__ == "__main__":
    main()
