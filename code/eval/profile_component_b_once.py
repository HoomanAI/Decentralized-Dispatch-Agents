"""Profile one resumed Component B rollout iteration."""

from pathlib import Path

import pandas as pd
import yaml

from code.eval.stage11_component_b import train_component_b


root = Path(__file__).resolve().parents[2]
config = yaml.safe_load((root / "code/config/default.yaml").read_text(encoding="utf-8"))
affected = pd.read_csv(root / "data/daily_affected_population_clean.csv")
train_component_b(
    affected,
    config,
    root / "data",
    root / "results",
    updates=52,
    minimum_environment_steps=0,
)
