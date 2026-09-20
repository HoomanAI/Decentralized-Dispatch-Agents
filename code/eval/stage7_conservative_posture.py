"""Four-cell Stage 7 robustness check under the active-memory risk posture."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pandas as pd
import yaml

from code.eval.stage7 import _run_parallel


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    results = root / "results"
    config = yaml.safe_load(
        (root / "code" / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config = deepcopy(config)
    config["environment"].update(
        {
            "scenario_alpha": 0.75,
            "scenario_gamma": 0.75,
            "phi_min": 0.70,
            "scenario_alpha_concentration": None,
        }
    )
    affected = pd.read_csv(root / "data" / "daily_affected_population_clean.csv")
    partial = results / "stage7_conservative_posture.partial.csv"
    output = results / "stage7_conservative_posture.csv"
    frames = [pd.read_csv(partial)] if partial.exists() else []
    complete = set()
    if frames:
        complete = set(zip(frames[0].communication_reliability, frames[0].method))

    arms = [
        ("belief_aware", (0.0, 0.0, 0.0), 0.0),
        ("positive_ration", (0.0, 0.05, 0.0), 1.0),
    ]
    for reliability in [1.0, 0.0]:
        for method, budgets, information_weight in arms:
            if (reliability, method) in complete:
                continue
            frame = _run_parallel(
                affected,
                config,
                root / "data",
                budgets,
                information_weight,
                list(range(30)),
                [8, 9, 10, 11, 12],
                communication_reliability=reliability,
                reduced_instance=False,
                max_workers=8,
            ).assign(
                method=method,
                posture="conservative_active_memory",
                scenario_alpha=0.75,
                scenario_gamma=0.75,
                phi_min=0.70,
            )
            frames.append(frame)
            pd.concat(frames, ignore_index=True).to_csv(partial, index=False)
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(output, index=False)
    print(result.groupby(["communication_reliability", "method"]).survival_total.mean())


if __name__ == "__main__":
    main()
