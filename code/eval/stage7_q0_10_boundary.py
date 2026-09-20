"""Evaluate the pre-specified Stage 7 boundary cell at communication q=0.10."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from code.eval.stage7 import _run, _run_parallel


Q = 0.10


def _belief_seed(args: tuple[Any, ...]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    affected, config, data_dir, seed = args
    audit: list[dict[str, Any]] = []
    frame = _run(
        affected,
        config,
        data_dir,
        (0.0, 0.0, 0.0),
        0.0,
        [seed],
        [8, 9, 10, 11, 12],
        communication_reliability=Q,
        reduced_instance=False,
        communication_audit_records=audit,
    ).assign(method="belief_aware")
    return frame, audit


def _calibration_seed(args: tuple[Any, ...]) -> pd.DataFrame:
    affected, config, data_dir, seed, epsilon_2, epsilon_3 = args
    return _run(
        affected,
        config,
        data_dir,
        (0.0, epsilon_2, epsilon_3),
        1.0 if epsilon_2 + epsilon_3 > 0 else 0.0,
        [seed],
        [8, 9, 10, 11],
        communication_reliability=Q,
        reduced_instance=False,
    )


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    results = root / "results"
    config = yaml.safe_load(
        (root / "code" / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    affected = pd.read_csv(root / "data" / "daily_affected_population_clean.csv")

    calibration_path = results / "stage7_q0_10_calibration.csv"
    calibration_partial = results / "stage7_q0_10_calibration.partial.csv"
    frames = [pd.read_csv(calibration_partial)] if calibration_partial.exists() else []
    completed = set()
    if frames:
        completed = set(zip(frames[0].epsilon_2, frames[0].epsilon_3))
    tasks = []
    for epsilon_2 in [0.0, 0.05, 0.1]:
        for epsilon_3 in [0.0, 0.05, 0.1]:
            if (epsilon_2, epsilon_3) in completed:
                continue
            tasks.extend(
                (affected, config, root / "data", seed, epsilon_2, epsilon_3)
                for seed in range(3)
            )
    if tasks:
        with ProcessPoolExecutor(max_workers=8) as executor:
            new_frames = list(executor.map(_calibration_seed, tasks))
        frames.extend(new_frames)
        pd.concat(frames, ignore_index=True).to_csv(calibration_partial, index=False)
    calibration = pd.concat(frames, ignore_index=True)
    calibration.to_csv(calibration_path, index=False)
    means = calibration.groupby(["epsilon_2", "epsilon_3"], as_index=False).agg(
        survival=("survival_total", "mean")
    )
    positive = means[(means.epsilon_2 > 0) | (means.epsilon_3 > 0)].sort_values(
        ["survival", "epsilon_2", "epsilon_3"], ascending=[False, True, True]
    ).iloc[0]
    budgets = (0.0, float(positive.epsilon_2), float(positive.epsilon_3))

    evaluation_path = results / "stage7_q0_10_evaluation.csv"
    evaluation_partial = results / "stage7_q0_10_evaluation.partial.csv"
    eval_frames = [pd.read_csv(evaluation_partial)] if evaluation_partial.exists() else []
    evaluated = set(eval_frames[0].method.unique()) if eval_frames else set()
    if "belief_aware" not in evaluated:
        tasks = [(affected, config, root / "data", seed) for seed in range(30)]
        with ProcessPoolExecutor(max_workers=8) as executor:
            outputs = list(executor.map(_belief_seed, tasks))
        eval_frames.append(pd.concat([item[0] for item in outputs], ignore_index=True))
        audit = pd.DataFrame(
            [record for item in outputs for record in item[1]]
        )
        audit.to_csv(results / "stage7_q0_10_partition_audit.csv", index=False)
        pd.concat(eval_frames, ignore_index=True).to_csv(evaluation_partial, index=False)
    if "best_positive_ration" not in evaluated:
        eval_frames.append(
            _run_parallel(
                affected,
                config,
                root / "data",
                budgets,
                1.0,
                list(range(30)),
                [8, 9, 10, 11, 12],
                communication_reliability=Q,
            ).assign(method="best_positive_ration")
        )
        pd.concat(eval_frames, ignore_index=True).to_csv(evaluation_partial, index=False)
    evaluation = pd.concat(eval_frames, ignore_index=True)
    evaluation.to_csv(evaluation_path, index=False)
    print(f"selected budgets: epsilon_2={budgets[1]:.2f}, epsilon_3={budgets[2]:.2f}")
    print(evaluation.groupby("method").survival_total.mean().to_string())


if __name__ == "__main__":
    main()
