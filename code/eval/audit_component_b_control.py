"""Audit Component B against the matched conservative no-exploration control."""

from pathlib import Path

import pandas as pd
from scipy import stats


root = Path(__file__).resolve().parents[2]
component = pd.read_csv(root / "results/stage11_component_b_evaluation.csv")
posture = pd.read_csv(root / "results/stage7_conservative_posture.csv")
posture = posture[posture.communication_reliability.eq(0.0)]
control = posture[posture.method.eq("belief_aware")][
    ["seed", "day", "survival_total"]
].rename(columns={"survival_total": "control_survival"})
fixed = posture[posture.method.eq("positive_ration")][
    ["seed", "day", "survival_total"]
].rename(columns={"survival_total": "stage7_fixed_survival"})
merged = component.merge(control, on=["seed", "day"]).merge(fixed, on=["seed", "day"])
merged["learned_minus_control"] = (
    merged.learned_survival - merged.control_survival
)
merged["current_fixed_minus_stage7_fixed"] = (
    merged.fixed_ration_survival - merged.stage7_fixed_survival
)
merged.to_csv(root / "results/stage11_component_b_control_audit.csv", index=False)

rows = []
for endpoint, days in {
    "five_day": [8, 9, 10, 11, 12],
    "days_9_11": [9, 10, 11],
    "day_12": [12],
}.items():
    selected = merged[merged.day.isin(days)]
    values = selected.groupby("seed").learned_minus_control.sum()
    point = float(values.mean())
    if float(values.std(ddof=1)) == 0.0:
        lower = upper = point
        p_value = 1.0 if point == 0.0 else 0.0
    else:
        lower, upper = stats.t.interval(
            0.95, len(values) - 1, loc=point, scale=stats.sem(values)
        )
        p_value = float(stats.ttest_1samp(values, 0.0).pvalue)
    rows.append(
        {
            "endpoint": endpoint,
            "point": point,
            "lower": float(lower),
            "upper": float(upper),
            "p": p_value,
            "n": len(values),
            "maximum_absolute_daily_difference": float(
                selected.learned_minus_control.abs().max()
            ),
            "maximum_absolute_fixed_ration_difference_from_stage7": float(
                selected.current_fixed_minus_stage7_fixed.abs().max()
            ),
        }
    )
summary = pd.DataFrame(rows)
summary.to_csv(root / "results/stage11_component_b_control_audit_summary.csv", index=False)
print(summary.to_string(index=False))
