"""Generate provisional two-stream patient demand over an operational day."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def generate_patients(
    day: int,
    affected_population: float,
    evacuated_population: float,
    candidate_nodes: np.ndarray,
    age_65_share: float,
    config: dict[str, Any],
    seed: int,
) -> pd.DataFrame:
    """Sample a nonhomogeneous Poisson demand scenario, not observed incidents."""
    rng = np.random.default_rng(seed)
    fire_mean = config["provisional_fire_incidence"] * affected_population
    # The evacuation stream follows the persistent displaced population, not the
    # moving daily fire footprint. The released Total is the union count.
    evacuation_mean = (
        config["provisional_evacuation_incidence"]
        * evacuated_population
        * age_65_share
        * config["vulnerable_weight"]
    )
    expected_count = fire_mean + evacuation_mean
    profile = np.asarray(config["temporal_profile"], dtype=float)
    profile = profile / profile.sum()
    if "minimum_patients" in config or "maximum_patients" in config:
        total_count = int(rng.poisson(expected_count))
        total_count = max(int(config.get("minimum_patients", 0)), total_count)
        total_count = min(int(config.get("maximum_patients", total_count)), total_count)
        hour_counts = rng.multinomial(total_count, profile)
    else:
        hour_counts = rng.poisson(expected_count * profile)
    records = []
    probabilities = np.asarray(config["triage_base_probabilities"], dtype=float)
    probabilities = probabilities / probabilities.sum()
    patient_id = 0
    for hour, count in enumerate(hour_counts):
        for onset in np.sort(rng.uniform(hour * 60.0, (hour + 1) * 60.0, count)):
            records.append(
                {
                    "patient_id": f"d{day}_p{patient_id:04d}",
                    "day": day,
                    "onset_minute": float(onset),
                    "node": int(rng.choice(candidate_nodes)),
                    "triage": int(rng.choice([1, 2, 3], p=probabilities)),
                    "stream": "evacuation"
                    if rng.random() < evacuation_mean / expected_count
                    else "fire",
                }
            )
            patient_id += 1
    columns = ["patient_id", "day", "onset_minute", "node", "triage", "stream"]
    return pd.DataFrame.from_records(records, columns=columns)
