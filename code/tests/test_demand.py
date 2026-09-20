from __future__ import annotations

import numpy as np

from code.demand.patients import generate_patients


def test_evacuation_demand_persists_when_daily_fire_footprint_shrinks() -> None:
    config = {
        "provisional_fire_incidence": 0.0,
        "provisional_evacuation_incidence": 0.03,
        "vulnerable_weight": 2.0,
        "temporal_profile": [1.0],
        "triage_base_probabilities": [0.2, 0.3, 0.5],
    }
    patients = generate_patients(
        day=12,
        affected_population=0.0,
        evacuated_population=6729.9,
        candidate_nodes=np.array([1, 2]),
        age_65_share=0.257,
        config=config,
        seed=7,
    )
    assert len(patients) > 0
    assert set(patients["stream"]) == {"evacuation"}
