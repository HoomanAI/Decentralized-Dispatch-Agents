"""Sweep the operational threshold and measure active damage memory."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from code.env.dispatch import build_runtime_network, class_parameter_by_edge, damage_for_day


def run_memory_sweep(
    config: dict[str, Any], data_dir: Path, results_dir: Path
) -> Path:
    """Write blocked-mask identities and blocked-without-exposure counts."""
    graph, _, _ = build_runtime_network(
        data_dir / "synthetic_road_features.geojson",
        data_dir / "synthetic_fire_fronts.geojson",
        config["network"],
    )
    edges = list(graph.edges(data=True))
    rows = []
    for alpha in config["reliability"]["scenario_alpha"]:
        for gamma in config["reliability"]["scenario_gamma"]:
            for phi_min in config["reliability"]["scenario_phi_min"]:
                for concentration in config["reliability"][
                    "scenario_alpha_concentration"
                ]:
                    signatures = []
                    row: dict[str, Any] = {
                        "alpha": alpha,
                        "gamma": gamma,
                        "phi_min": phi_min,
                        "alpha_concentration": concentration,
                        "heterogeneous_alpha": concentration is not None,
                        "alpha_heterogeneity_seed": config["environment"][
                            "alpha_heterogeneity_seed"
                        ],
                    }
                    for day in range(7, 13):
                        exposure = np.array(
                            [
                                graph.graph["edge_day_flags"][data["edge_id"]][str(day)]
                                for _, _, data in edges
                            ],
                            dtype=bool,
                        )
                        damage = damage_for_day(
                            graph,
                            day,
                            float(alpha),
                            float(gamma),
                            concentration,
                            config["environment"]["alpha_heterogeneity_seed"],
                        )
                        upsilon = class_parameter_by_edge(
                            graph, config["environment"]["upsilon"], "upsilon"
                        )
                        reliability = (1.0 - damage) * (1.0 - upsilon * damage)
                        blocked = reliability < float(phi_min)
                        memory = blocked & ~exposure
                        signatures.append(np.packbits(blocked).tobytes())
                        row[f"day_{day}_exposed"] = int(np.count_nonzero(exposure))
                        row[f"day_{day}_blocked"] = int(np.count_nonzero(blocked))
                        row[f"day_{day}_blocked_not_exposed"] = int(
                            np.count_nonzero(memory)
                        )
                        row[f"day_{day}_exposed_not_blocked"] = int(
                            np.count_nonzero(exposure & ~blocked)
                        )
                    memory_counts = [
                        row[f"day_{day}_blocked_not_exposed"] for day in range(7, 13)
                    ]
                    row["memory_effect_total"] = int(sum(memory_counts))
                    row["memory_effect_peak"] = int(max(memory_counts))
                    row["memory_active"] = bool(row["memory_effect_total"] > 0)
                    row["network_realization_id"] = hashlib.sha256(
                        b"".join(signatures)
                    ).hexdigest()[:16]
                    rows.append(row)
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "stage4_memory_phi_sweep.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path
