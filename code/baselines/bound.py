"""Full-information daily dispatch ceiling solved as a time-expanded MILP.

Vehicles of each clinical class are pooled and each patient uses the best reachable
hospital gateway. This optimistic relaxation preserves fleet capacity and service-time
conflicts while ensuring the result remains a ceiling for decentralized policies.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from code.env.dispatch import (
    _deadline,
    _survival,
    attainable_survival,
    build_traversable_graph,
    class_parameter_by_edge,
    damage_for_day,
)


def run_optimistic_bound(
    graph: nx.Graph,
    hospitals: list[int],
    edge_lookup: dict[int, int],
    patients: pd.DataFrame,
    day: int,
    network_config: dict[str, Any],
    environment_config: dict[str, Any],
    operational_minutes: float,
    affected_population: float,
    charge_travel_capacity: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Maximize discretized survival with the realized road state known."""
    damage = damage_for_day(
        graph,
        day,
        environment_config["scenario_alpha"],
        environment_config["scenario_gamma"],
        environment_config.get("scenario_alpha_concentration"),
        environment_config.get("alpha_heterogeneity_seed", 0),
    )
    upsilon = class_parameter_by_edge(graph, environment_config["upsilon"], "upsilon")
    reliability = (1.0 - damage) * (1.0 - upsilon * damage)
    route_graph = build_traversable_graph(
        graph,
        reliability,
        edge_lookup,
        environment_config["phi_min"],
        affected_population,
        network_config,
    )
    hospital_distances = [
        nx.single_source_dijkstra_path_length(route_graph, hospital, weight="weight")
        for hospital in hospitals
        if hospital in route_graph
    ]
    bin_width = float(environment_config["bound_delay_bin_minutes"])
    slot_count = int(math.ceil(operational_minutes / bin_width))
    service_minutes = float(environment_config["service_minutes"])
    fleet_capacity = int(environment_config["fleet_per_class"])
    records = patients.sort_values("onset_minute").to_dict("records")
    variable_keys: list[tuple[str, int]] = []
    rewards: dict[tuple[str, int], float] = {}
    metadata: dict[tuple[str, int], dict[str, float | int]] = {}
    occupancy: dict[tuple[int, int], list[int]] = defaultdict(list)
    patient_variables: dict[str, list[int]] = defaultdict(list)

    for record in records:
        patient_id = str(record["patient_id"])
        patient_node = int(record["node"])
        triage = int(record["triage"])
        reachable = [
            (hospital, distances[patient_node])
            for hospital, distances in zip(hospitals, hospital_distances)
            if patient_node in distances
        ]
        if not reachable:
            continue
        hospital, inbound = min(reachable, key=lambda item: item[1])
        # The relaxed bound charges service only. The travel-capacity check charges
        # inbound travel, service, and the return trip to the selected gateway.
        resource_duration = (
            inbound + service_minutes + inbound
            if charge_travel_capacity
            else service_minutes
        )
        parameters = environment_config["survival"][f"type_{triage}"]
        deadline = _deadline(parameters)
        onset = float(record["onset_minute"])
        first_slot = int(math.ceil(onset / bin_width))
        last_slot = min(
            slot_count - 1,
            int(math.floor((onset + deadline - inbound) / bin_width)),
            first_slot + int(environment_config["bound_max_start_options"]) - 1,
        )
        for start_slot in range(first_slot, last_slot + 1):
            start_minute = start_slot * bin_width
            finish_minute = start_minute + resource_duration
            if finish_minute > operational_minutes:
                continue
            delay = start_minute + inbound - onset
            key = (patient_id, start_slot)
            variable_index = len(variable_keys)
            variable_keys.append(key)
            patient_variables[patient_id].append(variable_index)
            reward = parameters["priority"] * _survival(delay, parameters)
            rewards[key] = reward
            metadata[key] = {
                "hospital_node": hospital,
                "arrival_minute": start_minute + inbound,
                "delay_minutes": delay,
                "weighted_survival": reward,
                "triage": triage,
            }
            final_slot = min(
                slot_count - 1, int(math.ceil(finish_minute / bin_width)) - 1
            )
            for busy_slot in range(start_slot, final_slot + 1):
                occupancy[triage, busy_slot].append(variable_index)

    constraint_groups = list(patient_variables.values()) + list(occupancy.values())
    matrix = lil_matrix((len(constraint_groups), len(variable_keys)), dtype=float)
    upper = np.ones(len(constraint_groups), dtype=float)
    patient_constraint_count = len(patient_variables)
    upper[patient_constraint_count:] = fleet_capacity
    for row_index, indices in enumerate(constraint_groups):
        matrix[row_index, indices] = 1.0
    objective = -np.array([rewards[key] for key in variable_keys], dtype=float)
    result = milp(
        c=objective,
        integrality=np.ones(len(variable_keys), dtype=int),
        bounds=Bounds(0.0, 1.0),
        constraints=LinearConstraint(matrix.tocsr(), 0.0, upper),
        options={
            "time_limit": float(environment_config["bound_time_limit_seconds"]),
            "mip_rel_gap": 0.0,
        },
    )
    if result.x is None:
        raise RuntimeError(f"Bound produced no feasible solution on day {day}: {result.message}")
    status = "Optimal" if result.success else str(result.message)
    selected = {
        key: metadata[key]
        for index, key in enumerate(variable_keys)
        if result.x[index] > 0.5
    }
    selected_by_patient = {key[0]: (key, details) for key, details in selected.items()}
    rows = []
    for record in records:
        patient_id = str(record["patient_id"])
        if patient_id not in selected_by_patient:
            rows.append({"patient_id": patient_id, "vehicle_pool": None, "outcome": "unserved"})
            continue
        key, details = selected_by_patient[patient_id]
        rows.append(
            {
                "patient_id": patient_id,
                "vehicle_pool": f"class_{details['triage']}",
                "dispatch_slot": key[1],
                **details,
                "outcome": "served",
            }
        )
    total_survival = float(sum(item["weighted_survival"] for item in selected.values()))
    maximum = attainable_survival(patients, environment_config)
    count = len(patients)
    served = len(selected)
    return (
        {
            "day": day,
            "baseline": (
                "travel_capacity_bound"
                if charge_travel_capacity
                else "optimistic_bound"
            ),
            "patients": count,
            "served": served,
            "unreachable": count - served,
            "survival_total": total_survival,
            "survival_per_patient": total_survival / count if count else 0.0,
            "survival_fraction_attainable": total_survival / maximum if maximum else 0.0,
            "attainable_survival": maximum,
            "solver_status": status,
            "solver_objective_bound": float(-result.mip_dual_bound),
            "solver_mip_gap": float(result.mip_gap),
            "objective_upper_bound_note": (
                "Five-minute full-information pooled-fleet MILP with travel capacity"
                if charge_travel_capacity
                else "Five-minute full-information optimistic pooled-fleet MILP with travel capacity relaxed"
            ),
        },
        rows,
    )
