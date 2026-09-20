"""Event-driven dispatch environment with feasibility masks and survival rewards."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import Point, shape

from code.belief.beta import BetaBelief

DEGREES_TO_KM = 91.0
VEHICLE_TRIAGE = {"A": 1, "B": 2, "C": 3}


def build_runtime_network(
    roads_geojson: Path,
    fronts_geojson: Path,
    network_config: dict[str, Any],
) -> tuple[nx.Graph, list[int], dict[int, int]]:
    """Load synthetic roads and add three external hospitals through gateway arcs."""
    features = json.loads(roads_geojson.read_text(encoding="utf-8"))["features"]
    graph = nx.Graph()
    edge_day_flags: dict[int, dict[str, int]] = {}
    for feature in features:
        properties = feature["properties"]
        if not properties["routable"]:
            continue
        coordinates = feature["geometry"]["coordinates"]
        u, v = int(properties["u"]), int(properties["v"])
        for node, coordinate in ((u, coordinates[0]), (v, coordinates[-1])):
            graph.add_node(node, x=float(coordinate[0]), y=float(coordinate[1]))
        edge_id = int(properties["feature_id"].split("_")[1])
        road_class = properties["fclass"]
        length_km = float(properties["length"]) * DEGREES_TO_KM
        speed = network_config["speed_kph"][road_class]
        graph.add_edge(
            u,
            v,
            edge_id=edge_id,
            fclass=road_class,
            class_group=properties.get("class_group", "local"),
            length_km=length_km,
            free_time_min=60.0 * length_km / speed,
            capacity=network_config["capacity_vehicles_per_hour"][road_class],
        )
        edge_day_flags[edge_id] = {
            str(day): int(properties[str(day)]) for day in range(7, 13)
        }

    fronts = json.loads(fronts_geojson.read_text(encoding="utf-8"))["features"]
    day_7_front = shape(next(item["geometry"] for item in fronts if item["properties"]["day"] == 7))
    bbox = network_config["bbox"]
    hospital_coordinates = [
        (bbox["east"] + 0.020, bbox["south"] - 0.015),
        (bbox["east"] + 0.025, bbox["north"] + 0.012),
        (bbox["west"] - 0.020, bbox["north"] + 0.018),
    ]
    hospitals = []
    first_access_edge = len(edge_day_flags)
    core_nodes = list(graph.nodes())
    for offset, coordinate in enumerate(hospital_coordinates):
        hospital = max(core_nodes) + 1 + offset
        hospital_point = Point(coordinate)
        assert not day_7_front.covers(hospital_point)
        gateway = min(
            core_nodes,
            key=lambda node: (graph.nodes[node]["x"] - coordinate[0]) ** 2
            + (graph.nodes[node]["y"] - coordinate[1]) ** 2,
        )
        distance = hospital_point.distance(
            Point(graph.nodes[gateway]["x"], graph.nodes[gateway]["y"])
        ) * DEGREES_TO_KM
        edge_id = first_access_edge + offset
        graph.add_node(hospital, x=coordinate[0], y=coordinate[1], hospital=True)
        graph.add_edge(
            hospital,
            gateway,
            edge_id=edge_id,
            fclass="hospital_access",
            class_group="gateway",
            length_km=distance,
            free_time_min=60.0 * distance / network_config["speed_kph"]["hospital_access"],
            capacity=network_config["capacity_vehicles_per_hour"]["hospital_access"],
        )
        edge_day_flags[edge_id] = {str(day): 0 for day in range(7, 13)}
        hospitals.append(hospital)
    edge_lookup = {
        int(data["edge_id"]): index
        for index, (_, _, data) in enumerate(graph.edges(data=True))
    }
    nx.set_node_attributes(graph, False, "hospital")
    for hospital in hospitals:
        graph.nodes[hospital]["hospital"] = True
    graph.graph["edge_day_flags"] = edge_day_flags
    return graph, hospitals, edge_lookup


def damage_for_day(
    graph: nx.Graph,
    day: int,
    alpha: float | dict[str, float],
    gamma: float | dict[str, float],
    alpha_concentration: float | None = None,
    alpha_seed: int = 0,
) -> np.ndarray:
    """Compute scenario damage through the selected day from observed exposure."""
    edge_count = graph.number_of_edges()
    damage = np.zeros(edge_count, dtype=float)
    class_groups = [str(data.get("class_group", "local")) for _, _, data in graph.edges(data=True)]

    def expand(value: float | dict[str, float], label: str) -> np.ndarray:
        if isinstance(value, dict):
            missing = sorted(set(class_groups) - set(value))
            if missing:
                raise ValueError(f"Missing {label} values for classes: {missing}")
            return np.asarray([float(value[group]) for group in class_groups])
        return np.full(edge_count, float(value), dtype=float)

    alpha_by_edge = expand(alpha, "alpha")
    gamma_by_edge = expand(gamma, "gamma")
    if alpha_concentration is None or np.all((alpha_by_edge == 0.0) | (alpha_by_edge == 1.0)):
        exposure_sensitivity = alpha_by_edge
    else:
        if alpha_concentration <= 0.0:
            raise ValueError("Alpha concentration must be positive")
        rng = np.random.default_rng(alpha_seed)
        exposure_sensitivity = rng.beta(
            alpha_by_edge * alpha_concentration,
            (1.0 - alpha_by_edge) * alpha_concentration,
        )
    for current_day in range(7, day + 1):
        exposure = np.array(
            [
                graph.graph["edge_day_flags"][data["edge_id"]][str(current_day)]
                for _, _, data in graph.edges(data=True)
            ],
            dtype=float,
        )
        damage = np.minimum(
            1.0, exposure_sensitivity * exposure + (1.0 - gamma_by_edge) * damage
        )
    return damage


def class_parameter_by_edge(
    graph: nx.Graph, value: float | dict[str, float], label: str
) -> np.ndarray:
    """Expand a scalar or class-indexed parameter in graph edge order."""
    if not isinstance(value, dict):
        return np.full(graph.number_of_edges(), float(value), dtype=float)
    groups = [str(data.get("class_group", "local")) for _, _, data in graph.edges(data=True)]
    missing = sorted(set(groups) - set(value))
    if missing:
        raise ValueError(f"Missing {label} values for classes: {missing}")
    return np.asarray([float(value[group]) for group in groups], dtype=float)


def _survival(delay: float, parameters: dict[str, float]) -> float:
    return parameters["a"] * math.exp(
        parameters["b"] * delay ** parameters["c"]
    ) + parameters["d"]


def attainable_survival(patients: pd.DataFrame, environment_config: dict[str, Any]) -> float:
    """Return weighted survival if every patient were served at zero delay."""
    total = 0.0
    for triage, count in patients["triage"].value_counts().items():
        parameters = environment_config["survival"][f"type_{int(triage)}"]
        total += float(count) * parameters["priority"] * _survival(0.0, parameters)
    return total


def _deadline(parameters: dict[str, float]) -> float:
    ratio = (parameters["lower"] - parameters["d"]) / parameters["a"]
    return (math.log(ratio) / parameters["b"]) ** (1.0 / parameters["c"])


def _edge_cost(
    attributes: dict[str, Any],
    reliability: float,
    affected_population: float,
    network_config: dict[str, Any],
) -> float:
    capacity_fraction = max(reliability, network_config["minimum_capacity_fraction"])
    capacity = attributes["capacity"]
    background = network_config["background_volume_fraction"] * capacity
    evacuation = network_config["evacuation_volume_scale"] * affected_population
    ratio = (background + evacuation) / (capacity_fraction * capacity)
    return attributes["free_time_min"] * (
        1.0 + network_config["bpr_rho"] * ratio ** network_config["bpr_beta"]
    )


def build_traversable_graph(
    graph: nx.Graph,
    reliability: np.ndarray,
    edge_lookup: dict[int, int],
    phi_min: float,
    affected_population: float,
    network_config: dict[str, Any],
) -> nx.Graph:
    """Build one weighted graph for repeated routing under a fixed belief state."""
    usable = nx.Graph()
    for u, v, attributes in graph.edges(data=True):
        index = edge_lookup[int(attributes["edge_id"])]
        if reliability[index] < phi_min:
            continue
        cost = _edge_cost(
            attributes, reliability[index], affected_population, network_config
        )
        usable.add_edge(
            u,
            v,
            weight=cost,
            length_km=attributes["length_km"],
            edge_id=int(attributes["edge_id"]),
            class_group=attributes.get("class_group", "local"),
        )
    return usable


def shortest_path_metrics(
    graph: nx.Graph,
    source: int,
    target: int,
    reliability: np.ndarray,
    edge_lookup: dict[int, int],
    phi_min: float,
    affected_population: float,
    network_config: dict[str, Any],
    prepared_graph: nx.Graph | None = None,
) -> tuple[list[int], float, float] | None:
    usable = prepared_graph
    if usable is None:
        usable = build_traversable_graph(
            graph,
            reliability,
            edge_lookup,
            phi_min,
            affected_population,
            network_config,
        )
    try:
        nodes = nx.shortest_path(usable, source, target, weight="weight")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None
    edge_ids = [int(graph.edges[u, v]["edge_id"]) for u, v in zip(nodes[:-1], nodes[1:])]
    travel_time = sum(usable.edges[u, v]["weight"] for u, v in zip(nodes[:-1], nodes[1:]))
    length = sum(usable.edges[u, v]["length_km"] for u, v in zip(nodes[:-1], nodes[1:]))
    return edge_ids, float(travel_time), float(length)


def run_episode(
    graph: nx.Graph,
    hospitals: list[int],
    edge_lookup: dict[int, int],
    patients: pd.DataFrame,
    day: int,
    baseline: str,
    network_config: dict[str, Any],
    belief_config: dict[str, Any],
    environment_config: dict[str, Any],
    affected_population: float,
    controller: Any | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run one myopic episode and return summary metrics plus a replayable trace."""
    damage = damage_for_day(
        graph,
        day,
        environment_config["scenario_alpha"],
        environment_config["scenario_gamma"],
        environment_config.get("scenario_alpha_concentration"),
        environment_config.get("alpha_heterogeneity_seed", 0),
    )
    upsilon = class_parameter_by_edge(graph, environment_config["upsilon"], "upsilon")
    true_reliability = (1.0 - damage) * (1.0 - upsilon * damage)
    blocked_arc_count = int(
        np.count_nonzero(true_reliability < environment_config["phi_min"])
    )
    belief = BetaBelief.from_reliability(
        1.0 - damage,
        belief_config["prior_strength"],
        belief_config["discount"],
        belief_config["corridor_pooling"],
    )
    vehicles = [
        {
            "vehicle_id": f"{kind}_{index}",
            "kind": kind,
            "node": hospitals[index % len(hospitals)],
            "available": 0.0,
        }
        for kind in VEHICLE_TRIAGE
        for index in range(
            int(
                environment_config.get("fleet_by_class", {}).get(
                    kind, environment_config["fleet_per_class"]
                )
            )
        )
    ]
    trace: list[dict[str, Any]] = []
    total_survival = 0.0
    served = 0
    unreachable = 0
    blocked_arc_encounters = 0
    known_blocked_edges: set[int] = set()
    edge_by_id = {
        int(data["edge_id"]): (u, v, data) for u, v, data in graph.edges(data=True)
    }
    for patient in patients.sort_values("onset_minute").to_dict("records"):
        triage = int(patient["triage"])
        survival_parameters = environment_config["survival"][f"type_{triage}"]
        deadline = _deadline(survival_parameters)
        candidates = [vehicle for vehicle in vehicles if VEHICLE_TRIAGE[vehicle["kind"]] == triage]
        route_reliability = (
            np.where(
                true_reliability >= environment_config["phi_min"],
                belief.mean,
                0.0,
            )
            if baseline == "perfect_information"
            else belief.mean
            if baseline == "belief_aware"
            else np.ones_like(true_reliability)
        )
        route_graph = build_traversable_graph(
            graph,
            route_reliability,
            edge_lookup,
            environment_config["phi_min"],
            affected_population,
            network_config,
        )
        for edge_id in known_blocked_edges:
            u, v, _ = edge_by_id[edge_id]
            if route_graph.has_edge(u, v):
                route_graph.remove_edge(u, v)
        if triage < 3:
            hospital_routes = [
                (
                    hospital,
                    shortest_path_metrics(
                        graph,
                        int(patient["node"]),
                        hospital,
                        route_reliability,
                        edge_lookup,
                        environment_config["phi_min"],
                        affected_population,
                        network_config,
                        route_graph,
                    ),
                )
                for hospital in hospitals
            ]
            hospital_routes = [item for item in hospital_routes if item[1] is not None]
        else:
            hospital_routes = []
        options = []
        for vehicle in candidates:
            start_time = max(float(patient["onset_minute"]), vehicle["available"])
            inbound = shortest_path_metrics(
                graph,
                vehicle["node"],
                int(patient["node"]),
                route_reliability,
                edge_lookup,
                environment_config["phi_min"],
                affected_population,
                network_config,
                route_graph,
            )
            if inbound is None or start_time + inbound[1] - patient["onset_minute"] > deadline:
                continue
            if triage < 3:
                if not hospital_routes:
                    continue
                destination, outbound = min(
                    hospital_routes, key=lambda item: item[1][1]
                )
            else:
                outbound = ([], 0.0, 0.0)
                destination = int(patient["node"])
            total_length = inbound[2] + outbound[2]
            if total_length > environment_config["battery_range_km"]:
                continue
            options.append(
                (start_time + inbound[1], vehicle, inbound, outbound, destination)
            )
        if not options:
            unreachable += 1
            trace.append(
                {
                    "epoch_minute": patient["onset_minute"],
                    "patient_id": patient["patient_id"],
                    "assignment": None,
                    "candidate_route_count": 0,
                    "arcs_traversed": [],
                    "observations": [],
                    "blocked_arc_encounters": 0,
                    "belief_before": {},
                    "belief_after": {},
                    "outcome": "unreachable",
                    "patient_reward": 0.0,
                }
            )
            continue
        if controller is None:
            selected_option = min(options, key=lambda option: option[0])
        else:
            selected_index = controller.select(
                graph=graph,
                belief=belief,
                patient=patient,
                options=options,
                operational_minutes=float(environment_config.get("operational_minutes", 720.0)),
                survival_parameters=survival_parameters,
            )
            selected_option = options[selected_index]
        _, vehicle, inbound, outbound, destination = selected_option
        before: dict[str, float] = {}
        observations = []
        traversed_edges = []
        current_node = int(vehicle["node"])
        current_time = max(float(patient["onset_minute"]), float(vehicle["available"]))
        travelled_length = 0.0

        def traverse_leg(target: int, enforce_deadline: bool) -> bool:
            nonlocal current_node, current_time, travelled_length, blocked_arc_encounters
            while current_node != target:
                current_route_reliability = (
                    np.where(
                        true_reliability >= environment_config["phi_min"],
                        belief.mean,
                        0.0,
                    )
                    if baseline == "perfect_information"
                    else belief.mean
                    if baseline == "belief_aware"
                    else np.ones_like(true_reliability)
                )
                prepared = build_traversable_graph(
                    graph,
                    current_route_reliability,
                    edge_lookup,
                    environment_config["phi_min"],
                    affected_population,
                    network_config,
                )
                for known_edge_id in known_blocked_edges:
                    known_u, known_v, _ = edge_by_id[known_edge_id]
                    if prepared.has_edge(known_u, known_v):
                        prepared.remove_edge(known_u, known_v)
                plan = shortest_path_metrics(
                    graph,
                    current_node,
                    target,
                    current_route_reliability,
                    edge_lookup,
                    environment_config["phi_min"],
                    affected_population,
                    network_config,
                    prepared,
                )
                if plan is None:
                    return False
                encountered_block = False
                for edge_id in plan[0]:
                    edge_index = edge_lookup[edge_id]
                    before.setdefault(
                        str(edge_id), float(belief.mean[edge_index])
                    )
                    passed = bool(
                        true_reliability[edge_index] >= environment_config["phi_min"]
                    )
                    edge = edge_by_id[edge_id]
                    peer_ids = {
                        int(graph.edges[neighbor]["edge_id"])
                        for node in edge[:2]
                        for neighbor in graph.edges(node)
                        if graph.edges[neighbor]["fclass"] == edge[2]["fclass"]
                    }
                    peers = np.array(
                        [edge_lookup[item] for item in peer_ids], dtype=int
                    )
                    belief.update(edge_index, passed, peers)
                    observations.append({"edge_id": edge_id, "passed": passed})
                    traversed_edges.append(edge_id)
                    if not passed:
                        known_blocked_edges.add(edge_id)
                        blocked_arc_encounters += 1
                        current_time += float(
                            environment_config["blocked_encounter_minutes"]
                        )
                        encountered_block = True
                        break
                    u, v, attributes = edge
                    next_node = v if current_node == u else u
                    current_time += float(prepared.edges[current_node, next_node]["weight"])
                    travelled_length += float(attributes["length_km"])
                    current_node = int(next_node)
                    if travelled_length > environment_config["battery_range_km"]:
                        return False
                    if (
                        enforce_deadline
                        and current_time - float(patient["onset_minute"]) > deadline
                    ):
                        return False
                if not encountered_block and current_node != target:
                    return False
            return True

        completed = traverse_leg(int(patient["node"]), True)
        patient_arrival = current_time
        if completed and triage < 3:
            completed = traverse_leg(int(destination), False)
        after = {
            edge: float(belief.mean[edge_lookup[int(edge)]]) for edge in before
        }
        if completed:
            delay = patient_arrival - patient["onset_minute"]
            patient_reward = survival_parameters["priority"] * _survival(
                delay, survival_parameters
            )
            total_survival += patient_reward
            served += 1
            vehicle["available"] = current_time + environment_config["service_minutes"]
            vehicle["node"] = current_node
            outcome = "served"
        else:
            patient_reward = 0.0
            unreachable += 1
            vehicle["available"] = current_time
            vehicle["node"] = current_node
            outcome = "blocked"
        if controller is not None:
            controller.observe(float(patient_reward))
        trace.append(
            {
                "epoch_minute": patient["onset_minute"],
                "patient_id": patient["patient_id"],
                "assignment": vehicle["vehicle_id"],
                "candidate_route_count": 1,
                "arcs_traversed": traversed_edges,
                "observations": observations,
                "blocked_arc_encounters": sum(
                    not observation["passed"] for observation in observations
                ),
                "belief_before": before,
                "belief_after": after,
                "outcome": outcome,
                "patient_reward": float(patient_reward),
            }
        )
    belief_changed = any(item["belief_before"] != item["belief_after"] for item in trace)
    if controller is not None:
        controller.finish_episode()
    patient_count = len(patients)
    maximum_survival = attainable_survival(patients, environment_config)
    return (
        {
            "day": day,
            "baseline": baseline,
            "patients": patient_count,
            "served": served,
            "unreachable": unreachable,
            "survival_total": total_survival,
            "survival_per_patient": total_survival / patient_count if patient_count else 0.0,
            "survival_fraction_attainable": (
                total_survival / maximum_survival if maximum_survival else 0.0
            ),
            "attainable_survival": maximum_survival,
            "belief_changed": belief_changed,
            "blocked_arc_count": blocked_arc_count,
            "blocked_arc_encounters": blocked_arc_encounters,
        },
        trace,
    )
