"""Vehicle-centric multi-agent dispatch with local beliefs and collision resolution."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Protocol

import networkx as nx
import numpy as np
import pandas as pd
from scipy.special import betaln, digamma

from code.belief.beta import BetaBelief
from code.env.dispatch import (
    VEHICLE_TRIAGE,
    _deadline,
    _survival,
    build_traversable_graph,
    class_parameter_by_edge,
    damage_for_day,
    shortest_path_metrics,
)


@dataclass
class Candidate:
    patient_id: str | None
    patient: dict[str, Any] | None
    inbound: tuple[list[int], float, float] | None
    outbound: tuple[list[int], float, float] | None
    destination: int
    estimated_arrival: float
    immediate_survival: float
    exploration_cost: float = 0.0
    information_value: float = 0.0
    detour_distance_km: float = 0.0
    route_rank: int = 0


def _beta_information_gain(a: float, b: float) -> float:
    """Expected one-observation reduction in Beta differential entropy."""
    def entropy(x: float, y: float) -> float:
        return float(
            betaln(x, y)
            - (x - 1.0) * digamma(x)
            - (y - 1.0) * digamma(y)
            + (x + y - 2.0) * digamma(x + y)
        )

    probability = a / (a + b)
    reduction = entropy(a, b) - (
        probability * entropy(a + 1.0, b)
        + (1.0 - probability) * entropy(a, b + 1.0)
    )
    return max(float(reduction), 0.0)


def _path_metrics(graph: nx.Graph, usable: nx.Graph, nodes: list[int]) -> tuple[list[int], float, float]:
    edge_ids = [int(graph.edges[u, v]["edge_id"]) for u, v in zip(nodes[:-1], nodes[1:])]
    travel_time = sum(float(usable.edges[u, v]["weight"]) for u, v in zip(nodes[:-1], nodes[1:]))
    distance = sum(float(usable.edges[u, v]["length_km"]) for u, v in zip(nodes[:-1], nodes[1:]))
    return edge_ids, float(travel_time), float(distance)


class VehiclePolicy(Protocol):
    def propose(self, context: dict[str, Any]) -> tuple[int, float]: ...
    def observe(self, reward: float, terminal: bool = False) -> None: ...


LOOKAHEAD_FEATURE_NAMES = (
    "day_index",
    "days_remaining",
    "arterial_cumulative_exposure",
    "arterial_days_since_exposure",
    "local_cumulative_exposure",
    "local_days_since_exposure",
    "queue_type_1_share",
    "queue_type_2_share",
    "queue_type_3_share",
    "queue_type_1_wait",
    "queue_type_2_wait",
    "queue_type_3_wait",
    "busy_vehicle_share",
    "mean_time_until_free",
    "maximum_time_until_free",
    "reachable_belief_width",
)


def _lookahead_features(
    graph: nx.Graph,
    route_graph: nx.Graph,
    vehicle: dict[str, Any],
    vehicles: list[dict[str, Any]],
    pending: list[dict[str, Any]],
    belief: BetaBelief,
    edge_lookup: dict[int, int],
    day: int,
    now: float,
) -> np.ndarray:
    """Return history/current-state features without any future exposure information."""
    if int(vehicle["node"]) in route_graph:
        reachable_nodes = nx.node_connected_component(route_graph, int(vehicle["node"]))
        reachable_edges = list(route_graph.subgraph(reachable_nodes).edges(data=True))
    else:
        reachable_edges = []
    flags = graph.graph["edge_day_flags"]
    history = []
    for group in ("arterial", "local"):
        edges = [
            data for _, _, data in reachable_edges if data.get("class_group", "local") == group
        ]
        cumulative = []
        recency = []
        for data in edges:
            daily = flags[int(data["edge_id"])]
            observed_days = [past for past in range(7, day + 1) if int(daily[str(past)])]
            cumulative.append(sum(int(daily[str(past)]) for past in range(7, day + 1)) / 6.0)
            recency.append((day - max(observed_days)) / 5.0 if observed_days else 1.0)
        history.extend(
            [float(np.mean(cumulative)) if cumulative else 0.0,
             float(np.mean(recency)) if recency else 1.0]
        )
    queue_shares = []
    queue_waits = []
    denominator = max(len(pending), 1)
    for triage in (1, 2, 3):
        group = [item for item in pending if int(item["triage"]) == triage]
        queue_shares.append(len(group) / denominator)
        queue_waits.append(
            float(np.mean([max(now - float(item["onset_minute"]), 0.0) for item in group])) / 720.0
            if group else 0.0
        )
    free_delays = np.asarray(
        [max(float(item["available"]) - now, 0.0) for item in vehicles], dtype=float
    )
    reachable_indices = [edge_lookup[int(data["edge_id"])] for _, _, data in reachable_edges]
    total = belief.a + belief.b
    width = np.sqrt(belief.a * belief.b / (total**2 * (total + 1.0)))
    uncertainty = float(np.mean(width[reachable_indices])) if reachable_indices else 0.0
    return np.asarray(
        [
            (day - 7.0) / 5.0,
            (12.0 - day) / 5.0,
            *history,
            *queue_shares,
            *queue_waits,
            float(np.mean(free_delays > 0.0)),
            float(np.mean(free_delays)) / 720.0,
            float(np.max(free_delays)) / 720.0,
            uncertainty,
        ],
        dtype=np.float32,
    )


def _merge_component(beliefs: list[BetaBelief], indices: list[int]) -> None:
    """Pool component evidence without recounting observations already shared."""
    if len(indices) < 2:
        return
    reference = beliefs[indices[0]]
    evidence_a = np.stack(
        [np.maximum(beliefs[i].a - beliefs[i].a0, 0.0) for i in indices]
    )
    evidence_b = np.stack(
        [np.maximum(beliefs[i].b - beliefs[i].b0, 0.0) for i in indices]
    )
    # Without per-observation provenance, elementwise union must be idempotent so that
    # reconnecting the same copies cannot manufacture evidence. Maximum is associative,
    # commutative, and conservative when two vehicles independently observe one arc.
    merged_a = reference.a0 + np.max(evidence_a, axis=0)
    merged_b = reference.b0 + np.max(evidence_b, axis=0)
    for i in indices:
        beliefs[i].a = merged_a.copy()
        beliefs[i].b = merged_b.copy()


def _components(vehicle_count: int, reliability: float, rng: np.random.Generator) -> list[list[int]]:
    communication = nx.Graph()
    communication.add_nodes_from(range(vehicle_count))
    for i in range(vehicle_count):
        for j in range(i + 1, vehicle_count):
            if rng.random() <= reliability:
                communication.add_edge(i, j)
    return [sorted(component) for component in nx.connected_components(communication)]


def _candidate_set(
    graph: nx.Graph,
    edge_lookup: dict[int, int],
    hospitals: list[int],
    vehicle: dict[str, Any],
    belief: BetaBelief,
    pending: list[dict[str, Any]],
    now: float,
    affected_population: float,
    network_config: dict[str, Any],
    environment: dict[str, Any],
    route_graph: nx.Graph | None = None,
    route_metrics_cache: dict[
        tuple[int, int], tuple[list[int], float, float] | None
    ] | None = None,
) -> list[Candidate]:
    if route_graph is None:
        route_graph = build_traversable_graph(
            graph,
            belief.mean,
            edge_lookup,
            environment["phi_min"],
            affected_population,
            network_config,
        )
    if route_metrics_cache is None:
        route_metrics_cache = {}

    def cached_route(source: int, target: int) -> tuple[list[int], float, float] | None:
        key = (source, target)
        if key not in route_metrics_cache:
            route_metrics_cache[key] = shortest_path_metrics(
                graph,
                source,
                target,
                belief.mean,
                edge_lookup,
                environment["phi_min"],
                affected_population,
                network_config,
                route_graph,
            )
        return route_metrics_cache[key]

    candidates: list[Candidate] = []
    for patient in pending:
        triage = int(patient["triage"])
        if VEHICLE_TRIAGE[vehicle["kind"]] != triage:
            continue
        parameters = environment["survival"][f"type_{triage}"]
        inbound = cached_route(int(vehicle["node"]), int(patient["node"]))
        if inbound is None or now + inbound[1] - float(patient["onset_minute"]) > _deadline(parameters):
            continue
        if triage < 3:
            routes = [(h, cached_route(int(patient["node"]), h)) for h in hospitals]
            routes = [
                (h, route_metrics)
                for h, route_metrics in routes
                if route_metrics is not None
            ]
            if not routes:
                continue
            destination, outbound = min(routes, key=lambda item: item[1][1])
        else:
            destination, outbound = int(patient["node"]), ([], 0.0, 0.0)
        if inbound[2] + outbound[2] > environment["battery_range_km"]:
            continue
        routes = [inbound]
        exploration_budgets = environment.get("exploration_budgets", (np.inf, np.inf, np.inf))
        if environment.get("exploration_enabled", False) and exploration_budgets[triage - 1] > 0.0:
            route_limit = int(environment.get("exploration_k", 3)) + 1
            detour_limit = float(environment.get("exploration_detour_minutes", 20.0))
            try:
                node_paths = nx.shortest_simple_paths(route_graph, int(vehicle["node"]), int(patient["node"]), weight="weight")
                routes = []
                for nodes_path in node_paths:
                    route = _path_metrics(graph, route_graph, nodes_path)
                    if route[1] <= inbound[1] + detour_limit + 1.0e-9:
                        routes.append(route)
                    if len(routes) >= route_limit:
                        break
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                routes = [inbound]
            if not routes:
                routes = [inbound]
        base_delay = max(now + inbound[1] - float(patient["onset_minute"]), 0.0)
        base_reward = parameters["priority"] * _survival(base_delay, parameters)
        for route_rank, selected_inbound in enumerate(routes):
            delay = max(now + selected_inbound[1] - float(patient["onset_minute"]), 0.0)
            reward = parameters["priority"] * _survival(delay, parameters)
            information = sum(
                _beta_information_gain(float(belief.a[edge_lookup[edge_id]]), float(belief.b[edge_lookup[edge_id]]))
                for edge_id in selected_inbound[0]
            )
            candidates.append(Candidate(str(patient["patient_id"]), patient, selected_inbound, outbound, int(destination), now + selected_inbound[1], reward, max(base_reward - reward, 0.0), information, max(selected_inbound[2] - inbound[2], 0.0), route_rank))
    candidates.append(Candidate(None, None, None, None, int(vehicle["node"]), now, 0.0))
    return candidates


def _resolve(
    vehicle_indices: list[int],
    contexts: dict[int, dict[str, Any]],
    policies: list[VehiclePolicy],
) -> tuple[dict[int, Candidate], int]:
    remaining = {i: list(range(len(contexts[i]["candidates"]))) for i in vehicle_indices}
    committed: dict[int, Candidate] = {}
    rounds = 0
    while remaining:
        rounds += 1
        proposals: dict[int, tuple[int, float]] = {}
        for i, allowed in remaining.items():
            local = dict(contexts[i])
            local["allowed_indices"] = allowed
            proposals[i] = policies[i].propose(local)
        by_patient: dict[str, list[tuple[int, int, float]]] = {}
        for i, (choice, _policy_score) in proposals.items():
            candidate = contexts[i]["candidates"][choice]
            if candidate.patient_id is None:
                committed[i] = candidate
            else:
                # Collision priority is a fixed state function, not a policy-derived
                # probability or logit. This keeps scores comparable across vehicles
                # with different feasible action sets and keeps transition dynamics
                # stationary while the policy trains.
                by_patient.setdefault(candidate.patient_id, []).append(
                    (i, choice, float(candidate.immediate_survival))
                )
        for contenders in by_patient.values():
            winner, choice, _ = max(contenders, key=lambda item: (item[2], -item[0]))
            committed[winner] = contexts[winner]["candidates"][choice]
            for loser, _, _ in contenders:
                if loser != winner:
                    remaining[loser] = [index for index in remaining[loser] if contexts[loser]["candidates"][index].patient_id != contexts[winner]["candidates"][choice].patient_id]
        for i in list(committed):
            remaining.pop(i, None)
        assert rounds <= len(vehicle_indices)
    ordered = sorted(committed)
    for position, vehicle_index in enumerate(ordered):
        if hasattr(policies[vehicle_index], "set_other_actions"):
            earlier = position
            later = len(ordered) - position - 1
            later_noop = sum(
                committed[item].patient_id is None for item in ordered[position + 1 :]
            )
            policies[vehicle_index].set_other_actions(
                np.array([earlier, later, later_noop], dtype=float)
                / max(len(ordered), 1)
            )
    return committed, rounds


class ProtocolMyopic:
    def propose(self, context: dict[str, Any]) -> tuple[int, float]:
        allowed = context["allowed_indices"]
        choice = max(allowed, key=lambda i: context["candidates"][i].immediate_survival)
        return choice, float(context["candidates"][choice].immediate_survival)

    def observe(self, reward: float, terminal: bool = False) -> None:
        del reward, terminal


def run_vehicle_episode(
    graph: nx.Graph,
    hospitals: list[int],
    edge_lookup: dict[int, int],
    patients: pd.DataFrame,
    day: int,
    policies: list[VehiclePolicy],
    network_config: dict[str, Any],
    belief_config: dict[str, Any],
    environment: dict[str, Any],
    affected_population: float,
    communication_reliability: float = 1.0,
    seed: int = 0,
    belief_evidence: list[tuple[np.ndarray, np.ndarray]] | None = None,
    alternative_route_audit: set[int] | None = None,
    posterior_audit: dict[str, Any] | None = None,
    communication_audit: dict[str, int] | None = None,
    isolation_epochs_state: list[int] | None = None,
    centralized_stale_view: bool = False,
    planner_evidence: list[tuple[np.ndarray, np.ndarray]] | None = None,
    belief_architecture: str = "decentralized_merge",
    architecture_state: dict[str, Any] | None = None,
    architecture_audit: dict[str, Any] | None = None,
    mark_terminal: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run one event-driven vehicle-centric episode."""
    valid_architectures = {
        "decentralized_merge",
        "decentralized_no_merge",
        "centralized_coordinator",
    }
    if belief_architecture not in valid_architectures:
        raise ValueError(f"Unknown belief architecture: {belief_architecture}")
    if centralized_stale_view and belief_architecture != "decentralized_merge":
        raise ValueError("Legacy stale view cannot be combined with a belief architecture")
    damage = damage_for_day(graph, day, environment["scenario_alpha"], environment["scenario_gamma"], environment.get("scenario_alpha_concentration"), environment.get("alpha_heterogeneity_seed", 0))
    upsilon = class_parameter_by_edge(graph, environment["upsilon"], "upsilon")
    true_reliability = (1.0 - damage) * (1.0 - upsilon * damage)
    prior = 1.0 - damage
    prior_strength = float(belief_config["prior_strength"])
    if environment.get("belief_blind", False):
        # Stage 9 reliability-blind dispatch begins from an intact-network belief.
        # A weak prior lets a failed contact close the encountered arc immediately.
        prior = np.full_like(damage, 1.0 - 1.0e-9)
        prior_strength = float(environment.get("blind_prior_strength", 1.0e-6))
    if environment.get("perfect_information", False):
        prior = np.where(
            true_reliability >= environment["phi_min"],
            prior,
            1.0e-6,
        )
        prior_strength = max(prior_strength, 1.0e9)
    vehicles = [{"vehicle_id": f"{kind}_{index}", "kind": kind, "node": hospitals[index % len(hospitals)], "available": 0.0} for kind in VEHICLE_TRIAGE for index in range(int(environment["fleet_by_class"][kind]))]
    isolation_epochs = (
        list(isolation_epochs_state)
        if isolation_epochs_state is not None and isolation_epochs_state
        else [0 for _ in vehicles]
    )
    if len(isolation_epochs) != len(vehicles):
        raise ValueError("isolation epoch state must match the fleet size")
    beliefs = [BetaBelief.from_reliability(prior, prior_strength, belief_config["discount"], belief_config["corridor_pooling"]) for _ in vehicles]
    if belief_evidence is not None and belief_evidence and not environment.get("perfect_information", False):
        if len(belief_evidence) != len(beliefs):
            raise ValueError("belief evidence must match the fleet size")
        for belief, (evidence_a, evidence_b) in zip(beliefs, belief_evidence):
            belief.a = np.maximum(belief.a0 + evidence_a, belief.a0)
            belief.b = np.maximum(belief.b0 + evidence_b, belief.b0)
    planner_belief = BetaBelief.from_reliability(
        prior,
        prior_strength,
        belief_config["discount"],
        belief_config["corridor_pooling"],
    )
    if planner_evidence is not None and planner_evidence:
        evidence_a, evidence_b = planner_evidence[0]
        planner_belief.a = np.maximum(planner_belief.a0 + evidence_a, planner_belief.a0)
        planner_belief.b = np.maximum(planner_belief.b0 + evidence_b, planner_belief.b0)
    coordinator_belief = BetaBelief.from_reliability(
        prior,
        prior_strength,
        belief_config["discount"],
        belief_config["corridor_pooling"],
    )
    pending_observations = [0 for _ in vehicles]
    incorporated_observations = 0
    if architecture_state is not None:
        if "coordinator_evidence" in architecture_state:
            evidence_a, evidence_b = architecture_state["coordinator_evidence"]
            coordinator_belief.a = np.maximum(
                coordinator_belief.a0 + evidence_a, coordinator_belief.a0
            )
            coordinator_belief.b = np.maximum(
                coordinator_belief.b0 + evidence_b, coordinator_belief.b0
            )
        pending_observations = list(
            architecture_state.get("pending_observations", pending_observations)
        )
        incorporated_observations = int(
            architecture_state.get("incorporated_observations", 0)
        )
    if posterior_audit is not None:
        fleet_mean = np.mean([belief.mean for belief in beliefs], axis=0)
        true_open = true_reliability >= environment["phi_min"]
        believed_open = fleet_mean >= environment["phi_min"]
        posterior_audit["mismatch_indices"] = set(np.flatnonzero(true_open != believed_open).tolist())
        posterior_audit["fleet_mean"] = fleet_mean.copy()
        posterior_audit["true_open"] = true_open.copy()
    reveal_edges = environment.get("day12_reveal_edges", set()) if day == 12 else set()
    if reveal_edges:
        reveal_strength = 1.0e9
        for edge_id in reveal_edges:
            index = edge_lookup[int(edge_id)]
            passed = bool(true_reliability[index] >= environment["phi_min"])
            for belief in beliefs:
                belief.a[index] = reveal_strength if passed else 1.0
                belief.b[index] = 1.0 if passed else reveal_strength
    arrivals = patients.sort_values("onset_minute").to_dict("records")
    pending: list[dict[str, Any]] = []
    assigned: set[str] = set()
    trace = []
    total = 0.0
    survival_by_triage = {1: 0.0, 2: 0.0, 3: 0.0}
    duplicate_assignments = 0
    merge_divergences = []
    route_set_sizes: list[int] = []
    rng = np.random.default_rng(seed)
    cursor = 0
    now = 0.0
    while cursor < len(arrivals) or pending or any(v["available"] > now for v in vehicles):
        future = []
        if cursor < len(arrivals):
            future.append(float(arrivals[cursor]["onset_minute"]))
        future.extend(float(v["available"]) for v in vehicles if v["available"] > now)
        if not future:
            break
        now = max(now, min(future))
        while cursor < len(arrivals) and float(arrivals[cursor]["onset_minute"]) <= now:
            pending.append(arrivals[cursor]); cursor += 1
        pending = [p for p in pending if str(p["patient_id"]) not in assigned]
        idle = [i for i, v in enumerate(vehicles) if float(v["available"]) <= now]
        if not idle or not pending:
            continue
        coordinator_connected: set[int] = set()
        if belief_architecture == "centralized_coordinator":
            coordinator_connected = {
                index
                for index in range(len(vehicles))
                if rng.random() <= communication_reliability
            }
            components = []
            if coordinator_connected:
                components.append(sorted(coordinator_connected))
            components.extend(
                [index]
                for index in range(len(vehicles))
                if index not in coordinator_connected
            )
        else:
            components = _components(len(vehicles), communication_reliability, rng)
        for component in components:
            if len(component) > 1:
                for vehicle_index in component:
                    isolation_epochs[vehicle_index] = 0
            else:
                isolation_epochs[component[0]] += 1
        if communication_audit is not None:
            communication_audit["communication_epochs"] = (
                communication_audit.get("communication_epochs", 0) + 1
            )
            communication_audit["partitioned_epochs"] = (
                communication_audit.get("partitioned_epochs", 0)
                + int(len(components) > 1)
            )
            communication_audit["component_count_sum"] = (
                communication_audit.get("component_count_sum", 0) + len(components)
            )
            communication_audit["maximum_component_count"] = max(
                communication_audit.get("maximum_component_count", 1),
                len(components),
            )
        if belief_architecture == "centralized_coordinator":
            if coordinator_connected:
                combined = [coordinator_belief] + [
                    beliefs[index] for index in sorted(coordinator_connected)
                ]
                before = float(
                    np.mean(
                        [
                            np.mean(np.abs(item.mean - coordinator_belief.mean))
                            for item in combined[1:]
                        ]
                    )
                )
                _merge_component(combined, list(range(len(combined))))
                merge_divergences.append(before)
                incorporated_observations += sum(
                    pending_observations[index] for index in coordinator_connected
                )
                for index in coordinator_connected:
                    pending_observations[index] = 0
        elif belief_architecture == "decentralized_merge":
            for component in components:
                before = np.mean(
                    [
                        np.mean(
                            np.abs(beliefs[i].mean - beliefs[component[0]].mean)
                        )
                        for i in component
                    ]
                )
                _merge_component(beliefs, component)
                merge_divergences.append(float(before))
        if centralized_stale_view and len(components) == 1:
            planner_belief.a = beliefs[0].a.copy()
            planner_belief.b = beliefs[0].b.copy()
        planning_components = (
            [list(range(len(vehicles)))] if centralized_stale_view else components
        )
        for component in planning_components:
            active = [i for i in component if i in idle]
            if not active:
                continue
            shared_belief = planner_belief if centralized_stale_view else beliefs[component[0]]
            shared_route_graph = build_traversable_graph(
                graph,
                shared_belief.mean,
                edge_lookup,
                environment["phi_min"],
                affected_population,
                network_config,
            )
            shared_route_cache: dict[
                tuple[int, int], tuple[list[int], float, float] | None
            ] = {}
            contexts = {}
            for i in active:
                individual_belief = (
                    beliefs[i]
                    if belief_architecture == "decentralized_no_merge"
                    else shared_belief
                )
                individual_route_graph = (
                    build_traversable_graph(
                        graph,
                        individual_belief.mean,
                        edge_lookup,
                        environment["phi_min"],
                        affected_population,
                        network_config,
                    )
                    if belief_architecture == "decentralized_no_merge"
                    else shared_route_graph
                )
                individual_route_cache = (
                    {}
                    if belief_architecture == "decentralized_no_merge"
                    else shared_route_cache
                )
                contexts[i] = {
                    "vehicle": vehicles[i],
                    "vehicle_index": i,
                    "belief": individual_belief,
                    "candidates": _candidate_set(
                        graph,
                        edge_lookup,
                        hospitals,
                        vehicles[i],
                        individual_belief,
                        pending,
                        now,
                        affected_population,
                        network_config,
                        environment,
                        route_graph=individual_route_graph,
                        route_metrics_cache=individual_route_cache,
                    ),
                    "now": now,
                    "day": day,
                    "isolation_epochs": isolation_epochs[i],
                    "lookahead_features": _lookahead_features(
                        graph,
                        individual_route_graph,
                        vehicles[i],
                        vehicles,
                        pending,
                        individual_belief,
                        edge_lookup,
                        day,
                        now,
                    ),
                }
            if environment.get("exploration_enabled", False):
                for context in contexts.values():
                    counts: dict[str, int] = {}
                    for candidate in context["candidates"]:
                        if candidate.patient_id is not None:
                            counts[candidate.patient_id] = counts.get(candidate.patient_id, 0) + 1
                    route_set_sizes.extend(counts.values())
                route_count: dict[int, int] = {}
                total_routes = 0
                for context in contexts.values():
                    for candidate in context["candidates"]:
                        if candidate.patient_id is None:
                            continue
                        total_routes += 1
                        for edge_id in candidate.inbound[0]:
                            route_count[edge_id] = route_count.get(edge_id, 0) + 1
                if total_routes:
                    for context in contexts.values():
                        belief = context["belief"]
                        for candidate in context["candidates"]:
                            if candidate.patient_id is None:
                                continue
                            candidate.information_value = sum(
                                _beta_information_gain(float(belief.a[edge_lookup[edge_id]]), float(belief.b[edge_lookup[edge_id]]))
                                * route_count.get(edge_id, 0) / total_routes
                                for edge_id in candidate.inbound[0]
                            )
                            if alternative_route_audit is not None and candidate.route_rank > 0:
                                alternative_route_audit.update(int(edge_id) for edge_id in candidate.inbound[0])
            commits, rounds = _resolve(active, contexts, policies)
            for i, candidate in commits.items():
                if candidate.patient_id is None:
                    policies[i].observe(0.0)
                    continue
                duplicate = candidate.patient_id in assigned
                duplicate_assignments += int(duplicate)
                reward = 0.0
                traversed = []
                elapsed = 0.0
                passed = True
                for edge_id in [*candidate.inbound[0], *candidate.outbound[0]]:
                    index = edge_lookup[edge_id]
                    ok = bool(true_reliability[index] >= environment["phi_min"])
                    beliefs[i].update(index, ok)
                    traversed.append(edge_id)
                    if not ok:
                        elapsed += environment["blocked_encounter_minutes"]
                        passed = False
                        break
                observation_count = len(traversed)
                if belief_architecture == "centralized_coordinator":
                    if i in coordinator_connected:
                        combined = [coordinator_belief, beliefs[i]]
                        _merge_component(combined, [0, 1])
                        incorporated_observations += observation_count
                    else:
                        pending_observations[i] += observation_count
                else:
                    incorporated_observations += observation_count
                if passed:
                    elapsed += candidate.inbound[1] + candidate.outbound[1]
                    vehicles[i]["node"] = candidate.destination
                    if not duplicate:
                        reward = candidate.immediate_survival
                        assigned.add(candidate.patient_id)
                        total += reward
                        survival_by_triage[int(candidate.patient["triage"])] += reward
                vehicles[i]["available"] = now + elapsed + environment["service_minutes"]
                policies[i].observe(float(reward))
                trace.append({"epoch": now, "vehicle": vehicles[i]["vehicle_id"], "vehicle_class": vehicles[i]["kind"], "patient_id": candidate.patient_id, "triage": int(candidate.patient["triage"]), "duplicate": duplicate, "reward": reward, "rounds": rounds, "arcs": traversed, "route_rank": candidate.route_rank, "information_value": candidate.information_value, "exploration_cost": candidate.exploration_cost, "detour_distance_km": candidate.detour_distance_km})
        if not any(str(p["patient_id"]) not in assigned for p in pending) and cursor >= len(arrivals):
            break
    if mark_terminal:
        for policy in policies:
            if hasattr(policy, "transitions") and policy.transitions:
                policy.transitions[-1]["terminal"] = True
    if belief_evidence is not None and not environment.get("perfect_information", False):
        belief_evidence.clear()
        belief_evidence.extend(
            (belief.a - belief.a0, belief.b - belief.b0) for belief in beliefs
        )
    if planner_evidence is not None and not environment.get("perfect_information", False):
        planner_evidence.clear()
        planner_evidence.append(
            (
                planner_belief.a - planner_belief.a0,
                planner_belief.b - planner_belief.b0,
            )
        )
    if architecture_state is not None:
        architecture_state.clear()
        architecture_state.update(
            {
                "coordinator_evidence": (
                    coordinator_belief.a - coordinator_belief.a0,
                    coordinator_belief.b - coordinator_belief.b0,
                ),
                "pending_observations": pending_observations,
                "incorporated_observations": incorporated_observations,
            }
        )
    if architecture_audit is not None:
        true_state = (true_reliability >= environment["phi_min"]).astype(float)
        if belief_architecture == "centralized_coordinator":
            belief_error = float(np.mean(np.abs(coordinator_belief.mean - true_state)))
        else:
            belief_error = float(
                np.mean([np.mean(np.abs(belief.mean - true_state)) for belief in beliefs])
            )
        architecture_audit.update(
            {
                "mean_belief_error": belief_error,
                "observations_incorporated": incorporated_observations,
                "observations_pending": int(sum(pending_observations)),
            }
        )
    if isolation_epochs_state is not None:
        isolation_epochs_state.clear()
        isolation_epochs_state.extend(isolation_epochs)
    decisions = max(len(trace), 1)
    return {"survival_total": total, "survival_type_1": survival_by_triage[1], "survival_type_2": survival_by_triage[2], "survival_type_3": survival_by_triage[3], "patients": len(patients), "served": len(assigned), "duplicate_assignment_rate": duplicate_assignments / decisions, "mean_belief_divergence_at_reconnection": float(np.mean(merge_divergences)) if merge_divergences else 0.0, "route_set_observations": len(route_set_sizes), "route_set_mean": float(np.mean(route_set_sizes)) if route_set_sizes else 1.0, "route_set_median": float(np.median(route_set_sizes)) if route_set_sizes else 1.0, "route_set_max": int(max(route_set_sizes)) if route_set_sizes else 1, "route_set_share_multiple": float(np.mean(np.asarray(route_set_sizes) > 1)) if route_set_sizes else 0.0}, trace
