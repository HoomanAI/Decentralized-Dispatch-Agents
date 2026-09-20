from __future__ import annotations

from pathlib import Path

import yaml
import networkx as nx
import pandas as pd
from shapely.geometry import Point, shape

import numpy as np

from code.env.dispatch import (
    build_runtime_network,
    class_parameter_by_edge,
    damage_for_day,
    run_episode,
)
from code.env.vehicle_centric import _merge_component
from code.belief.beta import BetaBelief

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_hospitals_are_outside_day_7_fire_front() -> None:
    config = yaml.safe_load(
        (PROJECT_ROOT / "code" / "config" / "default.yaml").read_text(
            encoding="utf-8"
        )
    )
    graph, hospitals, _ = build_runtime_network(
        PROJECT_ROOT / "data" / "synthetic_road_features.geojson",
        PROJECT_ROOT / "data" / "synthetic_fire_fronts.geojson",
        config["network"],
    )
    import json

    fronts = json.loads(
        (PROJECT_ROOT / "data" / "synthetic_fire_fronts.geojson").read_text(
            encoding="utf-8"
        )
    )["features"]
    day_7 = shape(next(item["geometry"] for item in fronts if item["properties"]["day"] == 7))
    assert all(
        not day_7.covers(Point(graph.nodes[node]["x"], graph.nodes[node]["y"]))
        for node in hospitals
    )


def test_alpha_heterogeneity_is_seeded_and_breaks_first_day_tie() -> None:
    config = yaml.safe_load(
        (PROJECT_ROOT / "code" / "config" / "default.yaml").read_text(
            encoding="utf-8"
        )
    )
    graph, _, _ = build_runtime_network(
        PROJECT_ROOT / "data" / "synthetic_road_features.geojson",
        PROJECT_ROOT / "data" / "synthetic_fire_fronts.geojson",
        config["network"],
    )
    homogeneous = damage_for_day(graph, 7, 0.375, 0.5)
    heterogeneous_a = damage_for_day(graph, 7, 0.375, 0.5, 20.0, 202605)
    heterogeneous_b = damage_for_day(graph, 7, 0.375, 0.5, 20.0, 202605)
    assert np.array_equal(heterogeneous_a, heterogeneous_b)
    assert np.unique(homogeneous[homogeneous > 0.0]).size == 1
    assert np.unique(heterogeneous_a[heterogeneous_a > 0.0]).size > 1


def test_homogeneous_class_parameters_are_bit_identical() -> None:
    graph = nx.Graph()
    graph.add_edge(0, 1, edge_id=0, class_group="arterial")
    graph.add_edge(1, 2, edge_id=1, class_group="local")
    graph.graph["edge_day_flags"] = {
        0: {str(day): int(day in {7, 9}) for day in range(7, 13)},
        1: {str(day): int(day in {8, 12}) for day in range(7, 13)},
    }
    scalar = damage_for_day(graph, 12, 0.375, 0.5)
    indexed = damage_for_day(
        graph,
        12,
        {"arterial": 0.375, "local": 0.375},
        {"arterial": 0.5, "local": 0.5},
    )
    assert np.array_equal(scalar, indexed)
    scalar_reliability = (1.0 - scalar) * (1.0 - 0.5 * scalar)
    indexed_upsilon = class_parameter_by_edge(
        graph, {"arterial": 0.5, "local": 0.5}, "upsilon"
    )
    indexed_reliability = (1.0 - indexed) * (1.0 - indexed_upsilon * indexed)
    assert np.array_equal(scalar_reliability, indexed_reliability)


def test_blind_policy_contacts_blockage_and_replans() -> None:
    config = yaml.safe_load(
        (PROJECT_ROOT / "code" / "config" / "default.yaml").read_text(
            encoding="utf-8"
        )
    )
    graph = nx.Graph()
    edge_specs = [
        (0, 1, 0, 1.0, 0),
        (1, 3, 1, 1.0, 1),
        (0, 2, 2, 2.0, 0),
        (2, 3, 3, 2.0, 0),
    ]
    flags = {}
    for u, v, edge_id, free_time, exposed in edge_specs:
        graph.add_edge(
            u,
            v,
            edge_id=edge_id,
            fclass="residential",
            length_km=free_time,
            free_time_min=free_time,
            capacity=1000.0,
        )
        flags[edge_id] = {str(day): int(exposed if day == 7 else 0) for day in range(7, 13)}
    graph.graph["edge_day_flags"] = flags
    edge_lookup = {
        int(data["edge_id"]): index
        for index, (_, _, data) in enumerate(graph.edges(data=True))
    }
    patients = pd.DataFrame(
        [{"patient_id": "p0", "node": 3, "triage": 3, "onset_minute": 0.0}]
    )
    low_environment = dict(config["environment"], scenario_alpha=0.375)
    high_environment = dict(config["environment"], scenario_alpha=0.5)
    low, _ = run_episode(
        graph,
        [0],
        edge_lookup,
        patients,
        7,
        "reliability_blind",
        config["network"],
        config["belief"],
        low_environment,
        0.0,
    )
    high, trace = run_episode(
        graph,
        [0],
        edge_lookup,
        patients,
        7,
        "reliability_blind",
        config["network"],
        config["belief"],
        high_environment,
        0.0,
    )
    assert high["blocked_arc_encounters"] == 1
    assert trace[0]["outcome"] == "served"
    assert high["survival_total"] < low["survival_total"]


def test_conservative_operational_threshold_activates_memory() -> None:
    graph = nx.Graph()
    graph.add_edge(
        0,
        1,
        edge_id=0,
        fclass="residential",
        length_km=1.0,
        free_time_min=1.0,
        capacity=1000.0,
    )
    graph.graph["edge_day_flags"] = {
        0: {str(day): int(day == 7) for day in range(7, 13)}
    }
    damage = damage_for_day(graph, 8, alpha=1.0, gamma=0.75)
    reliability = (1.0 - damage) * (1.0 - 0.5 * damage)
    assert reliability[0] >= 0.5
    assert reliability[0] < 0.7


def test_component_merge_is_idempotent_and_order_independent() -> None:
    beliefs = [BetaBelief.from_reliability(np.array([0.7, 0.8]), 5.0, 0.9, 0.0) for _ in range(3)]
    beliefs[0].a += np.array([1.0, 0.0])
    beliefs[1].a += np.array([0.0, 2.0])
    beliefs[2].b += np.array([3.0, 0.0])
    _merge_component(beliefs, [0, 1, 2])
    expected_a = beliefs[0].a.copy()
    expected_b = beliefs[0].b.copy()
    _merge_component(beliefs, [2, 0, 1])
    for belief in beliefs:
        assert np.allclose(belief.a, expected_a)
        assert np.allclose(belief.b, expected_b)
