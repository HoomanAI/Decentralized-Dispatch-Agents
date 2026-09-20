"""Generate the calibrated planar fallback network with explicit geometry."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from shapely.geometry import LineString, Point, mapping
from shapely.ops import unary_union

from code.data.loaders import DAY_COLUMNS, ROUTABLE_CLASSES
from code.reliability.estimator import ARTERIAL_CLASSES

LOGGER = logging.getLogger(__name__)
LOCAL_CLASSES = ("residential", "service", "unclassified", "living_street")
NONROUTABLE_CLASSES = ("path", "footway", "steps", "bridleway", "cycleway")


def _candidate_canyon_backbone_graph(
    config: dict[str, Any]
) -> tuple[nx.Graph, np.ndarray]:
    """Build a small planar canyon graph with a concentrated central spine."""
    node_count = int(config["candidate_node_count"])
    edge_count = int(config["routable_edge_count"])
    spine_count = int(config.get("backbone_node_count", 45))
    branch_count = int(config.get("branch_count", 62))
    if node_count <= spine_count or branch_count <= 0:
        raise ValueError("Reduced canyon graph needs spine and branch nodes")
    bbox = config["bbox"]
    spine_x = np.linspace(bbox["west"] + 0.004, bbox["east"] - 0.004, spine_count)
    spine_y = np.linspace(bbox["south"] + 0.018, bbox["north"] - 0.018, spine_count)
    spine_y += 0.003 * np.sin(np.linspace(0.0, 3.0 * np.pi, spine_count))
    points: list[tuple[float, float]] = list(zip(spine_x, spine_y))
    graph = nx.path_graph(spine_count)
    remaining_nodes = node_count - spine_count
    branch_lengths = [remaining_nodes // branch_count] * branch_count
    for index in range(remaining_nodes % branch_count):
        branch_lengths[index] += 1
    anchors = list(
        np.linspace(0, max(2, spine_count // 6), branch_count // 2, dtype=int)
    ) + list(
        np.linspace(
            spine_count - 1 - max(2, spine_count // 6),
            spine_count - 1,
            branch_count - branch_count // 2,
            dtype=int,
        )
    )
    next_node = spine_count
    branch_records = []
    for branch_index, (anchor, length) in enumerate(zip(anchors, branch_lengths)):
        anchor = int(anchor)
        branch_nodes = []
        side = -1.0 if branch_index % 2 == 0 else 1.0
        lane = 1.0 + (branch_index % 5) * 0.16
        previous = anchor
        for offset in range(1, length + 1):
            node = next_node
            next_node += 1
            radial = 0.0016 * offset * lane
            x = spine_x[anchor] + side * radial * 0.42
            y = spine_y[anchor] + side * radial
            x = float(np.clip(x, bbox["west"], bbox["east"]))
            y = float(np.clip(y, bbox["south"], bbox["north"]))
            points.append((x, y))
            graph.add_edge(previous, node)
            branch_nodes.append(node)
            previous = node
        branch_records.append((anchor, branch_nodes))
    for anchor, branch_nodes in branch_records:
        if graph.number_of_edges() >= edge_count:
            break
        if len(branch_nodes) >= 2:
            graph.add_edge(anchor, branch_nodes[1])
    for start, stop in ((0, spine_count // 6), (5 * spine_count // 6, spine_count)):
        for node in range(start, max(start, stop - 2)):
            if graph.number_of_edges() >= edge_count:
                break
            graph.add_edge(node, node + 2)
    if graph.number_of_edges() != edge_count:
        raise RuntimeError(
            f"Reduced canyon construction produced {graph.number_of_edges()} edges"
        )
    point_array = np.asarray(points, dtype=float)
    for node, (longitude, latitude) in enumerate(point_array):
        graph.nodes[node]["x"] = float(longitude)
        graph.nodes[node]["y"] = float(latitude)
    for edge_id, (u, v) in enumerate(sorted(graph.edges())):
        geometry = LineString([point_array[u], point_array[v]])
        graph.edges[u, v].update(
            edge_id=edge_id,
            length=float(geometry.length),
            geometry_wkt=geometry.wkt,
        )
    return graph, point_array


def _candidate_planar_graph(config: dict[str, Any]) -> tuple[nx.Graph, np.ndarray]:
    """Build a connected planar Delaunay subgraph with the requested edge count."""
    rng = np.random.default_rng(config["seed"])
    bbox = config["bbox"]
    count = config["candidate_node_count"]
    coastal_count = int(float(config.get("coastal_node_fraction", 0.68)) * count)
    canyon_count = count - coastal_count
    coastal_x = rng.uniform(bbox["west"], bbox["east"], coastal_count)
    coastal_y = bbox["south"] + (bbox["north"] - bbox["south"]) * rng.beta(
        1.8, 4.5, coastal_count
    )
    canyon_center_count = int(config.get("canyon_center_count", 5))
    canyon_centers = np.linspace(
        bbox["west"], bbox["east"], canyon_center_count + 2
    )[1:-1]
    canyon_y = rng.uniform(bbox["south"], bbox["north"], canyon_count)
    canyon_x = rng.choice(canyon_centers, canyon_count) + rng.normal(
        0.0, float(config.get("canyon_noise_degrees", 0.0045)), canyon_count
    )
    canyon_x = np.clip(canyon_x, bbox["west"], bbox["east"])
    points = np.column_stack(
        [
            np.concatenate([coastal_x, canyon_x]),
            np.concatenate([coastal_y, canyon_y]),
        ]
    )
    triangulation = Delaunay(points)
    candidates: set[tuple[int, int]] = set()
    for triangle in triangulation.simplices:
        for first, second in ((0, 1), (1, 2), (0, 2)):
            u, v = sorted((int(triangle[first]), int(triangle[second])))
            candidates.add((u, v))
    weighted = [
        (u, v, float(np.linalg.norm(points[u] - points[v]))) for u, v in candidates
    ]
    complete = nx.Graph()
    complete.add_weighted_edges_from(weighted, weight="length")
    tree = nx.minimum_spanning_tree(complete, weight="length")
    selected = {tuple(sorted(edge)) for edge in tree.edges()}
    tree_degree = dict(tree.degree())
    if config.get("local_cycle_bias", False):
        tree_distance = dict(nx.all_pairs_shortest_path_length(tree))
        remaining = sorted(
            (edge for edge in weighted if (edge[0], edge[1]) not in selected),
            key=lambda edge: (
                tree_degree[edge[0]] == 1 or tree_degree[edge[1]] == 1,
                tree_distance[edge[0]][edge[1]],
                edge[2],
            ),
        )
    else:
        remaining = sorted(
            (edge for edge in weighted if (edge[0], edge[1]) not in selected),
            key=lambda edge: (
                tree_degree[edge[0]] == 1 or tree_degree[edge[1]] == 1,
                edge[2],
            ),
        )
    target = config["routable_edge_count"]
    if len(weighted) < target:
        raise ValueError(f"Delaunay graph has only {len(weighted)} candidate edges")
    for u, v, _ in remaining:
        if len(selected) >= target:
            break
        selected.add((u, v))
    graph = nx.Graph()
    for node, (longitude, latitude) in enumerate(points):
        graph.add_node(node, x=float(longitude), y=float(latitude))
    for edge_id, (u, v) in enumerate(sorted(selected)):
        geometry = LineString([points[u], points[v]])
        graph.add_edge(
            u,
            v,
            edge_id=edge_id,
            length=float(geometry.length),
            geometry_wkt=geometry.wkt,
        )
    return graph, points


def _assign_classes(graph: nx.Graph, config: dict[str, Any]) -> dict[tuple[int, int], float]:
    """Assign arterial status to structurally central edges and local classes elsewhere."""
    centrality = nx.edge_betweenness_centrality(graph, normalized=True)
    arterial_count = config["arterial_edge_count"]
    seed_edge = max(graph.edges(), key=lambda edge: centrality[edge])
    arterial_edges = {tuple(sorted(seed_edge))}
    spine_nodes = set(seed_edge)
    while len(arterial_edges) < arterial_count:
        frontier = [
            edge
            for edge in graph.edges()
            if tuple(sorted(edge)) not in arterial_edges
            and ((edge[0] in spine_nodes) != (edge[1] in spine_nodes))
        ]
        if not frontier:
            raise RuntimeError("Could not extend the connected arterial spine")
        chosen = max(frontier, key=lambda edge: centrality[edge])
        arterial_edges.add(tuple(sorted(chosen)))
        spine_nodes.update(chosen)
    arterial_cycle = tuple(sorted(ARTERIAL_CLASSES))
    arterial_rank = 0
    local_rank = 0
    for edge in graph.edges():
        if tuple(sorted(edge)) in arterial_edges:
            road_class = arterial_cycle[arterial_rank % len(arterial_cycle)]
            class_group = "arterial"
            arterial_rank += 1
        else:
            road_class = LOCAL_CLASSES[local_rank % len(LOCAL_CLASSES)]
            class_group = "local"
            local_rank += 1
        graph.edges[edge]["fclass"] = road_class
        graph.edges[edge]["class_group"] = class_group
    return centrality


def _spatial_pattern_assignment(
    edges: pd.DataFrame, patterns: pd.DataFrame, seed: int
) -> pd.DataFrame:
    """Assign observed temporal patterns along a reproducible moving fire-front score."""
    rng = np.random.default_rng(seed)
    day_weights = np.array([1.0, 0.55, 0.25, -0.10, -0.20, 0.85])
    pattern_score = patterns.loc[:, DAY_COLUMNS].to_numpy() @ day_weights
    pattern_score += rng.normal(0.0, 0.02, len(patterns))
    spatial_score = edges["mid_x"].to_numpy() + 0.65 * edges["mid_y"].to_numpy()
    assigned = patterns.iloc[np.argsort(pattern_score)].reset_index(drop=True)
    target_order = np.argsort(spatial_score)
    output = pd.DataFrame(index=np.arange(len(edges)), columns=DAY_COLUMNS, dtype=int)
    output.iloc[target_order] = assigned.loc[:, DAY_COLUMNS].to_numpy(dtype=int)
    return output


def _feature_collection(frame: pd.DataFrame) -> dict[str, Any]:
    features = []
    for row in frame.to_dict("records"):
        properties = {
            key: value.item() if isinstance(value, np.generic) else value
            for key, value in row.items()
            if key != "geometry"
        }
        features.append(
            {
                "type": "Feature",
                "geometry": mapping(row["geometry"]),
                "properties": properties,
            }
        )
    return {"type": "FeatureCollection", "features": features}


def _validate_network(
    graph: nx.Graph,
    features: pd.DataFrame,
    config: dict[str, Any],
    centrality: dict[tuple[int, int], float],
) -> dict[str, Any]:
    routable = features.loc[features["routable"]]
    arterial = routable.loc[routable["class_group"] == "arterial"]
    local = routable.loc[routable["class_group"] == "local"]
    assert nx.is_connected(graph)
    assert nx.check_planarity(graph)[0]
    mean_degree = 2.0 * graph.number_of_edges() / graph.number_of_nodes()
    dead_end_share = sum(degree == 1 for _, degree in graph.degree()) / graph.number_of_nodes()
    assert config["minimum_mean_degree"] <= mean_degree <= config["maximum_mean_degree"]
    assert config["minimum_dead_end_share"] <= dead_end_share <= config[
        "maximum_dead_end_share"
    ]
    assert len(features) == config["total_road_feature_count"]
    assert len(routable) == config["routable_edge_count"]
    assert len(arterial) == config["arterial_edge_count"]
    assert len(local) == config["local_edge_count"]
    assert routable.loc[:, DAY_COLUMNS].sum().tolist() == config[
        "target_routable_daily_counts"
    ]
    if "target_routable_daily_fractions" in config:
        actual_fractions = routable.loc[:, DAY_COLUMNS].mean().to_numpy(dtype=float)
        target_fractions = np.asarray(
            config["target_routable_daily_fractions"], dtype=float
        )
        assert np.all(
            np.abs(actual_fractions - target_fractions)
            <= 0.5 / config["routable_edge_count"] + 1.0e-12
        )
    assert arterial.loc[:, DAY_COLUMNS].sum().tolist() == config[
        "target_arterial_daily_counts"
    ]
    assert local.loc[:, DAY_COLUMNS].sum().tolist() == config[
        "target_local_daily_counts"
    ]
    duration_counts = (
        features.loc[:, DAY_COLUMNS]
        .sum(axis=1)
        .value_counts()
        .sort_index()
        .reindex(range(6), fill_value=0)
        .tolist()
    )
    if "target_road_duration_counts" in config:
        assert duration_counts == config["target_road_duration_counts"]
    day_11 = routable["11"].astype(bool)
    day_12 = routable["12"].astype(bool)
    new_relapse = int((day_12 & ~day_11).sum())
    assert routable["7"].mean() > 0.90
    assert routable.loc[:, ["9", "10", "11"]].mean().max() < 0.01
    assert routable["12"].mean() > 0.20
    assert new_relapse / int(day_12.sum()) > 0.90
    arterial_graph = nx.edge_subgraph(
        graph,
        [edge for edge in graph.edges() if graph.edges[edge]["class_group"] == "arterial"],
    )
    arterial_components = nx.number_connected_components(arterial_graph)
    assert arterial_components <= int(config.get("maximum_arterial_components", 3))
    top_count = max(1, int(np.ceil(config["top_betweenness_fraction"] * len(centrality))))
    ranked_betweenness = sorted(centrality.values(), reverse=True)
    betweenness_concentration = sum(ranked_betweenness[:top_count]) / sum(
        ranked_betweenness
    )
    if "minimum_betweenness_concentration" in config:
        assert betweenness_concentration >= config["minimum_betweenness_concentration"]
    if "maximum_betweenness_concentration" in config:
        assert betweenness_concentration <= config["maximum_betweenness_concentration"]
    assert arterial_components <= int(config.get("maximum_arterial_components", 3))
    return {
        "seed": config["seed"],
        "node_count": graph.number_of_nodes(),
        "routable_edge_count": len(routable),
        "total_road_feature_count": len(features),
        "arterial_count": len(arterial),
        "local_count": len(local),
        "connected": True,
        "planar": True,
        "mean_node_degree": mean_degree,
        "dead_end_count": sum(degree == 1 for _, degree in graph.degree()),
        "dead_end_share": dead_end_share,
        "top_5_percent_edge_betweenness_share": betweenness_concentration,
        "arterial_spine_components": arterial_components,
        "routable_daily_counts": routable.loc[:, DAY_COLUMNS].sum().astype(int).tolist(),
        "routable_daily_fractions": routable.loc[:, DAY_COLUMNS].mean().tolist(),
        "arterial_daily_counts": arterial.loc[:, DAY_COLUMNS].sum().astype(int).tolist(),
        "arterial_daily_fractions": arterial.loc[:, DAY_COLUMNS].mean().tolist(),
        "local_daily_counts": local.loc[:, DAY_COLUMNS].sum().astype(int).tolist(),
        "local_daily_fractions": local.loc[:, DAY_COLUMNS].mean().tolist(),
        "road_duration_counts": duration_counts,
        "day_12_affected": int(day_12.sum()),
        "day_12_new_since_day_11": new_relapse,
        "day_12_new_fraction": new_relapse / int(day_12.sum()),
    }


def build_synthetic_network(
    roads: pd.DataFrame, config: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    """Build and persist a calibrated synthetic network and daily fire fronts."""
    if config.get("generator") == "canyon_backbone":
        LOGGER.warning("Using the seeded reduced canyon-backbone generator")
        graph, points = _candidate_canyon_backbone_graph(config)
    else:
        try:
            import osmnx  # noqa: F401
        except ImportError:
            LOGGER.warning(
                "OSMnx is unavailable. Using the seeded Delaunay planar fallback generator"
            )
        else:
            LOGGER.warning(
                "Using the seeded Delaunay generator because the real export has no osm_id "
                "for a reproducible OSM attachment"
            )
        graph, points = _candidate_planar_graph(config)
    centrality = _assign_classes(graph, config)
    edge_rows = []
    for u, v, attributes in graph.edges(data=True):
        geometry = LineString([points[u], points[v]])
        edge_rows.append(
            {
                "feature_id": f"r_{attributes['edge_id']:04d}",
                "u": int(u),
                "v": int(v),
                "routable": True,
                "fclass": attributes["fclass"],
                "class_group": attributes["class_group"],
                "length": attributes["length"],
                "mid_x": geometry.centroid.x,
                "mid_y": geometry.centroid.y,
                "geometry": geometry,
            }
        )
    features = pd.DataFrame(edge_rows)
    names = roads["fclass"].astype("string").str.lower()
    real_routable = roads.loc[names.isin(ROUTABLE_CLASSES)].copy()
    for group, offset in (("arterial", 0), ("local", 1)):
        synthetic_index = features.index[features["class_group"] == group]
        if group == "arterial":
            real_group = real_routable.loc[
                real_routable["fclass"].astype("string").str.lower().isin(ARTERIAL_CLASSES)
            ]
        else:
            real_group = real_routable.loc[
                ~real_routable["fclass"].astype("string").str.lower().isin(ARTERIAL_CLASSES)
            ]
        if config.get("reduced_instance", False):
            group_counts = config[f"target_{group}_daily_counts"]
            group_frame = features.loc[synthetic_index]
            order = group_frame.sort_values(["mid_x", "mid_y"]).index.to_numpy()
            features.loc[synthetic_index, DAY_COLUMNS] = 0
            for day_index, (day, count) in enumerate(zip(DAY_COLUMNS, group_counts)):
                if day == "12":
                    selected = order[-int(count) :] if count else np.array([], dtype=int)
                elif day == "11":
                    selected = order[: int(count)]
                else:
                    shift = min(day_index, max(len(order) - int(count), 0))
                    selected = order[shift : shift + int(count)]
                features.loc[selected, day] = 1
        else:
            assigned = _spatial_pattern_assignment(
                features.loc[synthetic_index], real_group, config["seed"] + offset
            )
            features.loc[synthetic_index, DAY_COLUMNS] = assigned.to_numpy(dtype=int)

    nonroutable_target = config["total_road_feature_count"] - config["routable_edge_count"]
    nonroutable_real = roads.loc[~names.isin(ROUTABLE_CLASSES)].reset_index(drop=True)
    if config.get("reduced_instance", False):
        nonroutable_real = nonroutable_real.iloc[:nonroutable_target]
    rng = np.random.default_rng(config["seed"] + 2)
    bbox = config["bbox"]
    span_x = bbox["east"] - bbox["west"]
    span_y = bbox["north"] - bbox["south"]
    nonroutable_rows = []
    for index, real_row in nonroutable_real.iterrows():
        x = rng.uniform(bbox["west"], bbox["east"])
        y = rng.uniform(bbox["south"], bbox["north"])
        angle = rng.uniform(0.0, 2.0 * np.pi)
        length = rng.uniform(0.001, 0.004)
        geometry = LineString(
            [(x, y), (x + length * np.cos(angle), y + length * np.sin(angle))]
        )
        row = {
            "feature_id": f"n_{index:04d}",
            "u": -1,
            "v": -1,
            "routable": False,
            "fclass": NONROUTABLE_CLASSES[index % len(NONROUTABLE_CLASSES)],
            "class_group": "nonroutable",
            "length": float(geometry.length),
            "mid_x": geometry.centroid.x,
            "mid_y": geometry.centroid.y,
            "geometry": geometry,
        }
        row.update({day: int(real_row[day]) for day in DAY_COLUMNS})
        nonroutable_rows.append(row)
    features = pd.concat([features, pd.DataFrame(nonroutable_rows)], ignore_index=True)
    features.loc[:, DAY_COLUMNS] = features.loc[:, DAY_COLUMNS].astype(int)
    validation = _validate_network(graph, features, config, centrality)

    output_dir.mkdir(parents=True, exist_ok=True)
    roads_path = output_dir / "synthetic_road_features.geojson"
    roads_path.write_text(
        json.dumps(_feature_collection(features.drop(columns=["mid_x", "mid_y"]))),
        encoding="utf-8",
    )
    node_features = [
        {
            "type": "Feature",
            "geometry": mapping(Point(data["x"], data["y"])),
            "properties": {"node_id": int(node)},
        }
        for node, data in graph.nodes(data=True)
    ]
    (output_dir / "synthetic_network_nodes.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": node_features}),
        encoding="utf-8",
    )
    front_features = []
    for day in DAY_COLUMNS:
        points_for_day = [
            geometry.centroid
            for geometry in features.loc[features[day] == 1, "geometry"]
        ]
        front = unary_union(points_for_day).convex_hull
        front_features.append(
            {
                "type": "Feature",
                "geometry": mapping(front),
                "properties": {"day": int(day), "synthetic_input": True},
            }
        )
    (output_dir / "synthetic_fire_fronts.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": front_features}),
        encoding="utf-8",
    )
    validation_path = output_dir / "synthetic_network_validation.json"
    validation_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    LOGGER.warning("Generated synthetic network input with seed %d", config["seed"])
    return validation
