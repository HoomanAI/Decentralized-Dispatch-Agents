from __future__ import annotations

import json
from pathlib import Path

import yaml

from code.data.loaders import load_all_inputs
from code.network.synthetic import build_synthetic_network

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_synthetic_network_matches_all_calibration_targets(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PROJECT_ROOT / "code" / "config" / "default.yaml").read_text(
            encoding="utf-8"
        )
    )
    bundle = load_all_inputs(PROJECT_ROOT.parent / "Data")
    validation = build_synthetic_network(bundle.roads, config["network"], tmp_path)
    assert validation["connected"]
    assert validation["planar"]
    assert 2.3 <= validation["mean_node_degree"] <= 2.8
    assert 0.15 <= validation["dead_end_share"] <= 0.35
    assert validation["arterial_spine_components"] <= 3
    assert validation["routable_edge_count"] == 1297
    assert validation["arterial_count"] == 138
    assert validation["local_count"] == 1159
    assert validation["road_duration_counts"] == [149, 930, 587, 126, 5, 2]
    assert validation["day_12_new_fraction"] > 0.90
    saved = json.loads(
        (tmp_path / "synthetic_network_validation.json").read_text(encoding="utf-8")
    )
    assert saved == validation


def test_reduced_network_preserves_structural_targets(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PROJECT_ROOT / "code" / "config" / "default.yaml").read_text(
            encoding="utf-8"
        )
    )
    bundle = load_all_inputs(PROJECT_ROOT.parent / "Data")
    validation = build_synthetic_network(bundle.roads, config["reduced_network"], tmp_path)
    assert validation["node_count"] == 283
    assert validation["routable_edge_count"] == 350
    assert validation["arterial_count"] == 37
    assert validation["local_count"] == 313
    assert 2.3 <= validation["mean_node_degree"] <= 2.8
    assert 0.15 <= validation["dead_end_share"] <= 0.35
    assert 0.40 <= validation["top_5_percent_edge_betweenness_share"] <= 0.50
    assert validation["arterial_spine_components"] == 1
    assert validation["routable_daily_counts"] == [317, 97, 2, 1, 1, 72]
    assert validation["day_12_new_fraction"] == 1.0
