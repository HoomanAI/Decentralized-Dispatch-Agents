from __future__ import annotations

from pathlib import Path

import pytest

from code.data.loaders import DAY_COLUMNS, load_all_inputs

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA = PROJECT_ROOT.parent / "Data"


@pytest.fixture(scope="module")
def bundle():
    return load_all_inputs(RAW_DATA)


def test_road_defects_are_handled(bundle) -> None:
    assert len(bundle.roads) == 1799
    assert bundle.audit["roads"]["waterway_class_counts"] == {
        "stream": 114,
        "drain": 9,
    }
    assert not any(name.startswith("Shape_Length_") for name in bundle.roads.columns)
    assert tuple(column for column in DAY_COLUMNS if column in bundle.roads) == DAY_COLUMNS
    assert bundle.requires_synthetic_network
    audit = bundle.audit["roads"]
    assert audit["dropped_non_road_poi"]["fclass_values"] == {
        "christian_methodist": 1
    }
    assert audit["routable_feature_count"] == 1297
    assert audit["missing_shape_length_after_cleaning"] == 0
    assert audit["road_total_shape_length_native_units"] == pytest.approx(4.4292, abs=0.00005)


def test_release_sanity_numbers(bundle) -> None:
    population = bundle.audit["population"]
    assert population["total_population"] == 47831
    assert population["housing_units"] == 21138
    assert population["population_65_plus"] == 12276
    assert population["population_65_plus_share"] == pytest.approx(0.257, abs=0.0005)
    assert population["block_age_65_share_min"] == pytest.approx(0.195, abs=0.0005)
    assert population["block_age_65_share_max"] == pytest.approx(0.338, abs=0.0005)
    assert bundle.audit["roads"]["exposure_duration_counts_all_features"] == {
        "0": 159,
        "1": 995,
        "2": 621,
        "3": 138,
        "4": 6,
        "5": 4,
    }
    assert bundle.audit["roads"]["exposure_duration_counts_roads_only"] == {
        "0": 149,
        "1": 930,
        "2": 587,
        "3": 126,
        "4": 5,
        "5": 2,
    }


def test_offset_header_and_union_total(bundle) -> None:
    assert len(bundle.landcover_weights) == 15
    pds = bundle.affected_population.query("region == 'PDS'").sort_values("day")
    assert pds["affected_population"].tolist() == pytest.approx(
        [1773.7, 6188.4, 2192.8, 79.0, 186.5, 186.5], abs=0.051
    )
    check = bundle.audit["affected_population"]["union_count_checks"]["PDS"]
    assert check["union_total"] < check["sum_across_days"]
