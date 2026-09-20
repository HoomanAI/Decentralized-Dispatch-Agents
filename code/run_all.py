"""Single command entry point for reproducible study artifacts."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml

from code.data.loaders import load_all_inputs, validate_release
from code.network.synthetic import build_synthetic_network
from code.eval.stage4 import run_stage4
from code.eval.stage5 import run_stage5
from code.eval.stage6 import run_stage6
from code.reliability.estimator import write_scenario_fallback

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_loaders(config: dict[str, object]) -> Path:
    """Run Stage 1, validate the release, and save cleaned inputs and an audit."""
    paths = config["paths"]
    raw_dir = (PROJECT_ROOT / paths["raw_data"]).resolve()
    output_dir = PROJECT_ROOT / paths["processed_data"]
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_all_inputs(raw_dir)
    mismatches = validate_release(bundle, config["validation"])
    if mismatches:
        raise RuntimeError("Release validation failed:\n" + "\n".join(mismatches))

    bundle.roads.to_csv(output_dir / "roads_daily_flags_clean.csv", index=False)
    bundle.population.to_csv(output_dir / "population_blocks_clean.csv", index=False)
    bundle.landcover_weights.to_csv(output_dir / "landcover_weights_clean.csv", index=False)
    bundle.affected_population.to_csv(
        output_dir / "daily_affected_population_clean.csv", index=False
    )
    audit_path = output_dir / "input_audit.json"
    audit_path.write_text(json.dumps(bundle.audit, indent=2), encoding="utf-8")
    LOGGER.info("Stage 1 validation passed and wrote %s", audit_path)
    return audit_path


def run_reliability(config: dict[str, object]) -> tuple[Path, Path]:
    """Run Stage 2 only when an independent closure observation stream exists."""
    paths = config["paths"]
    raw_dir = (PROJECT_ROOT / paths["raw_data"]).resolve()
    bundle = load_all_inputs(raw_dir)
    mismatches = validate_release(bundle, config["validation"])
    if mismatches:
        raise RuntimeError("Release validation failed:\n" + "\n".join(mismatches))
    return write_scenario_fallback(
        bundle.roads,
        config["reliability"],
        raw_dir,
        PROJECT_ROOT / "results",
    )


def run_network(config: dict[str, object]) -> Path:
    """Run Stage 3 and save the synthetic network validation record."""
    paths = config["paths"]
    raw_dir = (PROJECT_ROOT / paths["raw_data"]).resolve()
    bundle = load_all_inputs(raw_dir)
    mismatches = validate_release(bundle, config["validation"])
    if mismatches:
        raise RuntimeError("Release validation failed:\n" + "\n".join(mismatches))
    build_synthetic_network(bundle.roads, config["network"], PROJECT_ROOT / "data")
    return PROJECT_ROOT / "data" / "synthetic_network_validation.json"


def run_environment(config: dict[str, object]) -> tuple[Path, Path, Path, Path, Path]:
    """Run Stage 4 for both myopic information assumptions."""
    paths = config["paths"]
    raw_dir = (PROJECT_ROOT / paths["raw_data"]).resolve()
    bundle = load_all_inputs(raw_dir)
    return run_stage4(
        bundle.affected_population,
        config,
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "results",
    )


def run_bound(config: dict[str, object]) -> tuple[Path, Path]:
    """Run Stage 5 optimistic and travel-capacity bounds."""
    paths = config["paths"]
    raw_dir = (PROJECT_ROOT / paths["raw_data"]).resolve()
    bundle = load_all_inputs(raw_dir)
    return run_stage5(
        bundle.affected_population,
        config,
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "results",
    )


def run_policy(config: dict[str, object]) -> tuple[Path, Path, Path]:
    """Run Stage 6 exploitation-only policy training and held-out evaluation."""
    paths = config["paths"]
    raw_dir = (PROJECT_ROOT / paths["raw_data"]).resolve()
    bundle = load_all_inputs(raw_dir)
    return run_stage6(
        bundle.affected_population,
        config,
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "results",
    )


def main() -> None:
    """Parse the requested stage and run it from the default configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("loaders", "reliability", "network", "environment", "bound", "policy"),
        default="loaders",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config_path = PROJECT_ROOT / "code" / "config" / "default.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.stage == "loaders":
        run_loaders(config)
    elif args.stage == "reliability":
        status_path, grid_path = run_reliability(config)
        LOGGER.info("Stage 2 wrote %s and %s", status_path, grid_path)
    elif args.stage == "network":
        validation_path = run_network(config)
        LOGGER.info("Stage 3 wrote %s", validation_path)
    elif args.stage == "environment":
        summary_path, trace_path, sweep_path, grid_path, comparison_path = run_environment(config)
        LOGGER.info(
            "Stage 4 wrote %s, %s, %s, %s, and %s",
            summary_path,
            trace_path,
            sweep_path,
            grid_path,
            comparison_path,
        )
    elif args.stage == "bound":
        summary_path, assignment_path = run_bound(config)
        LOGGER.info("Stage 5 wrote %s and %s", summary_path, assignment_path)
    elif args.stage == "policy":
        training_path, evaluation_path, checkpoint_path = run_policy(config)
        LOGGER.info(
            "Stage 6 wrote %s, %s, and %s",
            training_path,
            evaluation_path,
            checkpoint_path,
        )


if __name__ == "__main__":
    main()
