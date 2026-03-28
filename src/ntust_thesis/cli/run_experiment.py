"""CLI for running experiments."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from ntust_thesis.core.bootstrap import register_builtin_components
from ntust_thesis.core.config import load_experiment_config
from ntust_thesis.core.pipeline import ExperimentPipeline
from ntust_thesis.utils.artifacts import write_json, write_jsonl, write_yaml


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser."""
    parser = argparse.ArgumentParser(description="Run strict benchmark experiment")
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to experiment yaml"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/runs"),
        help="Directory for run artifacts",
    )
    return parser


def main() -> None:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    register_builtin_components()
    pipeline = ExperimentPipeline(config=config)
    result = pipeline.run()

    experiment_name = config.experiment_name
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_{experiment_name}"
    run_dir = args.output_root / run_id

    write_jsonl(run_dir / "predictions.jsonl", result.rows)
    write_json(run_dir / "metrics.json", result.metrics)
    write_json(
        run_dir / "failed_samples.json", {"failed_samples": result.failed_samples}
    )
    write_yaml(run_dir / "config_snapshot.yaml", config.model_dump())

    failed_sample_ids = [item["sample_id"] for item in result.failed_samples]
    payload = {
        "rows": len(result.rows),
        "failed_count": len(result.failed_samples),
        "failed_sample_ids": failed_sample_ids,
        "metrics": result.metrics,
    }
    sys.stdout.write(f"{json.dumps(payload, ensure_ascii=False)}\n")


if __name__ == "__main__":
    main()
