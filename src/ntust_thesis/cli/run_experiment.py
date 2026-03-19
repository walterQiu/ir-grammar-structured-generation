"""CLI for running experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ntust_thesis.core.config import load_experiment_config
from ntust_thesis.core.pipeline import ExperimentPipeline


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser."""
    parser = argparse.ArgumentParser(description="Run strict benchmark experiment")
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to experiment yaml"
    )
    return parser


def main() -> None:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    _ = load_experiment_config(args.config)
    pipeline = ExperimentPipeline()
    result = pipeline.run()

    payload = {"rows": len(result.rows), "metrics": result.metrics}
    sys.stdout.write(f"{json.dumps(payload, ensure_ascii=False)}\n")


if __name__ == "__main__":
    main()
