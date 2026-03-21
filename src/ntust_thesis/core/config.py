"""Config loading helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import yaml

from ntust_thesis.core.config_models import ExperimentConfig

if TYPE_CHECKING:
    from pathlib import Path


def load_experiment_config(path: Path) -> ExperimentConfig:
    """Load and validate an experiment YAML config."""
    if not path.exists():
        msg = f"Config file does not exist: {path}"
        raise FileNotFoundError(msg)

    raw = path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        msg = "Experiment config root must be a mapping."
        raise TypeError(msg)
    return ExperimentConfig.model_validate(data)
