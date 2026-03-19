"""Config loading helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from pathlib import Path


def load_experiment_config(path: Path) -> dict[str, Any]:
    """Load and minimally validate an experiment YAML config."""
    if not path.exists():
        msg = f"Config file does not exist: {path}"
        raise FileNotFoundError(msg)

    raw = path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        msg = "Experiment config root must be a mapping."
        raise TypeError(msg)

    required = ("dataset", "model", "evaluation")
    missing = [key for key in required if key not in data]
    if missing:
        msg = f"Missing required config sections: {missing}"
        raise ValueError(msg)
    return data
