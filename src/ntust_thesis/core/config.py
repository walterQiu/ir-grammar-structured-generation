"""Config loading helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


def load_experiment_config(path: Path) -> dict[str, Any]:
    """Load experiment config.

    The detailed YAML parser will be added in a later step.
    """
    return {"config_path": str(path)}
