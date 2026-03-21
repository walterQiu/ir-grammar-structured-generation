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
    return _normalize_config(data)


def _normalize_config(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize config for backward-compatible model settings."""
    model_cfg = data.get("model", {})
    if not isinstance(model_cfg, dict):
        return data

    if model_cfg.get("name") != "ir_pipeline":
        return data

    stage_defaults = _extract_stage_defaults(model_cfg)
    if "extraction_model" not in model_cfg:
        model_cfg["extraction_model"] = dict(stage_defaults)
    if "ir_model" not in model_cfg:
        model_cfg["ir_model"] = dict(stage_defaults)
    return data


def _extract_stage_defaults(model_cfg: dict[str, Any]) -> dict[str, Any]:
    """Extract shared stage options from legacy flat IR model config."""
    keys = (
        "backend",
        "llm_name",
        "api_key_env",
        "dotenv_path",
        "temperature",
        "timeout_seconds",
    )
    defaults = {key: model_cfg[key] for key in keys if key in model_cfg}
    if "backend" not in defaults:
        defaults["backend"] = "mock"
    if "temperature" not in defaults:
        defaults["temperature"] = 0.0
    return defaults
