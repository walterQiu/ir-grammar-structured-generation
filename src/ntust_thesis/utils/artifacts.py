"""Artifact writing helpers for experiment runs."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from pathlib import Path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON payload to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write JSONL rows to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(f"{json.dumps(row, ensure_ascii=False)}\n")


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    """Write YAML payload to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
