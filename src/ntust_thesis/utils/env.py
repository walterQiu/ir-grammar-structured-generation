"""Environment variable helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv

if TYPE_CHECKING:
    from collections.abc import Iterable

DEFAULT_DOTENV_PATH = Path("dotenv/.env")


def get_required_env(key: str, fallback_paths: Iterable[Path] | None = None) -> str:
    """Return required env var after trying optional dotenv files."""
    value = os.getenv(key)
    if value:
        return value

    for path in fallback_paths or []:
        load_dotenv(dotenv_path=path, override=False)

    value = os.getenv(key)
    if value:
        return value

    msg = f"Missing required environment variable: {key}"
    raise RuntimeError(msg)


def get_env_float(
    key: str,
    default: float,
    fallback_paths: Iterable[Path] | None = None,
) -> float:
    """Read float from env with default fallback."""
    raw = _get_env_value(key, fallback_paths)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        msg = f"Environment variable {key} must be float, got: {raw}"
        raise RuntimeError(msg) from exc


def get_env_int(
    key: str,
    default: int,
    fallback_paths: Iterable[Path] | None = None,
) -> int:
    """Read int from env with default fallback."""
    raw = _get_env_value(key, fallback_paths)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        msg = f"Environment variable {key} must be int, got: {raw}"
        raise RuntimeError(msg) from exc


def get_env_int_list(
    key: str,
    default: list[int],
    fallback_paths: Iterable[Path] | None = None,
) -> list[int]:
    """Read comma-separated integer list from env with default fallback."""
    raw = _get_env_value(key, fallback_paths)
    if raw is None or raw.strip() == "":
        return default
    values = [part.strip() for part in raw.split(",")]
    try:
        return [int(value) for value in values if value]
    except ValueError as exc:
        msg = f"Environment variable {key} must be comma-separated ints, got: {raw}"
        raise RuntimeError(msg) from exc


def _get_env_value(
    key: str,
    fallback_paths: Iterable[Path] | None = None,
) -> str | None:
    """Get env value after optionally loading dotenv files."""
    value = os.getenv(key)
    if value:
        return value
    for path in fallback_paths or []:
        load_dotenv(dotenv_path=path, override=False)
    return os.getenv(key)
