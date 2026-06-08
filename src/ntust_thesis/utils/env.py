"""Environment variable helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

_MIN_QUOTE_LEN = 2
DEFAULT_DOTENV_PATH = Path("dotenv/.env")


def get_required_env(key: str, fallback_paths: Iterable[Path] | None = None) -> str:
    """Return required env var after trying optional dotenv files."""
    value = os.getenv(key)
    if value:
        return value

    for path in fallback_paths or []:
        _load_dotenv_file(path)

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
        _load_dotenv_file(path)
    return os.getenv(key)


def _load_dotenv_file(path: Path) -> None:
    """Load key-value pairs from a dotenv file into process environment."""
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = _strip_quotes(value.strip())
        os.environ.setdefault(key, value)


def _strip_quotes(value: str) -> str:
    """Strip one layer of surrounding single/double quotes."""
    if (
        len(value) >= _MIN_QUOTE_LEN
        and value[0] == value[-1]
        and value[0] in {"'", '"'}
    ):
        return value[1:-1]
    return value
