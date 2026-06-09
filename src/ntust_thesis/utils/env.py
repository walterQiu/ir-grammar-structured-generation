"""Environment variable helpers."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_DOTENV_PATH = Path("dotenv/.env")


def load_environment(path: Path = DEFAULT_DOTENV_PATH) -> None:
    """Load project environment variables without overriding the shell."""
    load_dotenv(dotenv_path=path, override=False)


def get_required_env(key: str) -> str:
    """Return a required environment variable."""
    value = os.getenv(key)
    if value:
        return value

    msg = f"Missing required environment variable: {key}"
    raise RuntimeError(msg)


def get_env_float(
    key: str,
    default: float,
) -> float:
    """Read a float environment variable with a default fallback."""
    raw = os.getenv(key)
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
) -> int:
    """Read an integer environment variable with a default fallback."""
    raw = os.getenv(key)
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
) -> list[int]:
    """Read a comma-separated integer list with a default fallback."""
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    values = [part.strip() for part in raw.split(",")]
    try:
        return [int(value) for value in values if value]
    except ValueError as exc:
        msg = f"Environment variable {key} must be comma-separated ints, got: {raw}"
        raise RuntimeError(msg) from exc
