"""Environment variable helpers."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

_MIN_QUOTE_LEN = 2


def load_dotenv_file(path: Path) -> None:
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


def get_required_env(key: str, fallback_paths: Iterable[Path] | None = None) -> str:
    """Return required env var after trying optional dotenv files."""
    value = os.getenv(key)
    if value:
        return value

    for path in fallback_paths or []:
        load_dotenv_file(path)

    value = os.getenv(key)
    if value:
        return value

    msg = f"Missing required environment variable: {key}"
    raise RuntimeError(msg)


def _strip_quotes(value: str) -> str:
    """Strip one layer of surrounding single/double quotes."""
    if (
        len(value) >= _MIN_QUOTE_LEN
        and value[0] == value[-1]
        and value[0] in {"'", '"'}
    ):
        return value[1:-1]
    return value
