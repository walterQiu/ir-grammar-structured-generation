"""Shared helpers for RAMS analysis scripts."""

from __future__ import annotations

import re
from pathlib import Path

DATA_DIR = Path("datasets/RAMS/data")
SPLITS = ("train", "dev", "test")
_ROLE_PREFIX_PATTERN = re.compile(r"^evt\d+arg\d+")


def normalize_role_name(raw_role: str) -> str:
    """Normalize RAMS role label by removing evt/arg prefix."""
    return _ROLE_PREFIX_PATTERN.sub("", raw_role).strip()
