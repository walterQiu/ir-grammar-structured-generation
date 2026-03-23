"""Utilities for parsing model text output into JSON objects."""

from __future__ import annotations

import json
from json import JSONDecodeError
from typing import Any


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Parse JSON object from raw text, with a small tolerant fallback."""
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except JSONDecodeError:
        pass

    candidate = _extract_braced_object(stripped)
    if candidate is None:
        return None

    try:
        parsed = json.loads(candidate)
    except JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _extract_braced_object(text: str) -> str | None:
    """Extract the first balanced {...} block from text."""
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            parsed, end = decoder.raw_decode(text[idx:])
        except JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return text[idx : idx + end]
    return None
