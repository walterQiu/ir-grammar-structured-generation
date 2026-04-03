"""Helpers for converting between role paths and nested role structures."""

from __future__ import annotations

from typing import Any


def role_path_to_structure(role_path: str) -> str | dict[str, Any]:
    """Convert dot-path role into nested structure."""
    path = role_path.strip()
    if not path:
        return ""
    segments = [segment.strip() for segment in path.split(".") if segment.strip()]
    if not segments:
        return ""
    if len(segments) == 1:
        return segments[0]

    nested: Any = {}
    for segment in reversed(segments[1:]):
        nested = {segment: nested}
    return {segments[0]: nested}


def role_to_path(role: str | dict[str, Any]) -> str | None:
    """Convert role string/nested object into canonical dot-path."""
    if isinstance(role, str):
        normalized = role.strip()
        return normalized or None
    if isinstance(role, dict):
        segments = _extract_segments_from_role_dict(role)
        if not segments:
            return None
        return ".".join(segments)
    return None


def _extract_segments_from_role_dict(role_dict: dict[str, Any]) -> list[str] | None:
    """Extract ordered role path segments from nested role dict."""
    if not role_dict:
        return None

    segments: list[str] = []
    node: Any = role_dict
    while isinstance(node, dict):
        if not node:
            return segments if segments else None

        if len(node) != 1:
            return None
        key, child = next(iter(node.items()))
        if not isinstance(key, str):
            return None
        key = key.strip()
        if not key:
            return None
        segments.append(key)
        node = child

    return segments if segments else None
