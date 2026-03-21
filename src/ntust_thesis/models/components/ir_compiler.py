"""Deterministic compiler from IR text to JSON."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ntust_thesis.core.interfaces import Compiler

if TYPE_CHECKING:
    from ntust_thesis.core.types import JSONDict

_PAIR_SIZE = 2


class DeterministicIRCompiler(Compiler):
    """Compile dot-notation IR into final JSON object."""

    def compile(self, ir_text: str, schema: JSONDict) -> JSONDict:
        """Compile IR text under strict schema constraints."""
        _ = schema
        event_type = "unknown.event"
        args_by_idx: dict[int, dict[str, Any]] = {}

        for raw_line in ir_text.splitlines():
            line = raw_line.strip()
            if not line or "=" not in line:
                continue
            left, right = line.split("=", 1)
            key = left.strip()
            value = right.strip()

            if key == "event.type":
                event_type = value
                continue

            prefix = "event.arguments."
            if not key.startswith(prefix):
                continue

            tail = key[len(prefix) :]
            parts = tail.split(".")
            if len(parts) != _PAIR_SIZE:
                continue
            idx_text, field = parts
            if not idx_text.isdigit():
                continue

            idx = int(idx_text)
            bucket = args_by_idx.setdefault(idx, {})
            if field == "span":
                bucket["span"] = _parse_span(value)
            elif field in {"role", "text"}:
                bucket[field] = value

        arguments = []
        for idx in sorted(args_by_idx):
            row = args_by_idx[idx]
            arguments.append(
                {
                    "role": str(row.get("role", "unknown.role")),
                    "text": str(row.get("text", "")),
                    "span": row.get("span", [0, 0]),
                }
            )

        return {"event_type": event_type, "arguments": arguments}


def _parse_span(text: str) -> list[int]:
    """Parse '<start>,<end>' into an integer span."""
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != _PAIR_SIZE:
        return [0, 0]
    try:
        return [int(parts[0]), int(parts[1])]
    except ValueError:
        return [0, 0]
