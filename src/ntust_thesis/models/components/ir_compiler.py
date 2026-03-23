"""Deterministic compiler from IR text to JSON."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from ntust_thesis.core.interfaces import Compiler
from ntust_thesis.core.schemas import EventOutput

if TYPE_CHECKING:
    from ntust_thesis.core.schemas import OutputSchema

_PAIR_SIZE = 2
_WORD_PATTERN = re.compile(r"[a-z0-9]+")


class DeterministicIRCompiler(Compiler):
    """Compile dot-notation IR into final JSON object."""

    def compile(
        self,
        ir_text: str,
        output_schema: OutputSchema,
        source_sentence: str,
        event_type: str,
    ) -> EventOutput:
        """Compile IR text under strict schema constraints."""
        _ = output_schema
        tokens = source_sentence.split()
        args_by_idx: dict[int, dict[str, Any]] = {}

        for raw_line in ir_text.splitlines():
            line = raw_line.strip()
            if not line or "=" not in line:
                continue
            left, right = line.split("=", 1)
            key = left.strip()
            value = right.strip()

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
            if field in {"role", "text"}:
                bucket[field] = value  # dict is mutable, so this updates args_by_idx

        arguments = []
        for idx in sorted(args_by_idx):
            row = args_by_idx[idx]
            arg_text = str(row.get("text", ""))
            span = _find_span_by_text(tokens, arg_text)
            arguments.append(
                {
                    "role": str(row.get("role", "unknown.role")),
                    "text": arg_text,
                    "span": span,
                }
            )

        return EventOutput.model_validate(
            {"event_type": event_type, "arguments": arguments}
        )


def _find_span_by_text(tokens: list[str], arg_text: str) -> tuple[int, int]:
    """Find first span by normalized sub-token sequence matching."""
    text = arg_text.strip()
    if not text:
        return (-1, -1)

    sent_subtokens, sent_map = _to_subtokens_with_map(tokens)
    query_subtokens = _to_subtokens(text.split())
    width = len(query_subtokens)
    if width == 0 or width > len(sent_subtokens):
        return (-1, -1)

    for start in range(len(sent_subtokens) - width + 1):
        end = start + width
        if sent_subtokens[start:end] == query_subtokens:
            return (sent_map[start], sent_map[end - 1])
    return (-1, -1)


def _to_subtokens_with_map(tokens: list[str]) -> tuple[list[str], list[int]]:
    """Normalize sentence tokens to sub-tokens and map to original indices."""
    subtokens: list[str] = []
    index_map: list[int] = []
    for idx, token in enumerate(tokens):
        parts = _token_to_subtokens(token)
        subtokens.extend(parts)
        index_map.extend([idx] * len(parts))
    return subtokens, index_map


def _to_subtokens(tokens: list[str]) -> list[str]:
    """Normalize query tokens into comparable sub-tokens."""
    subtokens: list[str] = []
    for token in tokens:
        subtokens.extend(_token_to_subtokens(token))
    return subtokens


def _token_to_subtokens(token: str) -> list[str]:
    """Split token into normalized alphanumeric pieces."""
    lowered = token.lower().replace("\u2019", "'")
    return _WORD_PATTERN.findall(lowered)
