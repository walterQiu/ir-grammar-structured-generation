"""Deterministic compiler from IR text to JSON."""

from __future__ import annotations

import re

from ntust_thesis.core.interfaces import Compiler
from ntust_thesis.core.schemas import EventOutput

_WORD_PATTERN = re.compile(r"[a-z0-9]+")
_ARGUMENT_PREFIX = "arguments."
_APPEND_OPERATOR = "+="


class DeterministicIRCompiler(Compiler):
    """Compile dot-notation IR into final JSON object."""

    def compile(
        self,
        ir_text: str,
        source_sentence: str,
        event_type: str,
    ) -> EventOutput:
        """Compile IR text under strict schema constraints."""
        tokens = source_sentence.split()
        role_mentions: list[tuple[str, str]] = []

        for raw_line in ir_text.splitlines():
            line = raw_line.strip()
            if (
                not line
                or _APPEND_OPERATOR not in line
                or not line.startswith(_ARGUMENT_PREFIX)
            ):
                continue

            left, right = line.split(_APPEND_OPERATOR, 1)
            key = left.strip()
            mention_text = right.strip()
            if not mention_text:
                continue

            role = key[len(_ARGUMENT_PREFIX) :].strip()
            if not role:
                continue
            role_mentions.append((role, mention_text))

        arguments = []
        for role, mention_text in role_mentions:
            span = _find_span_by_text(tokens, mention_text)
            arguments.append(
                {
                    "role": role,
                    "text": mention_text,
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
