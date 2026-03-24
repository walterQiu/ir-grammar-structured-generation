"""Deterministic span matching helpers shared by model components."""

from __future__ import annotations

import re

_WORD_PATTERN = re.compile(r"[a-z0-9]+")


def find_span_by_text(tokens: list[str], arg_text: str) -> tuple[int, int]:
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
