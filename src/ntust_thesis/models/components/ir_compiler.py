"""Deterministic compiler from IR text to JSON."""

from __future__ import annotations

from ntust_thesis.core.interfaces import Compiler
from ntust_thesis.core.schemas import EventOutput
from ntust_thesis.models.components.span_matcher import find_span_by_text

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
            span = find_span_by_text(tokens, mention_text)
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
