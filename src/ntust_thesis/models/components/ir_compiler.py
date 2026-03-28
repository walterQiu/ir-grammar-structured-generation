"""Deterministic compiler from IR text to JSON."""

from __future__ import annotations

from ntust_thesis.core.interfaces import Compiler
from ntust_thesis.core.schemas import EventOutput

_ARGUMENT_PREFIX = "arguments."
_APPEND_OPERATOR = "+="


class DeterministicIRCompiler(Compiler):
    """Compile dot-notation IR into final JSON object."""

    def compile(
        self,
        ir_text: str,
        event_type: str,
    ) -> EventOutput:
        """Compile IR text under strict schema constraints."""
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
            arguments.append(
                {
                    "role": role,
                    "text": mention_text,
                }
            )

        return EventOutput.model_validate(
            {"event_type": event_type, "arguments": arguments}
        )
