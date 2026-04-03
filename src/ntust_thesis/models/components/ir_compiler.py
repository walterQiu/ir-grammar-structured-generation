"""Deterministic compiler from IR text to JSON."""

from __future__ import annotations

from ntust_thesis.core.interfaces import Compiler
from ntust_thesis.core.schemas import EventOutput
from ntust_thesis.ir import (
    get_ir_grammar_parser,
    get_ir_grammar_validator,
)


class DeterministicIRCompiler(Compiler):
    """Compile dot-notation IR into final JSON object."""

    def __init__(self, ir_grammar: str = "dot_notation_ir") -> None:
        """Initialize compiler with grammar validator."""
        self._ir_grammar = ir_grammar
        self._validator = get_ir_grammar_validator(ir_grammar)
        self._parser = get_ir_grammar_parser(ir_grammar)

    def compile(
        self,
        ir_text: str,
        event_type: str,
    ) -> EventOutput:
        """Compile IR text under strict schema constraints."""
        validation = self._validator.validate(ir_text)
        if not validation.is_valid:
            msg = (
                f"Invalid {self._ir_grammar} IR"
                f" at line {validation.error_line_no}: {validation.error_message}"
            )
            raise ValueError(msg)
        role_mentions = self._parser(ir_text)

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
