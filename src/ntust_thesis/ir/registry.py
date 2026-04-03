"""Registry-style resolver for IR grammars."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ntust_thesis.ir.dot_notation_ir import (
    DotNotationIRValidator,
    parse_dot_notation_ir,
)

if TYPE_CHECKING:
    from ntust_thesis.ir.common import IRGrammarValidator, IRParser


def get_ir_grammar_validator(grammar_name: str) -> IRGrammarValidator:
    """Return validator instance for grammar name."""
    if grammar_name == "dot_notation_ir":
        return DotNotationIRValidator()
    msg = f"Unsupported IR grammar: {grammar_name}"
    raise ValueError(msg)


def get_ir_grammar_parser(grammar_name: str) -> IRParser:
    """Return parser function for grammar name."""
    if grammar_name == "dot_notation_ir":
        return parse_dot_notation_ir
    msg = f"Unsupported IR grammar: {grammar_name}"
    raise ValueError(msg)
