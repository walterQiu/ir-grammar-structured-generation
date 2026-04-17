"""Registry-style resolver for IR grammars."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ntust_thesis.ir.code4struct_ir import (
    Code4StructIRValidator,
    parse_code4struct_ir,
)
from ntust_thesis.ir.incremental_assignment_ir import (
    IncrementalAssignmentIRValidator,
    parse_incremental_assignment_ir,
)
from ntust_thesis.ir.json_ir import JsonIRValidator, parse_json_ir

if TYPE_CHECKING:
    from ntust_thesis.ir.common import IRGrammarValidator, IRParser


def get_ir_grammar_validator(grammar_name: str) -> IRGrammarValidator:
    """Return validator instance for grammar name."""
    if grammar_name == "json":
        return JsonIRValidator()
    if grammar_name == "incremental_assignment_ir":
        return IncrementalAssignmentIRValidator()
    if grammar_name == "code4struct_ir":
        return Code4StructIRValidator()
    msg = f"Unsupported IR grammar: {grammar_name}"
    raise ValueError(msg)


def get_ir_grammar_parser(grammar_name: str) -> IRParser:
    """Return parser function for grammar name."""
    if grammar_name == "json":
        return parse_json_ir
    if grammar_name == "incremental_assignment_ir":
        return parse_incremental_assignment_ir
    if grammar_name == "code4struct_ir":
        return parse_code4struct_ir
    msg = f"Unsupported IR grammar: {grammar_name}"
    raise ValueError(msg)
