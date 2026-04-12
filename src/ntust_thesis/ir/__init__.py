"""IR grammar package."""

from ntust_thesis.ir.code4struct_ir import (
    Code4StructIRValidator,
    parse_code4struct_ir,
)
from ntust_thesis.ir.common import IRGrammarValidator, IRParser, IRValidationResult
from ntust_thesis.ir.dot_notation_ir import (
    DotNotationIRValidator,
    parse_dot_notation_ir,
)
from ntust_thesis.ir.json_ir import JsonIRValidator, parse_json_ir
from ntust_thesis.ir.registry import (
    get_ir_grammar_parser,
    get_ir_grammar_validator,
)

__all__ = [
    "Code4StructIRValidator",
    "DotNotationIRValidator",
    "IRGrammarValidator",
    "IRParser",
    "IRValidationResult",
    "JsonIRValidator",
    "get_ir_grammar_parser",
    "get_ir_grammar_validator",
    "parse_code4struct_ir",
    "parse_dot_notation_ir",
    "parse_json_ir",
]
