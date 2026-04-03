"""IR grammar package."""

from ntust_thesis.ir.common import IRGrammarValidator, IRParser, IRValidationResult
from ntust_thesis.ir.dot_notation_ir import (
    DotNotationIRValidator,
    parse_dot_notation_ir,
)
from ntust_thesis.ir.registry import (
    get_ir_grammar_parser,
    get_ir_grammar_validator,
)

__all__ = [
    "DotNotationIRValidator",
    "IRGrammarValidator",
    "IRParser",
    "IRValidationResult",
    "get_ir_grammar_parser",
    "get_ir_grammar_validator",
    "parse_dot_notation_ir",
]
