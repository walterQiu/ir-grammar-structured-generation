"""IR grammar validators and parser helpers."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

_ARGUMENT_PREFIX = "arguments."
_APPEND_OPERATOR = "+="
_PATH_SEGMENT_PATTERN = r"[A-Za-z_][A-Za-z0-9_]*"
_DOT_NOTATION_LINE_RE = re.compile(
    rf"^{re.escape(_ARGUMENT_PREFIX)}"
    rf"(?P<path>{_PATH_SEGMENT_PATTERN}(?:\.{_PATH_SEGMENT_PATTERN})*)"
    rf"\s*{re.escape(_APPEND_OPERATOR)}\s*"
    r"(?P<span>\S(?:.*\S)?)$"
)


@dataclass(frozen=True, slots=True)
class IRValidationResult:
    """Validation result for one IR text."""

    is_valid: bool
    error_message: str | None = None
    error_line_no: int | None = None
    error_line_text: str | None = None


class IRGrammarValidator(ABC):
    """Interface for pluggable IR grammar validators."""

    @abstractmethod
    def name(self) -> str:
        """Return grammar identifier."""

    @abstractmethod
    def validate(self, ir_text: str) -> IRValidationResult:
        """Validate IR text and return structured result."""


class DotNotationIRValidator(IRGrammarValidator):
    """Validate dot-notation IR lines."""

    def name(self) -> str:
        """Return grammar identifier."""
        return "dot_notation_ir"

    def validate(self, ir_text: str) -> IRValidationResult:
        """Validate dot-notation IR text."""
        if not ir_text.strip():  # no role or span exists
            return IRValidationResult(is_valid=True)

        for idx, raw_line in enumerate(ir_text.splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            if _APPEND_OPERATOR not in line:
                return IRValidationResult(
                    is_valid=False,
                    error_message=f"Missing '{_APPEND_OPERATOR}' operator.",
                    error_line_no=idx,
                    error_line_text=raw_line,
                )
            if not line.startswith(_ARGUMENT_PREFIX):
                return IRValidationResult(
                    is_valid=False,
                    error_message=f"Line must start with '{_ARGUMENT_PREFIX}'.",
                    error_line_no=idx,
                    error_line_text=raw_line,
                )
            if _parse_dot_notation_line(line) is None:
                return IRValidationResult(
                    is_valid=False,
                    error_message="Invalid dot-notation path or empty span.",
                    error_line_no=idx,
                    error_line_text=raw_line,
                )

        return IRValidationResult(is_valid=True)


def get_ir_grammar_validator(grammar_name: str) -> IRGrammarValidator:
    """Return validator instance for grammar name."""
    if grammar_name == "dot_notation_ir":
        return DotNotationIRValidator()
    msg = f"Unsupported IR grammar: {grammar_name}"
    raise ValueError(msg)


def parse_dot_notation_ir(ir_text: str) -> list[tuple[str, str]]:
    """Parse validated dot-notation IR into (role, mention_text) pairs."""
    mentions: list[tuple[str, str]] = []
    for idx, raw_line in enumerate(ir_text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parsed = _parse_dot_notation_line(line)
        if parsed is None:
            msg = f"Invalid dot_notation_ir at line {idx}: {raw_line}"
            raise ValueError(msg)
        role, mention_text = parsed
        mentions.append((role, mention_text))
    return mentions


def _parse_dot_notation_line(line: str) -> tuple[str, str] | None:
    """Parse one non-empty line under dot-notation grammar."""
    match = _DOT_NOTATION_LINE_RE.fullmatch(line)
    if match is None:
        return None
    return match.group("path"), match.group("span")
