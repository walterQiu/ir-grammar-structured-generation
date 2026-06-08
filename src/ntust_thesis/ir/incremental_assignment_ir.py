"""Incremental-assignment IR grammar parser and validator."""

from __future__ import annotations

import re

from ntust_thesis.ir.common import IRGrammarValidator, IRValidationResult

_APPEND_OPERATOR = "+="
_PATH_SEGMENT_PATTERN = r"[A-Za-z_][A-Za-z0-9_]*"
_INCREMENTAL_ASSIGNMENT_LINE_RE = re.compile(
    rf"^(?P<path>{_PATH_SEGMENT_PATTERN}(?:\.{_PATH_SEGMENT_PATTERN})*)"
    rf"\s*{re.escape(_APPEND_OPERATOR)}\s*"
    r"(?P<span>\S(?:.*\S)?)$"
)


class IncrementalAssignmentIRValidator(IRGrammarValidator):
    """Validate incremental-assignment IR lines."""

    def name(self) -> str:
        """Return grammar identifier."""
        return "incremental_assignment_ir"

    def validate(self, ir_text: str) -> IRValidationResult:
        """Validate incremental-assignment IR text."""
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
            if _parse_incremental_assignment_line(line) is None:
                return IRValidationResult(
                    is_valid=False,
                    error_message="Invalid incremental-assignment path or empty span.",
                    error_line_no=idx,
                    error_line_text=raw_line,
                )

        return IRValidationResult(is_valid=True)


def parse_incremental_assignment_ir(ir_text: str) -> list[tuple[str, str]]:
    """Parse validated incremental-assignment IR into (role_path, span_text) pairs."""
    spans: list[tuple[str, str]] = []
    for idx, raw_line in enumerate(ir_text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parsed = _parse_incremental_assignment_line(line)
        if parsed is None:
            msg = f"Invalid incremental_assignment_ir at line {idx}: {raw_line}"
            raise ValueError(msg)
        role_path, span_text = parsed
        spans.append((role_path, span_text))
    return spans


def _parse_incremental_assignment_line(line: str) -> tuple[str, str] | None:
    """Parse one non-empty line under incremental-assignment grammar."""
    match = _INCREMENTAL_ASSIGNMENT_LINE_RE.fullmatch(line)
    if match is None:
        return None
    return match.group("path"), match.group("span")
