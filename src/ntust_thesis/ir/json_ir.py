"""JSON IR grammar parser and validator."""

from __future__ import annotations

import json

from ntust_thesis.core.role_path import role_to_path
from ntust_thesis.ir.common import IRGrammarValidator, IRValidationResult


class JsonIRValidator(IRGrammarValidator):
    """Validate JSON IR text."""

    def name(self) -> str:
        """Return grammar identifier."""
        return "json"

    def validate(self, ir_text: str) -> IRValidationResult:
        """Validate JSON IR text."""
        try:
            payload = json.loads(ir_text.strip())
        except json.JSONDecodeError as exc:
            return IRValidationResult(
                is_valid=False,
                error_message=f"Invalid JSON: {exc.msg}",
                error_line_no=exc.lineno,
                error_line_text=None,
            )

        if not isinstance(payload, dict):
            return IRValidationResult(
                is_valid=False,
                error_message="Top-level JSON must be an object.",
            )
        arguments = payload.get("arguments")
        if not isinstance(arguments, list):
            return IRValidationResult(
                is_valid=False,
                error_message="Field 'arguments' must be a list.",
            )
        for idx, item in enumerate(arguments, start=1):
            valid, error_message = _validate_argument_item(item)
            if not valid:
                return IRValidationResult(
                    is_valid=False,
                    error_message=error_message,
                    error_line_no=idx,
                    error_line_text=None,
                )
        return IRValidationResult(is_valid=True)


def parse_json_ir(ir_text: str) -> list[tuple[str, str]]:
    """Parse validated JSON IR into (role_path, span_text) pairs."""
    payload = json.loads(ir_text.strip())
    if not isinstance(payload, dict):
        msg = "Invalid json IR: top-level JSON must be an object."
        raise TypeError(msg)

    arguments = payload.get("arguments")
    if not isinstance(arguments, list):
        msg = "Invalid json IR: field 'arguments' must be a list."
        raise TypeError(msg)

    role_spans: list[tuple[str, str]] = []
    for idx, item in enumerate(arguments, start=1):
        valid, error_message = _validate_argument_item(item)
        if not valid:
            msg = f"Invalid json IR argument at index {idx}: {error_message}"
            raise ValueError(msg)
        if not isinstance(item, dict):  # type narrowing
            msg = f"Invalid json IR argument at index {idx}: item must be an object."
            raise TypeError(msg)

        raw_role = item["role"]
        span = item["span"].strip()
        role_path = role_to_path(raw_role)
        if role_path is None:
            msg = f"Invalid json IR argument at index {idx}: role path is empty."
            raise ValueError(msg)
        role_spans.append((role_path, span))

    return role_spans


def _validate_argument_item(item: object) -> tuple[bool, str]:
    """Validate one argument item from JSON IR arguments list."""
    if not isinstance(item, dict):
        return False, "Each argument must be an object."

    role = item.get("role")
    span = item.get("span")

    if not isinstance(role, (str, dict)):
        return False, "Field 'role' must be a string or object."
    if role_to_path(role) is None:
        return False, "Field 'role' cannot be empty."
    if not isinstance(span, str) or not span.strip():
        return False, "Field 'span' must be a non-empty string."
    return True, ""
