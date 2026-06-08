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
        for idx, (raw_role, raw_spans) in enumerate(payload.items(), start=1):
            valid, error_message = _validate_role_item(raw_role, raw_spans)
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

    role_spans: list[tuple[str, str]] = []
    for idx, (raw_role, raw_spans) in enumerate(payload.items(), start=1):
        valid, error_message = _validate_role_item(raw_role, raw_spans)
        if not valid:
            msg = f"Invalid json IR role entry at index {idx}: {error_message}"
            raise ValueError(msg)
        role_path = role_to_path(raw_role)
        if role_path is None:
            msg = f"Invalid json IR role entry at index {idx}: role path is empty."
            raise ValueError(msg)
        role_spans.extend((role_path, span_item.strip()) for span_item in raw_spans)

    return role_spans


def _validate_role_item(role: object, spans: object) -> tuple[bool, str]:
    """Validate one role->span-list entry from JSON IR object."""
    if not isinstance(role, str):
        return False, "Each JSON key must be a role string."
    if role_to_path(role) is None:
        return False, "Role key cannot be empty."
    if not isinstance(spans, list):
        return False, "Each role value must be a list."
    for item in spans:
        if not isinstance(item, str) or not item.strip():
            return False, "Each span in role list must be a non-empty string."
    return True, ""
