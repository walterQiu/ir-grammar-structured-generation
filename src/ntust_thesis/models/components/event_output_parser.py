"""Parsers that convert model text into canonical EventOutput."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from ntust_thesis.core.schemas import EventOutput
from ntust_thesis.utils.json_parser import parse_json_object


def parse_event_output_from_arguments_json(
    raw_output: str,
    event_type: str,
) -> EventOutput | None:
    """Parse JSON with arguments(role,span) into canonical EventOutput."""
    parsed = parse_json_object(raw_output)
    if parsed is None:
        return None
    raw_args = parsed.get("arguments")
    if not isinstance(raw_args, list):
        return None

    arguments: list[dict[str, Any]] = []
    for raw_arg in raw_args:
        if not isinstance(raw_arg, dict):
            continue
        role = raw_arg.get("role")
        span = raw_arg.get("span")
        if not isinstance(role, (str, dict)) or not isinstance(span, str):
            continue
        span_text = span.strip()
        if not span_text:
            continue
        if isinstance(role, str):
            normalized_role = role.strip()
            if not normalized_role:
                continue
            role_value: str | dict[str, Any] = normalized_role
        else:
            role_value = role
        arguments.append(
            {
                "role": role_value,
                "span": span_text,
            }
        )

    try:
        return EventOutput.model_validate(
            {"event_type": event_type, "arguments": arguments}
        )
    except ValidationError:
        return None
