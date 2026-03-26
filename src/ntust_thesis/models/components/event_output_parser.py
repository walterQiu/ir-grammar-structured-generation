"""Parsers that convert model text into canonical EventOutput."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from ntust_thesis.core.schemas import EventOutput
from ntust_thesis.models.components.span_matcher import find_span_by_text
from ntust_thesis.utils.json_parser import parse_json_object


def parse_event_output_from_arguments_json(
    raw_output: str,
    source_sentence: str,
    event_type: str,
) -> EventOutput | None:
    """Parse JSON with arguments(role,text) and fill deterministic spans."""
    parsed = parse_json_object(raw_output)
    if parsed is None:
        return None
    raw_args = parsed.get("arguments")
    if not isinstance(raw_args, list):
        return None

    tokens = source_sentence.split()
    arguments: list[dict[str, Any]] = []
    for raw_arg in raw_args:
        if not isinstance(raw_arg, dict):
            continue
        role = raw_arg.get("role")
        text = raw_arg.get("text")
        if not isinstance(role, str) or not isinstance(text, str):
            continue
        mention_text = text.strip()
        if not mention_text:
            continue
        span = find_span_by_text(tokens, mention_text)
        arguments.append(
            {
                "role": role.strip(),
                "text": mention_text,
                "span": span,
            }
        )

    try:
        return EventOutput.model_validate(
            {"event_type": event_type, "arguments": arguments}
        )
    except ValidationError:
        return None
