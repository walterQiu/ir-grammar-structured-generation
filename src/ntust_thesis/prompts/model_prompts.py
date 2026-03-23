"""Centralized prompt builders for model inference."""

from __future__ import annotations


def build_baseline_event_extraction_prompt(
    sentence: str,
    event_type: str | None = None,
    legal_roles: list[str] | None = None,
) -> str:
    """Build prompt for baseline direct JSON generation."""
    event_line = f"Event type: {event_type}\n" if event_type else ""
    roles_line = f"Legal roles: {', '.join(legal_roles)}\n" if legal_roles else ""
    return (
        "Extract event information from the sentence. "
        "The trigger word(s) of the event is marked with **trigger word**.\n"
        "Return only a JSON object with keys: "
        '"event_type" (string), "arguments" (array of objects). '
        'Each argument object has keys: "role", "text", "span".\n'
        f"Sentence: {sentence}\n"
        f"{event_line}"
        f"{roles_line}"
    )


def build_ir_extraction_prompt(
    sentence: str,
    event_type: str | None = None,
    legal_roles: list[str] | None = None,
) -> str:
    """Build prompt for IR extraction stage."""
    event_line = f"Event type: {event_type}\n" if event_type else ""
    roles_line = f"Legal roles: {', '.join(legal_roles)}\n" if legal_roles else ""
    return (
        "Extract event information from the sentence. "
        "The trigger word(s) of the event is marked with **trigger word**.\n"
        "Return concise plain text with two sections:\n"
        "1) event_type: <event type>\n"
        "2) arguments: one argument per line as '<role>: <text>' or 'none'.\n"
        "Do not return JSON.\n"
        f"Sentence: {sentence}\n"
        f"{event_line}"
        f"{roles_line}"
    )


def build_ir_generation_prompt(extraction_text: str) -> str:
    """Build prompt for converting extraction text to dot-notation IR."""
    return (
        "Convert extraction notes to dot-notation IR. "
        "Output only lines in this format:\n"
        "event.type = <event type>\n"
        "event.arguments.<index>.role = <role>\n"
        "event.arguments.<index>.text = <text>\n"
        "No markdown, no extra commentary.\n"
        f"Extraction notes:\n{extraction_text}"
    )
