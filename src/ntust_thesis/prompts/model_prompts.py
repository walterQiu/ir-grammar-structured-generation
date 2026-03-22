"""Centralized prompt builders for model inference."""

from __future__ import annotations


def build_baseline_event_extraction_prompt(input_text: str) -> str:
    """Build prompt for baseline direct JSON generation."""
    return (
        "Extract event information from the text. "
        "Return only a JSON object with keys: "
        '"event_type" (string), "arguments" (array of objects). '
        'Each argument object has keys: "role", "text", "span".\n'
        f"Text: {input_text}"
    )


def build_ir_extraction_prompt(input_text: str) -> str:
    """Build prompt for IR extraction stage."""
    return (
        "Extract event information from the text. "
        "Return concise plain text with two sections:\n"
        "1) event_type: <event type>\n"
        "2) arguments: one argument per line as '<role>: <text>' or 'none'.\n"
        "Do not return JSON.\n"
        f"Text: {input_text}"
    )


def build_ir_generation_prompt(extraction_text: str) -> str:
    """Build prompt for converting extraction text to dot-notation IR."""
    return (
        "Convert extraction notes to dot-notation IR. "
        "Output only lines in this format:\n"
        "event.type = <event type>\n"
        "event.arguments.<index>.role = <role>\n"
        "event.arguments.<index>.text = <text>\n"
        "event.arguments.<index>.span = <start>,<end>\n"
        "If span is unknown, use 0,0.\n"
        "No markdown, no extra commentary.\n"
        f"Extraction notes:\n{extraction_text}"
    )
