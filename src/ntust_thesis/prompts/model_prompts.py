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
    role_multiplicities: dict[str, int] | None = None,
) -> str:
    """Build prompt for IR extraction stage."""
    event_line = f"Event type (reference): {event_type}\n" if event_type else ""
    roles_line = f"Legal roles: {', '.join(legal_roles)}\n" if legal_roles else ""
    multiplicity_line = ""
    if role_multiplicities:
        pairs = ", ".join(
            f"{role}={count}" for role, count in role_multiplicities.items()
        )
        multiplicity_line = f"Role multiplicities: {pairs}\n"
    return (
        "Extract event information from the sentence. "
        "The trigger word(s) of the event is marked with **trigger word**.\n"
        "Return only argument lines in this format:\n"
        "<role>: <text>\n"
        "Respect role multiplicities: do not split one mention into multiple lines.\n"
        "If no argument is found, return exactly: none\n"
        "Do not return JSON.\n"
        f"Sentence: {sentence}\n"
        f"{event_line}"
        f"{roles_line}"
        f"{multiplicity_line}"
    )


def build_ir_generation_prompt(
    extraction_text: str,
    role_multiplicities: dict[str, int] | None = None,
) -> str:
    """Build prompt for converting extraction text to dot-notation IR."""
    multiplicity_line = ""
    if role_multiplicities:
        pairs = ", ".join(
            f"{role}={count}" for role, count in role_multiplicities.items()
        )
        multiplicity_line = f"Role multiplicities: {pairs}\n"
    return (
        "Convert extraction notes to dot-notation IR.\n"
        "Output only lines in this format:\n"
        "arguments.<role> += <text>\n"
        "Each line corresponds to one role assignment.\n"
        "Respect role multiplicities strictly:\n"
        "- If multiplicity = 1, output exactly ONE line for that role.\n"
        "- If multiplicity > 1, output multiple lines as needed.\n"
        "Do NOT split a single text span into multiple mentions.\n"
        "Do NOT decompose coordinated phrases (e.g., 'A, B, and C').\n"
        "Keep the original text span exactly as given.\n"
        "Do not output event type.\n"
        "No markdown, no extra commentary.\n"
        f"{multiplicity_line}"
        f"Extraction notes:\n{extraction_text}"
    )
