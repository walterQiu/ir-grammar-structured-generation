"""Centralized prompt builders for model inference."""

from __future__ import annotations


def build_baseline_event_extraction_prompt(
    sentence: str,
    event_type: str | None = None,
    candidate_roles: list[str] | None = None,
    role_multiplicities: dict[str, int] | None = None,
) -> str:
    """Build prompt for baseline direct JSON generation."""
    event_line = f"Event type (reference): {event_type}\n" if event_type else ""
    roles_line = (
        f"Candidate roles: {', '.join(candidate_roles)}\n" if candidate_roles else ""
    )
    multiplicity_line = ""
    if role_multiplicities:
        pairs = ", ".join(
            f"{role}={count}" for role, count in role_multiplicities.items()
        )
        multiplicity_line = f"Role multiplicities: {pairs}\n"
    return (
        "Extract event arguments from the sentence. "
        "The trigger word(s) of the event is marked with **trigger word**.\n"
        "Return only a JSON object with this structure:\n"
        '{"arguments":[{"role":"<role>","text":"<text>"}]}\n'
        "Do not output event type.\n"
        "Do not output span positions.\n"
        "Respect role multiplicities strictly.\n"
        "Do NOT split a single text span into multiple mentions.\n"
        "Do NOT decompose coordinated phrases (e.g., 'A, B, and C').\n"
        "Keep the original text span exactly as in the sentence.\n"
        "No markdown, no extra commentary.\n"
        f"Sentence: {sentence}\n"
        f"{event_line}"
        f"{roles_line}"
        f"{multiplicity_line}"
    )


def build_ir_extraction_prompt(
    sentence: str,
    event_type: str | None = None,
    candidate_roles: list[str] | None = None,
    role_multiplicities: dict[str, int] | None = None,
) -> str:
    """Build prompt for IR extraction stage."""
    event_line = f"Event type (reference): {event_type}\n" if event_type else ""
    roles_line = (
        f"Candidate roles: {', '.join(candidate_roles)}\n" if candidate_roles else ""
    )
    multiplicity_line = ""
    if role_multiplicities:
        pairs = ", ".join(
            f"{role}={count}" for role, count in role_multiplicities.items()
        )
        multiplicity_line = f"Role multiplicities: {pairs}\n"
    return (
        "Please identify the event arguments related to the marked trigger word in the following sentence.\n"
        "The trigger word(s) of the event is marked with **trigger word**.\n\n"
        "Reason step by step about which text spans in the sentence are valid arguments of this event.\n"
        "For each valid argument you find, briefly explain what text span it is and which role it most likely plays.\n"
        "Use the event type, candidate roles, and role multiplicities only as references for deciding validity.\n"
        "Do not invent arguments that are not clearly supported by the sentence.\n"
        "Do not treat one mention as multiple distinct arguments unless the sentence clearly supports that.\n"
        "Do not organize the answer into a table, JSON, key-value pairs, role-label lines, or any other fixed schema.\n"
        "Respond in free-form natural language only.\n\n"
        f"Sentence: {sentence}\n"
        f"{event_line}"
        f"{roles_line}"
        f"{multiplicity_line}"
        "Let's think step by step."
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
        "Convert free-form extraction notes to dot-notation IR.\n"
        "The extraction notes may contain explanations; ignore narrative text and keep only role-span facts.\n"
        "Output only lines in this format:\n"
        "arguments.<role> += <text>\n"
        "Each line corresponds to one role assignment.\n"
        "Respect role multiplicities strictly:\n"
        "- If multiplicity = 1, output exactly ONE line for that role.\n"
        "- If multiplicity > 1, output multiple lines as needed.\n"
        "- If no valid argument is present, output nothing.\n"
        "Do NOT split a single text span into multiple mentions.\n"
        "Do NOT decompose coordinated phrases (e.g., 'A, B, and C').\n"
        "Keep the original text span exactly as given.\n"
        "Do not output event type.\n"
        "No markdown, no extra commentary.\n"
        f"{multiplicity_line}"
        f"Extraction notes:\n{extraction_text}"
    )


def build_schema_generation_prompt(
    extraction_text: str,
    role_multiplicities: dict[str, int] | None = None,
) -> str:
    """Build prompt for converting extraction text directly to final JSON."""
    multiplicity_line = ""
    if role_multiplicities:
        pairs = ", ".join(
            f"{role}={count}" for role, count in role_multiplicities.items()
        )
        multiplicity_line = f"Role multiplicities: {pairs}\n"
    return (
        "Convert extraction notes into final JSON event arguments.\n"
        "Output only this JSON object structure:\n"
        '{"arguments":[{"role":"<role>","text":"<text>"}]}\n'
        "Do not output event type.\n"
        "Respect role multiplicities strictly.\n"
        "Do NOT split a single text span into multiple mentions.\n"
        "Do NOT decompose coordinated phrases (e.g., 'A, B, and C').\n"
        "Keep the original text span exactly as given.\n"
        "No markdown, no extra commentary.\n"
        f"{multiplicity_line}"
        f"Extraction notes:\n{extraction_text}"
    )


def build_direct_ir_prompt(
    sentence: str,
    event_type: str | None = None,
    candidate_roles: list[str] | None = None,
    role_multiplicities: dict[str, int] | None = None,
) -> str:
    """Build prompt for direct sentence-to-IR generation."""
    event_line = f"Event type (reference): {event_type}\n" if event_type else ""
    roles_line = (
        f"Candidate roles: {', '.join(candidate_roles)}\n" if candidate_roles else ""
    )
    multiplicity_line = ""
    if role_multiplicities:
        pairs = ", ".join(
            f"{role}={count}" for role, count in role_multiplicities.items()
        )
        multiplicity_line = f"Role multiplicities: {pairs}\n"
    return (
        "Extract event arguments from the sentence and output dot-notation IR directly.\n"
        "The trigger word(s) of the event is marked with **trigger word**.\n"
        "Output only lines in this format:\n"
        "arguments.<role> += <text>\n"
        "Each line corresponds to one role assignment.\n"
        "Respect role multiplicities strictly:\n"
        "- If multiplicity = 1, output exactly ONE line for that role.\n"
        "- If multiplicity > 1, output multiple lines as needed.\n"
        "- If no valid argument is present, output nothing.\n"
        "Do NOT split a single text span into multiple mentions.\n"
        "Do NOT decompose coordinated phrases (e.g., 'A, B, and C').\n"
        "Keep the original text span exactly as given.\n"
        "Do not output event type.\n"
        "No markdown, no extra commentary.\n"
        f"Sentence: {sentence}\n"
        f"{event_line}"
        f"{roles_line}"
        f"{multiplicity_line}"
    )
