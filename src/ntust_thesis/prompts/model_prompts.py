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


# def build_ir_generation_prompt(
#     extraction_text: str,
#     role_multiplicities: dict[str, int] | None = None,
# ) -> str:
#     """Build prompt for converting extraction text to dot-notation IR."""
#     multiplicity_line = ""
#     if role_multiplicities:
#         pairs = ", ".join(
#             f"{role}={count}" for role, count in role_multiplicities.items()
#         )
#         multiplicity_line = f"Role multiplicities: {pairs}\n"
#     return (
#         "Convert free-form extraction notes to dot-notation IR.\n"
#         "The extraction notes may contain explanations; ignore narrative text and keep only role-span facts.\n"
#         "Output only lines in this format:\n"
#         "arguments.<role> += <text>\n"
#         "Each line corresponds to one role assignment.\n"
#         "Respect role multiplicities strictly:\n"
#         "- If multiplicity = 1, output exactly ONE line for that role.\n"
#         "- If multiplicity > 1, output multiple lines as needed.\n"
#         "- If no valid argument is present, output nothing.\n"
#         "Do NOT split a single text span into multiple mentions.\n"
#         "Do NOT decompose coordinated phrases (e.g., 'A, B, and C').\n"
#         "Keep the original text span exactly as given.\n"
#         "Do not output event type.\n"
#         "No markdown, no extra commentary.\n"
#         f"{multiplicity_line}"
#         f"Extraction notes:\n{extraction_text}"
#     )
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
        multiplicity_line = f"Allowed roles and multiplicities: {pairs}\n"

    in_context_examples = build_ir_generation_in_context_examples()

    return (
        "Convert extraction notes to dot-notation IR.\n"
        "Output only lines in this format:\n"
        "arguments.<role> += <text>\n"
        "\n"
        "Rules:\n"
        "- Use only roles listed in 'Allowed roles and multiplicities'.\n"
        "- Do not invent or rename roles.\n"
        "- If a role has multiplicity 1, output at most one line for that role.\n"
        "- Output only explicit, valid role spans.\n"
        "- Do not output implied, hypothetical, uncertain, rejected, or explanatory content.\n"
        "- Keep spans as written. Do not split coordinated phrases.\n"
        "- Do not output event type or any extra text.\n"
        "\n"
        f"{in_context_examples}\n"
        f"Allowed roles and multiplicities: {multiplicity_line}"
        f"Extraction notes:\n{extraction_text}"
    )


def build_ir_generation_in_context_examples() -> str:
    """Return in-context examples used by IR-generation prompt."""
    return (
        "Example 1:\n"
        "Allowed roles and multiplicities: victim=1, place=1\n"
        "Extraction notes:\n"
        "The sentence suggests a possible fatal incident, though the exact event type is not explicitly stated. "
        "The victim is not clearly identified; the text refers to people who may have been affected, but this is speculative and not confirmed. "
        "The place is clearly mentioned as the coastal district of Norvik. "
        "The text also discusses unrest in neighboring settlements, which is contextual background and not a direct argument.\n"
        "Output:\n"
        "arguments.place += the coastal district of Norvik\n"
        "\n"
        "Example 2:\n"
        "Allowed roles and multiplicities: attacker=1, target=1, instrument=1\n"
        "Extraction notes:\n"
        "The sentence describes an attack scenario. "
        "The attacker is identified as a group of unidentified militants. "
        "The target is a convoy of supply vehicles. "
        "The instrument is described as improvised explosive devices and small arms fire used during the incident. "
        "The text also mentions prior warnings and rising tensions, which are explanatory context and not role arguments.\n"
        "Output:\n"
        "arguments.attacker += a group of unidentified militants\n"
        "arguments.target += a convoy of supply vehicles\n"
        "arguments.instrument += improvised explosive devices and small arms fire used during the incident\n"
        "\n"
        "Example 3:\n"
        "Allowed roles and multiplicities: attacker=1, target=2, place=1\n"
        "Extraction notes:\n"
        "The sentence reports an assault. "
        "The attacker is clearly identified as local militia members. "
        "Two targets are explicitly mentioned: the municipal checkpoint and a nearby storage depot. "
        "The place is given as the eastern edge of Darsin. "
        "The notes also mention public anger after the incident, which is background explanation rather than an argument.\n"
        "Output:\n"
        "arguments.attacker += local militia members\n"
        "arguments.target += the municipal checkpoint\n"
        "arguments.target += a nearby storage depot\n"
        "arguments.place += the eastern edge of Darsin\n"
        "\n"
        "Example 4:\n"
        "Allowed roles and multiplicities: instrument=2, place=1\n"
        "Extraction notes:\n"
        "The sentence refers to a violent confrontation. "
        "Two instruments are explicitly described: hunting rifles and homemade incendiary devices. "
        "The place is stated as a farming village outside Lembat. "
        "The text also includes discussion of longstanding land disputes, which provides context but is not an event argument.\n"
        "Output:\n"
        "arguments.instrument += hunting rifles\n"
        "arguments.instrument += homemade incendiary devices\n"
        "arguments.place += a farming village outside Lembat\n"
        "\n"
        "Example 5:\n"
        "Allowed roles and multiplicities: victim=2, place=1\n"
        "Extraction notes:\n"
        "The sentence implies a deadly incident. "
        "Two victims are clearly mentioned: a senior engineer and his assistant. "
        "The place is identified as a maintenance tunnel beneath the West Ardin plant. "
        "The notes also compare the site to safer facilities elsewhere, which is contrastive context and not part of the event arguments.\n"
        "Output:\n"
        "arguments.victim += a senior engineer\n"
        "arguments.victim += his assistant\n"
        "arguments.place += a maintenance tunnel beneath the West Ardin plant\n"
        "\n"
        "Example 6:\n"
        "Allowed roles and multiplicities: target=1, instrument=1, place=1\n"
        "Extraction notes:\n"
        "The sentence describes a destructive act. "
        "The target is clearly stated as the main transmission tower. "
        "The instrument is not a list of separate arguments; it is given as fuel cans, scrap metal, and wiring assembled into a single device. "
        "The place is identified as an industrial corridor near Pel Sanur. "
        "The text also mentions economic frustration in the area, which is explanatory background and not an event role.\n"
        "Output:\n"
        "arguments.target += the main transmission tower\n"
        "arguments.instrument += fuel cans, scrap metal, and wiring assembled into a single device\n"
        "arguments.place += an industrial corridor near Pel Sanur"
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
