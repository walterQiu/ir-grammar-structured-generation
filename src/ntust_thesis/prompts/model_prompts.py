"""Centralized prompt builders for model inference."""

from __future__ import annotations

from ntust_thesis.prompts.ir_generation_examples import (
    build_ir_generation_in_context_examples,
)

PromptPair = tuple[str, str]  # (system prompt, user_prompt)


def build_one_stage_json_prompt(
    sentence: str,
    event_type: str | None = None,
    candidate_roles: list[str] | None = None,
    role_multiplicities: dict[str, int] | None = None,
) -> PromptPair:
    """Build prompt for one-stage direct JSON generation."""
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
    system_prompt = (
        "Extract event arguments from the sentence. "
        "The trigger word(s) of the event is marked with **trigger word**.\n"
        "Return only a JSON object with this structure:\n"
        '{"arguments":[{"role":"<role>","span":"<span>"}]}\n'
        "Do not output event type.\n"
        "Do not output span positions.\n"
        "Respect role multiplicities strictly.\n"
        "Do NOT split a single text span into multiple spans.\n"
        "Do NOT decompose coordinated phrases (e.g., 'A, B, and C').\n"
        "Keep the original text span exactly as in the sentence.\n"
        "No markdown, no extra commentary.\n"
    )
    user_prompt = f"{event_line}{roles_line}{multiplicity_line}Sentence: {sentence}\n"
    return system_prompt, user_prompt


def build_two_stage_extraction_prompt(
    sentence: str,
    event_type: str | None = None,
    candidate_roles: list[str] | None = None,
    role_multiplicities: dict[str, int] | None = None,
) -> PromptPair:
    """Build prompt for two-stage extraction step."""
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
    system_prompt = (
        "Please identify the event arguments related to the marked trigger word in the following sentence.\n"
        "The trigger word(s) of the event is marked with **trigger word**.\n\n"
        "Reason step by step about which text spans in the sentence are valid arguments of this event.\n"
        "For each valid argument you find, briefly explain what text span it is and which role it most likely plays.\n"
        "Use the event type, candidate roles, and role multiplicities only as references for deciding validity.\n"
        "Do not invent arguments that are not clearly supported by the sentence.\n"
        "Do not treat one span as multiple distinct arguments unless the sentence clearly supports that.\n"
        "Do not organize the answer into a table, JSON, key-value pairs, role-label lines, or any other fixed schema.\n"
        "Respond in free-form natural language only.\n\n"
    )
    user_prompt = (
        f"{event_line}"
        f"{roles_line}"
        f"{multiplicity_line}"
        f"Sentence: {sentence}\n"
        "Let's think step by step."
    )
    return system_prompt, user_prompt


def build_two_stage_ir_prompt(
    extraction_text: str,
    event_type: str | None,
    candidate_roles: list[str] | None,
    role_multiplicities: dict[str, int] | None = None,
    ir_grammar: str = "dot_notation_ir",
) -> PromptPair:
    """Build two-stage IR-generation prompt by selected grammar."""
    if ir_grammar == "dot_notation_ir":
        return build_two_stage_dot_notation_ir_prompt(
            extraction_text=extraction_text,
            role_multiplicities=role_multiplicities,
        )
    if ir_grammar == "code4struct_ir":
        return build_two_stage_code4struct_ir_prompt(
            extraction_text=extraction_text,
            event_type=event_type,
            candidate_roles=candidate_roles,
            role_multiplicities=role_multiplicities,
        )
    msg = f"Unsupported IR grammar for two-stage prompt building: {ir_grammar}"
    raise ValueError(msg)


def build_two_stage_dot_notation_ir_prompt(
    extraction_text: str,
    role_multiplicities: dict[str, int] | None = None,
) -> PromptPair:
    """Build two-stage prompt for dot-notation IR generation."""
    multiplicity_line = ""
    if role_multiplicities:
        pairs = ", ".join(
            f"{role}={count}" for role, count in role_multiplicities.items()
        )
        multiplicity_line = f"Allowed roles and multiplicities: {pairs}\n"

    in_context_examples = build_ir_generation_in_context_examples(
        ir_grammar="dot_notation_ir"
    )

    system_prompt = (
        "Convert extraction notes to dot-notation IR.\n"
        "Output only lines in this format:\n"
        "arguments.<role> += <span>\n"
        "\n"
        "Rules:\n"
        "- Use only roles listed in 'Allowed roles and multiplicities'.\n"
        "- Do not invent or rename roles.\n"
        "- If a role has multiplicity 1, output at most one line for that role.\n"
        "- Output only explicit, valid role spans.\n"
        "- Do not output implied, hypothetical, uncertain, rejected, or explanatory content.\n"
        "- Keep spans as written. Do not split coordinated phrases.\n"
        "- If no valid arguments are found, output nothing (do not generate any lines or text).\n"
        "- Do not output event type or any extra text.\n"
        "\n"
        f"{in_context_examples}\n"
    )
    user_prompt = f"{multiplicity_line}Extraction notes:\n{extraction_text}"
    return system_prompt, user_prompt


def build_two_stage_code4struct_ir_prompt(
    extraction_text: str,
    event_type: str | None,
    candidate_roles: list[str] | None,
    role_multiplicities: dict[str, int] | None = None,
) -> PromptPair:
    """Build two-stage prompt for CODE4STRUCT-style IR generation."""
    event_class = _to_event_class_name(event_type)
    ontology_block = _build_code4struct_ontology_block(
        event_class=event_class,
        candidate_roles=candidate_roles,
    )
    in_context_examples = build_ir_generation_in_context_examples(
        ir_grammar="code4struct_ir"
    )
    multiplicity_line = ""
    if role_multiplicities:
        pairs = ", ".join(
            f"{role}={count}" for role, count in role_multiplicities.items()
        )
        multiplicity_line = f"Allowed roles and multiplicities: {pairs}\n"
    task_block = (
        '"""\n'
        f"Convert the following extraction notes into an instance of {event_class}.\n"
        f'"{extraction_text}"\n'
        '"""\n'
        f"{event_class.lower()}_event = {event_class}(\n"
    )

    system_prompt = (
        "Convert extraction notes to code-like IR.\n"
        "Output only a Python-like event instantiation in this format:\n"
        "\n"
        "Rules:\n"
        "- Use only roles listed in 'Allowed roles and multiplicities'.\n"
        "- Do not invent or rename roles.\n"
        "- If a role has multiplicity 1, include at most one span in that role's list.\n"
        "- Always represent role values as lists, even when there is only one span.\n"
        "- Output only explicit, valid role spans.\n"
        "- Do not output implied, hypothetical, uncertain, rejected, or explanatory content.\n"
        "- Keep spans as written. Do not split coordinated phrases.\n"
        f"- If no valid arguments are found, output:\n{event_class.lower()}_event = {event_class}()\n"
        "- Do not output comments, explanations, or any extra text.\n"
        "\n"
        f"{in_context_examples}\n"
    )
    user_prompt = f"{ontology_block}\n\n{multiplicity_line}{task_block}"
    return system_prompt, user_prompt


def build_two_stage_json_prompt(
    extraction_text: str,
    role_multiplicities: dict[str, int] | None = None,
) -> PromptPair:
    """Build prompt for two-stage extraction->JSON generation."""
    multiplicity_line = ""
    if role_multiplicities:
        pairs = ", ".join(
            f"{role}={count}" for role, count in role_multiplicities.items()
        )
        multiplicity_line = f"Role multiplicities: {pairs}\n"
    system_prompt = (
        "Convert extraction notes into final JSON event arguments.\n"
        "Output only this JSON object structure:\n"
        '{"arguments":[{"role":"<role>","span":"<span>"}]}\n'
        "Do not output event type.\n"
        "Respect role multiplicities strictly.\n"
        "Do NOT split a single text span into multiple spans.\n"
        "Do NOT decompose coordinated phrases (e.g., 'A, B, and C').\n"
        "Keep the original text span exactly as given.\n"
        "No markdown, no extra commentary.\n"
    )
    user_prompt = f"{multiplicity_line}Extraction notes:\n{extraction_text}"
    return system_prompt, user_prompt


def build_one_stage_dot_notation_ir_prompt(
    sentence: str,
    event_type: str | None = None,
    candidate_roles: list[str] | None = None,
    role_multiplicities: dict[str, int] | None = None,
) -> PromptPair:
    """Build one-stage prompt for direct dot-notation IR generation."""
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
    in_context_examples = build_ir_generation_in_context_examples(
        ir_grammar="dot_notation_ir"
    )
    system_prompt = (
        "Extract event arguments from the sentence and output dot-notation IR directly.\n"
        "The trigger word(s) of the event is marked with **trigger word**.\n"
        "Output only lines in this format:\n"
        "arguments.<role> += <span>\n"
        "Each line corresponds to one role assignment.\n"
        "Respect role multiplicities strictly:\n"
        "- If multiplicity = 1, output exactly ONE line for that role.\n"
        "- If multiplicity > 1, output multiple lines as needed.\n"
        "- If no valid argument is present, output nothing.\n"
        "Do NOT split a single text span into multiple spans.\n"
        "Do NOT decompose coordinated phrases (e.g., 'A, B, and C').\n"
        "Keep the original text span exactly as given.\n"
        "Do not output event type.\n"
        "No markdown, no extra commentary.\n"
        "\n"
        f"{in_context_examples}\n"
    )
    user_prompt = f"{event_line}{roles_line}{multiplicity_line}Sentence: {sentence}\n"
    return system_prompt, user_prompt


def _build_code4struct_ontology_block(
    event_class: str,
    candidate_roles: list[str] | None,
) -> str:
    """Build CODE4STRUCT ontology block without entity/event docstrings."""
    roles = candidate_roles or []
    role_identifiers = [_role_to_identifier(role) for role in roles]
    role_args = "\n".join(
        f"        {role}: List[Entity] = []," for role in role_identifiers
    )
    role_assignments = "\n".join(
        f"        self.{role} = {role}" for role in role_identifiers
    )
    if not role_args:
        role_args = "        pass"
    if not role_assignments:
        role_assignments = "        pass"

    return (
        "from typing import List\n\n"
        "class Entity:\n"
        "    def __init__(self, name: str):\n"
        "        self.name = name\n\n"
        "class Event:\n"
        '    def __init__(self, name: str = ""):\n'
        "        self.name = name\n\n"
        f"class {event_class}(Event):\n"
        "    def __init__(\n"
        "        self,\n"
        f"{role_args}\n"
        "    ):\n"
        f"{role_assignments}\n"
    )


def _to_event_class_name(event_type: str | None) -> str:
    """Convert full dot-delimited event_type into a Python class-like name."""
    if not event_type:
        return "TargetEvent"
    parts = [part.strip() for part in event_type.split(".") if part.strip()]
    if not parts:
        return "TargetEvent"

    class_name_parts: list[str] = []
    for part in parts:
        cleaned = part.replace("/", "")
        if not cleaned:
            continue
        class_name_parts.append(cleaned[0].upper() + cleaned[1:])
    if not class_name_parts:
        return "TargetEvent"
    return "".join(class_name_parts)


def _role_to_identifier(role: str) -> str:
    """Convert role path to a Python-identifier constructor argument."""
    identifier = role.strip().replace(".", "__").replace("-", "_")
    if not identifier:
        return "role"
    if not (identifier[0].isalpha() or identifier[0] == "_"):
        identifier = f"role_{identifier}"
    return identifier
