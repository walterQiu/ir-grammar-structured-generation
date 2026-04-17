"""Centralized prompt builders for model inference."""

from __future__ import annotations

from ntust_thesis.prompts.icl_examples import (
    build_one_stage_ir_in_context_examples,
    build_two_stage_ir_in_context_examples,
)

PromptPair = tuple[str, str]  # (system prompt, user_prompt)


def build_one_stage_ir_prompt(
    sentence: str,
    event_type: str,
    ir_grammar: str,
    role_multiplicities: dict[str, int],
) -> PromptPair:
    """Build one-stage structured-generation prompt by selected grammar."""
    if ir_grammar == "json":
        return build_one_stage_json_prompt(
            sentence=sentence,
            event_type=event_type,
            role_multiplicities=role_multiplicities,
        )
    if ir_grammar == "incremental_assignment_ir":
        return build_one_stage_incremental_assignment_ir_prompt(
            sentence=sentence,
            event_type=event_type,
            role_multiplicities=role_multiplicities,
        )
    if ir_grammar == "code4struct_ir":
        return build_one_stage_code4struct_ir_prompt(
            sentence=sentence,
            event_type=event_type,
            role_multiplicities=role_multiplicities,
        )
    msg = f"Unsupported IR grammar for one-stage prompt building: {ir_grammar}"
    raise ValueError(msg)


def build_one_stage_json_prompt(
    sentence: str,
    event_type: str,
    role_multiplicities: dict[str, int],
) -> PromptPair:
    """Build prompt for one-stage direct JSON generation."""
    event_line = f"Event type (reference): {event_type}\n" if event_type else ""
    multiplicity_line = ""
    if role_multiplicities:
        pairs = ", ".join(
            f"{role}={count}" for role, count in role_multiplicities.items()
        )
        multiplicity_line = f"Allowed roles and multiplicities: {pairs}\n"

    in_context_examples = build_one_stage_ir_in_context_examples(ir_grammar="json")

    system_prompt = (
        "Generate JSON event arguments from the given sentence.\n"
        "The trigger word(s) of the event is marked with **trigger word**.\n"
        "Output format:\n"
        '{"<role>":["<span1>","<span2>"]}\n'
        "\n"
        "Requirements:\n"
        "- Use only roles listed in 'Allowed roles and multiplicities'.\n"
        "- In 'Allowed roles and multiplicities', the number for each role indicates the maximum number of spans that can be included for that role.\n"
        "- Do not include more spans than the allowed number for each role.\n"
        "- Each JSON key is one role, and its value is a list of spans for that role.\n"
        "- Describe only arguments that are supported by the sentence.\n"
        "- If a role has no supported span, do NOT include that role in the output.\n"
        '- Do NOT output placeholders such as "none", "null", "not specified", or similar expressions.\n'
        "- Do NOT include any explanation, justification, or commentary.\n"
        "- Do not invent arguments or infer additional information beyond the sentence.\n"
        "- Copy text exactly from the sentence. Do not modify, paraphrase, or re-segment it.\n"
        "- Output only the JSON object. Do not include explanations or extra text.\n"
        "\n"
        f"{in_context_examples}\n"
    )

    user_prompt = f"{event_line}{multiplicity_line}Sentence: {sentence}\n"
    return system_prompt, user_prompt


def build_one_stage_incremental_assignment_ir_prompt(
    sentence: str,
    event_type: str,
    role_multiplicities: dict[str, int],
) -> PromptPair:
    """Build one-stage prompt for direct incremental-assignment IR generation."""
    event_line = f"Event type (reference): {event_type}\n" if event_type else ""
    multiplicity_line = ""
    if role_multiplicities:
        pairs = ", ".join(
            f"{role}={count}" for role, count in role_multiplicities.items()
        )
        multiplicity_line = f"Allowed roles and multiplicities: {pairs}\n"
    in_context_examples = build_one_stage_ir_in_context_examples(
        ir_grammar="incremental_assignment_ir"
    )
    system_prompt = (
        "Generate incremental-assignment IR from the given sentence.\n"
        "The trigger word(s) of the event is marked with **trigger word**.\n"
        "Output format:\n"
        "<role> += <span>\n"
        "\n"
        "Requirements:\n"
        "- Use only roles listed in 'Allowed roles and multiplicities'.\n"
        "- In 'Allowed roles and multiplicities', the number for each role indicates the maximum number of spans that can be included for that role.\n"
        "- Do not include more spans than the allowed number for each role.\n"
        "- Each output line assigns one argument value to one role.\n"
        "- Describe only arguments that are supported by the sentence.\n"
        "- If a role has no supported span, do NOT output that role.\n"
        '- Do NOT output placeholders such as "none", "null", "not specified", or similar expressions.\n'
        "- Do NOT include any explanation, justification, or commentary.\n"
        "- Do not invent arguments or infer additional information beyond the sentence.\n"
        "- Copy text exactly from the sentence. Do not modify, paraphrase, or re-segment it.\n"
        "- Output only the IR lines. Do not include explanations or extra text.\n"
        "\n"
        f"{in_context_examples}\n"
    )
    user_prompt = f"{event_line}{multiplicity_line}Sentence: {sentence}\n"
    return system_prompt, user_prompt


def build_one_stage_code4struct_ir_prompt(
    sentence: str,
    event_type: str,
    role_multiplicities: dict[str, int],
) -> PromptPair:
    """Build one-stage prompt for direct CODE4STRUCT-style IR generation."""
    event_class = _to_event_class_name(event_type)
    ontology_block = _build_code4struct_ontology_block(
        event_class=event_class,
        role_names=list(role_multiplicities.keys()) if role_multiplicities else [],
    )
    multiplicity_line = ""
    if role_multiplicities:
        pairs = ", ".join(
            f"{role}={count}" for role, count in role_multiplicities.items()
        )
        multiplicity_line = f"Argument value limits: {pairs}\n"

    task_block = (
        '"""\n'
        f"{multiplicity_line}"
        f"Convert the following sentence into an instance of {event_class}.\n"
        f'"{sentence}"\n'
        '"""\n'
        f"{event_class.lower()}_event = {event_class}(\n"
    )
    in_context_examples = build_one_stage_ir_in_context_examples(
        ir_grammar="code4struct_ir"
    )

    system_prompt = (
        "Complete the event instantiation shown in the prompt using the given sentence.\n"
        "The trigger word(s) of the event is marked with **trigger word**.\n"
        "\n"
        "Requirements:\n"
        "- Each argument corresponds to a field in the constructor.\n"
        "- In 'Argument value limits', the number for each argument indicates the maximum number of values that can be included for that argument.\n"
        "- Do not include more values than the allowed number for each argument.\n"
        "- Describe only arguments that are supported by the sentence.\n"
        "- Do not invent arguments or infer additional information beyond the sentence.\n"
        "- Copy text exactly from the sentence. Do not modify, paraphrase, or re-segment it.\n"
        "- Output only the completion of the event instantiation. Do not repeat the prefix or include explanations.\n"
        f"{in_context_examples}\n"
    )

    user_prompt = f"{ontology_block}\n\n{task_block}"
    return system_prompt, user_prompt


def build_two_stage_extraction_prompt(
    sentence: str,
    event_type: str,
    role_multiplicities: dict[str, int],
) -> PromptPair:
    """Build prompt for two-stage extraction step."""
    event_line = f"Event type (reference): {event_type}\n" if event_type else ""
    multiplicity_line = ""
    if role_multiplicities:
        pairs = ", ".join(
            f"{role}={count}" for role, count in role_multiplicities.items()
        )
        multiplicity_line = f"Allowed roles and multiplicities: {pairs}\n"

    system_prompt = (
        "Identify the event arguments related to the marked trigger word in the following sentence.\n"
        "The trigger word(s) of the event is marked with **trigger word**.\n\n"
        "Identify argument spans that are explicitly supported by the sentence and describe the role of each span.\n\n"
        "Requirements:\n"
        "- Use only roles listed in 'Allowed roles and multiplicities'.\n"
        "- In 'Allowed roles and multiplicities', the number for each role indicates the maximum number of spans that can be included for that role.\n"
        "- Do not include more spans than the allowed number for each role.\n"
        "- Describe only arguments that are supported by the sentence.\n"
        "- Do not invent arguments or infer additional information beyond the sentence.\n"
        "- Copy text exactly from the sentence. Do not modify, paraphrase, or re-segment it.\n"
        "- Do not split one text span into multiple arguments unless the sentence clearly supports that.\n"
        "- Do not organize the answer into a table, JSON, key-value pairs, role-label lines, or any other fixed schema.\n"
        "- Keep the response concise, in free-form natural language only, and without extended explanations.\n\n"
    )

    user_prompt = f"{event_line}{multiplicity_line}Sentence: {sentence}\n"
    return system_prompt, user_prompt


def build_two_stage_ir_prompt(
    extraction_text: str,
    event_type: str,
    ir_grammar: str,
    role_multiplicities: dict[str, int],
) -> PromptPair:
    """Build two-stage structured-generation prompt by selected grammar."""
    if ir_grammar == "json":
        return build_two_stage_json_prompt(
            extraction_text=extraction_text,
            role_multiplicities=role_multiplicities,
        )
    if ir_grammar == "incremental_assignment_ir":
        return build_two_stage_incremental_assignment_ir_prompt(
            extraction_text=extraction_text,
            role_multiplicities=role_multiplicities,
        )
    if ir_grammar == "code4struct_ir":
        return build_two_stage_code4struct_ir_prompt(
            extraction_text=extraction_text,
            event_type=event_type,
            role_multiplicities=role_multiplicities,
        )
    msg = f"Unsupported IR grammar for two-stage prompt building: {ir_grammar}"
    raise ValueError(msg)


def build_two_stage_json_prompt(
    extraction_text: str,
    role_multiplicities: dict[str, int],
) -> PromptPair:
    """Build prompt for two-stage extraction->JSON generation."""
    multiplicity_line = ""
    if role_multiplicities:
        pairs = ", ".join(
            f"{role}={count}" for role, count in role_multiplicities.items()
        )
        multiplicity_line = f"Allowed roles and multiplicities: {pairs}\n"

    in_context_examples = build_two_stage_ir_in_context_examples(ir_grammar="json")

    system_prompt = (
        "Generate JSON event arguments from the given extraction notes.\n"
        "Output format:\n"
        '{"<role>":["<span1>","<span2>"]}\n'
        "\n"
        "Requirements:\n"
        "- Each JSON key is one role, and its value is a list of spans for that role.\n"
        "- Use only roles listed in 'Allowed roles and multiplicities'.\n"
        "- In 'Allowed roles and multiplicities', the number for each role indicates the maximum number of spans that can be included for that role.\n"
        "- Do not include more spans than the allowed number for each role.\n"
        "- Use only text explicitly supported by the extraction notes.\n"
        "- If a role has no supported span, do NOT include that role in the output.\n"
        '- Do NOT output placeholders such as "none", "null", "not specified", or similar expressions.\n'
        "- Do NOT include any explanation, justification, or commentary.\n"
        "- Copy text exactly from the extraction notes. Do not modify, paraphrase, or re-segment it.\n"
        "- Output only the JSON object. Do not include explanations or extra text.\n"
        "\n"
        f"{in_context_examples}\n"
    )

    user_prompt = f"{multiplicity_line}Extraction notes:\n{extraction_text}"
    return system_prompt, user_prompt


def build_two_stage_incremental_assignment_ir_prompt(
    extraction_text: str,
    role_multiplicities: dict[str, int],
) -> PromptPair:
    """Build two-stage prompt for incremental-assignment IR generation."""
    multiplicity_line = ""
    if role_multiplicities:
        pairs = ", ".join(
            f"{role}={count}" for role, count in role_multiplicities.items()
        )
        multiplicity_line = f"Allowed roles and multiplicities: {pairs}\n"

    in_context_examples = build_two_stage_ir_in_context_examples(
        ir_grammar="incremental_assignment_ir"
    )

    system_prompt = (
        "Generate incremental-assignment IR from the given extraction notes.\n"
        "Output format:\n"
        "<role> += <span>\n"
        "\n"
        "Requirements:\n"
        "- Each output line assigns one argument value to one role.\n"
        "- Use only roles listed in 'Allowed roles and multiplicities'.\n"
        "- In 'Allowed roles and multiplicities', the number for each role indicates the maximum number of spans that can be included for that role.\n"
        "- Do not include more spans than the allowed number for each role.\n"
        "- Use only text explicitly supported by the extraction notes.\n"
        "- If a role has no supported span, do NOT output that role.\n"
        '- Do NOT output placeholders such as "none", "null", "not specified", or similar expressions.\n'
        "- Do NOT include any explanation, justification, or commentary.\n"
        "- Copy text exactly from the extraction notes. Do not modify, paraphrase, or re-segment it.\n"
        "- Output only the IR lines. Do not include explanations or extra text.\n"
        "\n"
        f"{in_context_examples}\n"
    )
    user_prompt = f"{multiplicity_line}Extraction notes:\n{extraction_text}"
    return system_prompt, user_prompt


def build_two_stage_code4struct_ir_prompt(
    extraction_text: str,
    event_type: str,
    role_multiplicities: dict[str, int],
) -> PromptPair:
    """Build two-stage prompt for CODE4STRUCT-style IR generation."""
    event_class = _to_event_class_name(event_type)
    ontology_block = _build_code4struct_ontology_block(
        event_class=event_class,
        role_names=list(role_multiplicities.keys()) if role_multiplicities else [],
    )
    in_context_examples = build_two_stage_ir_in_context_examples(
        ir_grammar="code4struct_ir"
    )
    multiplicity_line = ""
    if role_multiplicities:
        pairs = ", ".join(
            f"{role}={count}" for role, count in role_multiplicities.items()
        )
        multiplicity_line = f"Argument value limits: {pairs}\n"
    task_block = (
        '"""\n'
        f"{multiplicity_line}"
        f"Convert the following extraction notes into an instance of {event_class}.\n"
        f'"{extraction_text}"\n'
        '"""\n'
        f"{event_class.lower()}_event = {event_class}(\n"
    )

    system_prompt = (
        "Complete the event instantiation shown in the prompt using the given extraction notes.\n"
        "\n"
        "Requirements:\n"
        "- Each argument corresponds to a field in the constructor.\n"
        "- In 'Argument value limits', the number for each argument indicates the maximum number of values that can be included for that argument.\n"
        "- Do not include more values than the allowed number for each argument.\n"
        "- Use only text explicitly supported by the extraction notes.\n"
        "- Copy text exactly from the extraction notes. Do not modify, paraphrase, or re-segment it.\n"
        "- Output only the completion of the event instantiation. Do not repeat the prefix or include explanations.\n"
        "\n"
        f"{in_context_examples}\n"
    )
    user_prompt = f"{ontology_block}\n\n{task_block}"
    return system_prompt, user_prompt


def _build_code4struct_ontology_block(
    event_class: str,
    role_names: list[str],
) -> str:
    """Build CODE4STRUCT ontology block without entity/event docstrings."""
    role_identifiers = [_role_to_identifier(role) for role in role_names]
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
        "    def __init__(self, name: str):\n"
        "        self.name = name\n\n"
        f"class {event_class}(Event):\n"
        "    def __init__(\n"
        "        self,\n"
        f"{role_args}\n"
        "    ):\n"
        f"{role_assignments}\n"
    )


def _to_event_class_name(event_type: str) -> str:
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
