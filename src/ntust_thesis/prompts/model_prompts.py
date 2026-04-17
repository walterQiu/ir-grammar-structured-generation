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

    # (1) Instruction block
    instruction_block = (
        "You are given a sentence describing an event.\n"
        "Extract event arguments and represent them in JSON format.\n"
    )

    # (2) Output format block
    output_format_block = (
        "Output format:\n"
        "{\n"
        '  "<role1>": ["<span1>", "<span2>"],\n'
        '  "<role2>": ["<span3>"]\n'
        "}\n"
    )

    # (3) Output rules block
    output_rules_block = (
        "Rules:\n"
        "- Use only roles listed in 'Allowed roles and multiplicities'.\n"
        "- The number for each role indicates the maximum number of spans allowed.\n"
        "- Include a role only if at least one valid span exists.\n"
        "- If no valid role-span pairs exist, output an empty JSON object: {}.\n"
        "- Copy spans exactly from the sentence. Do not modify or paraphrase.\n"
        "- Do not infer or hallucinate information not supported by the sentence.\n"
        "- Do not output placeholders such as 'none', 'null', 'not specified', or similar.\n"
        "- Output only the JSON object. Do not include any explanation or extra text.\n"
    )

    # (4) ICL block
    icl_block = f"{in_context_examples}\n"

    system_prompt = (
        instruction_block
        + "\n"
        + output_format_block
        + "\n"
        + output_rules_block
        + "\n"
        + icl_block
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

    # (1) Instruction block
    instruction_block = (
        "You are given a sentence describing an event.\n"
        "Extract event arguments and represent them using incremental assignment.\n"
    )

    # (2) Output format block
    output_format_block = (
        "Output format:\n<role1> += <span1>\n<role1> += <span2>\n<role2> += <span3>\n"
    )

    # (3) Output rules block
    output_rules_block = (
        "Rules:\n"
        "- Use only roles listed in 'Allowed roles and multiplicities'.\n"
        "- The number for each role indicates the maximum number of spans allowed.\n"
        "- Output one line per span using the format '<role> += <span>'.\n"
        "- If a role has multiple spans, output multiple lines for that role.\n"
        "- Output a role only if at least one valid span exists.\n"
        "- If no valid role-span pairs exist, output nothing.\n"
        "- Copy spans exactly from the sentence. Do not modify or paraphrase.\n"
        "- Do not infer or hallucinate information not supported by the sentence.\n"
        "- Do not output placeholders such as 'none', 'null', 'not specified', or similar.\n"
        "- Output only the IR lines. Do not include any explanation or extra text.\n"
    )

    # (4) ICL block
    icl_block = f"{in_context_examples}\n"

    system_prompt = (
        instruction_block
        + "\n"
        + output_format_block
        + "\n"
        + output_rules_block
        + "\n"
        + icl_block
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
        f"{sentence}\n"
        '"""\n'
        f"{event_class.lower()}_event = {event_class}(\n"
    )

    in_context_examples = build_one_stage_ir_in_context_examples(
        ir_grammar="code4struct_ir"
    )

    # (1) Instruction block
    instruction_block = (
        "You are given a sentence describing an event.\n"
        "Complete a Python class instantiation that represents the event arguments.\n"
    )

    # (2) Output format block
    output_format_block = (
        "Output format:\n"
        '    <argument1>=[Entity("<value1>"), Entity("<value2>")],\n'
        '    <argument2>=[Entity("<value3>")],\n'
        ")\n"
    )

    # (3) Output rules block
    output_rules_block = (
        "Rules:\n"
        "- Use only arguments defined in the class constructor.\n"
        "- The number for each argument in 'Argument value limits' indicates the maximum number of values allowed.\n"
        "- Include an argument only if at least one valid value exists.\n"
        "- If no valid arguments exist, complete the empty instantiation with no fields.\n"
        "- Copy values exactly from the sentence. Do not modify or paraphrase.\n"
        "- Do not infer or hallucinate information not supported by the sentence.\n"
        "- Do not output placeholders such as 'none', 'null', 'not specified', or similar.\n"
        "- Output only the completion inside the parentheses. Do not repeat the prefix or include explanations.\n"
    )

    # (4) ICL block
    icl_block = f"{in_context_examples}\n"

    system_prompt = (
        instruction_block
        + "\n"
        + output_format_block
        + "\n"
        + output_rules_block
        + "\n"
        + icl_block
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

    # (1) Instruction block
    instruction_block = (
        "You are given a sentence describing an event.\n"
        "Identify the event arguments related to the marked trigger word and describe them in concise natural language.\n"
        "The trigger word(s) of the event is marked with **trigger word**.\n"
    )

    # (3) Output rules block
    output_rules_block = (
        "Rules:\n"
        "- Use only roles listed in 'Allowed roles and multiplicities'.\n"
        "- The number for each role indicates the maximum number of spans allowed.\n"
        "- Describe only arguments that are explicitly supported by the sentence.\n"
        "- Do not infer or hallucinate information beyond the sentence.\n"
        "- Copy argument text exactly from the sentence. Do not modify, paraphrase, or re-segment it.\n"
        "- Do not split one text span into multiple arguments unless the sentence clearly supports that.\n"
        "- If a role has no supported argument, do not mention that role.\n"
        "- Write the answer in concise free-form natural language.\n"
        "- Do not output JSON, code, tables, key-value pairs, role-label lines, or any other fixed schema.\n"
        "- Do not include extended explanations, justifications, or commentary.\n"
    )

    system_prompt = instruction_block + "\n" + output_rules_block + "\n"

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

    # (1) Instruction block
    instruction_block = (
        "You are given extraction notes describing event arguments.\n"
        "Organize the supported arguments into JSON format.\n"
    )

    # (2) Output format block
    output_format_block = (
        "Output format:\n"
        "{\n"
        '  "<role1>": ["<span1>", "<span2>"],\n'
        '  "<role2>": ["<span3>"]\n'
        "}\n"
    )

    # (3) Output rules block
    output_rules_block = (
        "Rules:\n"
        "- Use only roles listed in 'Allowed roles and multiplicities'.\n"
        "- The number for each role indicates the maximum number of spans allowed.\n"
        "- Include a role only if at least one valid span is explicitly supported by the extraction notes.\n"
        "- If no valid role-span pairs are supported by the extraction notes, output an empty JSON object: {}.\n"
        "- Use only text explicitly supported by the extraction notes.\n"
        "- Copy spans exactly from the extraction notes. Do not modify, paraphrase, or re-segment them.\n"
        "- Do not infer or hallucinate information beyond the extraction notes.\n"
        "- Do not output placeholders such as 'none', 'null', 'not specified', or similar.\n"
        "- Output only the JSON object. Do not include any explanation or extra text.\n"
    )

    # (4) ICL block
    icl_block = f"{in_context_examples}\n"

    system_prompt = (
        instruction_block
        + "\n"
        + output_format_block
        + "\n"
        + output_rules_block
        + "\n"
        + icl_block
    )

    user_prompt = f"{multiplicity_line}Extraction notes:\n{extraction_text}\n"

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

    # (1) Instruction block
    instruction_block = (
        "You are given extraction notes describing event arguments.\n"
        "Organize the supported arguments using incremental assignment.\n"
    )

    # (2) Output format block
    output_format_block = (
        "Output format:\n<role1> += <span1>\n<role1> += <span2>\n<role2> += <span3>\n"
    )

    # (3) Output rules block
    output_rules_block = (
        "Rules:\n"
        "- Use only roles listed in 'Allowed roles and multiplicities'.\n"
        "- The number for each role indicates the maximum number of spans allowed.\n"
        "- Output one line per span using the format '<role> += <span>'.\n"
        "- If a role has multiple spans, output multiple lines for that role.\n"
        "- Output a role only if at least one valid span is explicitly supported by the extraction notes.\n"
        "- If no valid role-span pairs are supported by the extraction notes, output nothing.\n"
        "- Use only text explicitly supported by the extraction notes.\n"
        "- Copy spans exactly from the extraction notes. Do not modify, paraphrase, or re-segment them.\n"
        "- Do not infer or hallucinate information beyond the extraction notes.\n"
        "- Do not output placeholders such as 'none', 'null', 'not specified', or similar.\n"
        "- Output only the IR lines. Do not include any explanation or extra text.\n"
    )

    # (4) ICL block
    icl_block = f"{in_context_examples}\n"

    system_prompt = (
        instruction_block
        + "\n"
        + output_format_block
        + "\n"
        + output_rules_block
        + "\n"
        + icl_block
    )

    user_prompt = f"{multiplicity_line}Extraction notes:\n{extraction_text}\n"

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
        f"{extraction_text}\n"
        '"""\n'
        f"{event_class.lower()}_event = {event_class}(\n"
    )

    # (1) Instruction block
    instruction_block = (
        "You are given extraction notes describing event arguments.\n"
        "Complete a Python class instantiation that organizes the supported arguments.\n"
    )

    # (2) Output format block
    output_format_block = (
        "Output format:\n"
        '    <argument1>=[Entity("<value1>"), Entity("<value2>")],\n'
        '    <argument2>=[Entity("<value3>")],\n'
        ")\n"
    )

    # (3) Output rules block
    output_rules_block = (
        "Rules:\n"
        "- Use only arguments defined in the class constructor.\n"
        "- The number for each argument in 'Argument value limits' indicates the maximum number of values allowed.\n"
        "- Include an argument only if at least one valid value is explicitly supported by the extraction notes.\n"
        "- If no valid arguments are supported by the extraction notes, complete the empty instantiation with no fields.\n"
        "- Use only text explicitly supported by the extraction notes.\n"
        "- Copy values exactly from the extraction notes. Do not modify, paraphrase, or re-segment them.\n"
        "- Do not infer or hallucinate information beyond the extraction notes.\n"
        "- Do not output placeholders such as 'none', 'null', 'not specified', or similar.\n"
        "- Output only the completion inside the parentheses. Do not repeat the prefix or include explanations.\n"
    )

    # (4) ICL block
    icl_block = f"{in_context_examples}\n"

    system_prompt = (
        instruction_block
        + "\n"
        + output_format_block
        + "\n"
        + output_rules_block
        + "\n"
        + icl_block
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
