"""CODE4STRUCT-style IR grammar parser and validator.

Reference:
    @inproceedings{wang-etal-2023-code4struct,
        title = "{C}ode4{S}truct: Code Generation for Few-Shot Event Structure Prediction",
        author = "Wang, Xingyao  and Li, Sha  and Ji, Heng",
        editor = "Rogers, Anna  and Boyd-Graber, Jordan  and Okazaki, Naoaki",
        booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
        month = jul,
        year = "2023",
        address = "Toronto, Canada",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/2023.acl-long.202/",
        doi = "10.18653/v1/2023.acl-long.202",
        pages = "3640--3663",
    }

Implementation note:
    This module implements the code-style structured representation idea inspired
    by the CODE4STRUCT methodology. The implementation was written independently
    for this thesis project and does not copy source code from the original paper
    or its repository.
"""

from __future__ import annotations

import ast
import re

from ntust_thesis.ir.common import IRGrammarValidator, IRValidationResult

_INSTANCE_HEADER_RE = re.compile(r"^\s*[A-Za-z_]\w*\s*=\s*[A-Za-z_]\w*\s*\(\s*$")


class Code4StructIRValidator(IRGrammarValidator):
    """Validate CODE4STRUCT-style completion text."""

    def name(self) -> str:
        """Return grammar identifier."""
        return "code4struct_ir"

    def validate(self, ir_text: str) -> IRValidationResult:
        """Validate CODE4STRUCT-style IR text."""
        if not ir_text.strip():
            return IRValidationResult(
                is_valid=False,
                error_message="Empty CODE4STRUCT IR is invalid.",
            )

        try:
            _parse_code4struct_ir_with_errors(ir_text)
        except (ValueError, TypeError) as exc:
            line_no, line_text, message = _decode_parse_error(str(exc), ir_text)
            return IRValidationResult(
                is_valid=False,
                error_message=message,
                error_line_no=line_no,
                error_line_text=line_text,
            )
        return IRValidationResult(is_valid=True)


def parse_code4struct_ir(ir_text: str) -> list[tuple[str, str]]:
    """Parse CODE4STRUCT IR into (role_path, span_text) pairs."""
    return _parse_code4struct_ir_with_errors(ir_text)


def _parse_code4struct_ir_with_errors(ir_text: str) -> list[tuple[str, str]]:
    """Internal parser that raises explicit ValueError on invalid format."""
    lines = ir_text.splitlines()
    non_empty_lines = _collect_non_empty_lines(lines)
    if not non_empty_lines:
        return []
    _validate_completion_only_format(non_empty_lines)
    return _parse_argument_block_with_ast(lines, non_empty_lines[-1][0])


def _collect_non_empty_lines(lines: list[str]) -> list[tuple[int, str]]:
    """Collect non-empty source lines with original line numbers."""
    return [
        (line_no, line.rstrip())
        for line_no, line in enumerate(lines, start=1)
        if line.strip()
    ]


def _validate_completion_only_format(non_empty_lines: list[tuple[int, str]]) -> None:
    """Validate strict completion-only format."""
    first_line_no, first_line = non_empty_lines[0]
    if _looks_like_instance_header(first_line):
        msg = (
            f"line:{first_line_no}|Do not output instance prefix "
            "(e.g., '<var> = <EventClass>('). Output completion only."
        )
        raise ValueError(msg)

    _validate_closing_line(non_empty_lines[-1])


def _looks_like_instance_header(line: str) -> bool:
    """Return whether line looks like '<var> = <EventClass>(' header."""
    return _INSTANCE_HEADER_RE.fullmatch(line) is not None


def _validate_closing_line(line_with_no: tuple[int, str]) -> None:
    """Ensure the final body line is exactly a closing parenthesis."""
    closing_line_no, closing_line = line_with_no
    if closing_line.strip() == ")":
        return
    msg = f"line:{closing_line_no}|Missing closing ')'."
    raise ValueError(msg)


def _parse_argument_block_with_ast(
    lines: list[str],
    closing_line_no: int,
) -> list[tuple[str, str]]:
    """Parse completion argument block as Python AST."""
    argument_block = "\n".join(lines[: closing_line_no - 1])
    if not argument_block.strip():
        return []

    synthesized = f"__instance = __Event(\n{argument_block}\n)\n"
    try:
        tree = ast.parse(synthesized)
    except SyntaxError as exc:
        mapped_line_no = _map_synthesized_line_to_input(exc.lineno)
        msg = f"line:{mapped_line_no}|Invalid Python syntax in argument block."
        raise ValueError(msg) from exc

    call = _extract_single_call_node(tree)
    return _extract_role_spans_from_call(call)


def _map_synthesized_line_to_input(synth_line_no: int | None) -> int:
    """Map synthesized source line number back to original input line."""
    if synth_line_no is None:
        return 1
    if synth_line_no <= 1:
        return 1
    return synth_line_no - 1


def _extract_single_call_node(tree: ast.AST) -> ast.Call:
    """Extract the synthesized call node from AST tree."""
    if not isinstance(tree, ast.Module) or len(tree.body) != 1:
        msg = "line:1|Invalid argument block structure."
        raise ValueError(msg)

    stmt = tree.body[0]
    if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
        msg = "line:1|Invalid argument block structure."
        raise ValueError(msg)

    if not isinstance(stmt.value, ast.Call):
        msg = "line:1|Invalid argument block structure."
        raise TypeError(msg)

    return stmt.value


def _extract_role_spans_from_call(call: ast.Call) -> list[tuple[str, str]]:
    """Extract (role_path, span_text) pairs from call keywords."""
    if call.args:
        line_no = getattr(call.args[0], "lineno", 1)
        msg = f"line:{line_no}|Positional arguments are not allowed."
        raise ValueError(msg)

    role_spans: list[tuple[str, str]] = []
    for keyword in call.keywords:
        if keyword.arg is None:
            line_no = getattr(keyword, "lineno", 1)
            msg = f"line:{line_no}|Dictionary unpacking is not allowed."
            raise ValueError(msg)
        role_name = _arg_identifier_to_role_path(keyword.arg.strip())
        role_spans.extend(
            _extract_spans_from_keyword_value(role_name=role_name, value=keyword.value)
        )
    return role_spans


def _extract_spans_from_keyword_value(
    role_name: str,
    value: ast.AST,
) -> list[tuple[str, str]]:
    """Extract spans from one role assignment value."""
    line_no = getattr(value, "lineno", 1)
    if not isinstance(value, ast.List):
        msg = f"line:{line_no}|Role value must be a list."
        raise TypeError(msg)

    role_spans: list[tuple[str, str]] = []
    for item in value.elts:
        span_text = _parse_entity_item(item)
        role_spans.append((role_name, span_text))
    return role_spans


def _parse_entity_item(item: ast.AST) -> str:
    """Parse one Entity('...') item into plain span text."""
    line_no = getattr(item, "lineno", 1)
    if not isinstance(item, ast.Call):
        msg = f'line:{line_no}|List items must be Entity("...").'
        raise TypeError(msg)

    if not isinstance(item.func, ast.Name) or item.func.id != "Entity":
        msg = f'line:{line_no}|List items must call Entity("...").'
        raise ValueError(msg)

    if len(item.args) != 1 or item.keywords:
        msg = f"line:{line_no}|Entity must take exactly one string argument."
        raise ValueError(msg)

    arg = item.args[0]
    if not isinstance(arg, ast.Constant) or not isinstance(arg.value, str):
        msg = f"line:{line_no}|Entity argument must be a string."
        raise TypeError(msg)

    span_text = arg.value.strip()
    if not span_text:
        msg = f"line:{line_no}|Entity argument must be non-empty."
        raise ValueError(msg)
    return span_text


def _decode_parse_error(
    message: str, ir_text: str
) -> tuple[int | None, str | None, str]:
    """Decode parser ValueError text into (line_no, line_text, message)."""
    match = re.match(r"line:(\d+)\|(.*)", message)
    if match is None:
        return None, None, message
    line_no = int(match.group(1))
    lines = ir_text.splitlines()
    line_text = lines[line_no - 1] if 1 <= line_no <= len(lines) else None
    return line_no, line_text, match.group(2).strip()


def _arg_identifier_to_role_path(arg_name: str) -> str:
    """Convert constructor argument identifier back to role path."""
    return arg_name.replace("__", ".")
