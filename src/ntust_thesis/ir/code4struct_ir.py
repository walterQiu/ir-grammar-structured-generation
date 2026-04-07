"""CODE4STRUCT-style IR grammar parser and validator."""

from __future__ import annotations

import re

from ntust_thesis.ir.common import IRGrammarValidator, IRValidationResult

_HEADER_RE = re.compile(r"^\s*[A-Za-z_]\w*\s*=\s*[A-Za-z_]\w*\s*\(\s*$")
_ARG_LINE_WITH_COMMA_RE = re.compile(
    r"^\s*(?P<role>[A-Za-z_]\w*)\s*=\s*\[(?P<items>.*)\]\s*,\s*$"
)
_ARG_LINE_NO_COMMA_RE = re.compile(
    r"^\s*(?P<role>[A-Za-z_]\w*)\s*=\s*\[(?P<items>.*)\]\s*$"
)
_SPAN_ITEM_RE = re.compile(
    r"""
    Entity\s*\(\s*                       # required constructor Entity(
    "(?P<d>(?:\\.|[^"\\])*)"          # double-quoted content
    \s*\)                               # close paren
    |
    Entity\s*\(\s*                       # required constructor Entity(
    '(?P<s>(?:\\.|[^'\\])*)'          # single-quoted content
    \s*\)                               # close paren
    """,
    re.VERBOSE,
)


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
        except ValueError as exc:
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
    non_empty_lines = _collect_non_empty_lines(ir_text)
    if not non_empty_lines:
        return []
    body_lines = _extract_validated_body_lines(non_empty_lines)
    return _parse_argument_lines(body_lines)


def _collect_non_empty_lines(ir_text: str) -> list[tuple[int, str]]:
    """Collect non-empty source lines with original line numbers."""
    return [
        (line_no, line.rstrip())
        for line_no, line in enumerate(ir_text.splitlines(), start=1)
        if line.strip()
    ]


def _extract_validated_body_lines(
    non_empty_lines: list[tuple[int, str]],
) -> list[tuple[int, str]]:
    """Validate header and closing line, then return body lines."""
    _validate_header_line(non_empty_lines[0])
    body_lines = non_empty_lines[1:]
    if not body_lines:
        header_line_no = non_empty_lines[0][0]
        msg = (
            f"line:{header_line_no}|Incomplete CODE4STRUCT instance: "
            "expected at least a closing ')' after header."
        )
        raise ValueError(msg)
    _validate_closing_line(body_lines[-1])
    return body_lines


def _validate_header_line(line_with_no: tuple[int, str]) -> None:
    """Ensure the first non-empty line is a CODE4STRUCT header."""
    header_line_no, header_line = line_with_no
    if _HEADER_RE.fullmatch(header_line) is not None:
        return
    msg = f"line:{header_line_no}|First non-empty line must be '<var> = <EventClass>('."
    raise ValueError(msg)


def _validate_closing_line(line_with_no: tuple[int, str]) -> None:
    """Ensure the final body line is exactly a closing parenthesis."""
    closing_line_no, closing_line = line_with_no
    if closing_line.strip() == ")":
        return
    msg = f"line:{closing_line_no}|Missing closing ')'."
    raise ValueError(msg)


def _parse_argument_lines(body_lines: list[tuple[int, str]]) -> list[tuple[str, str]]:
    """Parse all argument lines (excluding final closing line)."""
    role_spans: list[tuple[str, str]] = []
    argument_lines = body_lines[:-1]
    for arg_idx, (offset, raw_line) in enumerate(argument_lines):
        is_last_arg_line = arg_idx == len(argument_lines) - 1
        role_spans.extend(
            _parse_argument_line(
                line_no=offset,
                raw_line=raw_line,
                is_last_arg_line=is_last_arg_line,
            )
        )
    return role_spans


def _parse_argument_line(
    line_no: int,
    raw_line: str,
    *,
    is_last_arg_line: bool,
) -> list[tuple[str, str]]:
    """Parse one argument assignment line into role-span pairs."""
    line = raw_line.strip()
    if not line:
        return []
    match = _ARG_LINE_WITH_COMMA_RE.fullmatch(line)
    if match is None and is_last_arg_line:
        match = _ARG_LINE_NO_COMMA_RE.fullmatch(line)
    if match is None:
        if is_last_arg_line:
            msg = f"line:{line_no}|Invalid argument assignment format."
        else:
            msg = (
                f"line:{line_no}|Invalid argument assignment format "
                "(non-final line must end with ',')."
            )
        raise ValueError(msg)

    role_name = _arg_identifier_to_role_path(match.group("role").strip())
    list_body = match.group("items").strip()
    if not list_body:
        return []

    spans = _extract_spans_from_list_body(list_body)
    if not spans:
        msg = f"line:{line_no}|Invalid list items format."
        raise ValueError(msg)
    return [(role_name, span_text) for span_text in spans]


def _extract_spans_from_list_body(list_body: str) -> list[str]:
    """Extract spans from list literal body with required Entity constructors."""
    spans: list[str] = []
    consumed_ranges: list[tuple[int, int]] = []
    for match in _SPAN_ITEM_RE.finditer(list_body):
        raw = match.group("d") if match.group("d") is not None else match.group("s")
        if raw is None:
            continue
        text = _unescape(raw).strip()
        if text:
            spans.append(text)
            consumed_ranges.append((match.start(), match.end()))
    if not spans:
        return []

    # check if there aren't any content left
    leftovers = []
    cursor = 0
    for start, end in consumed_ranges:
        if cursor < start:
            leftovers.append(list_body[cursor:start])
        cursor = max(cursor, end)
    if cursor < len(list_body):
        leftovers.append(list_body[cursor:])

    leftover_text = "".join(leftovers).replace(",", "").strip()
    if leftover_text:
        return []

    return spans


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


def _unescape(text: str) -> str:
    """Unescape Python-like string content."""
    try:
        return bytes(text, "utf-8").decode("unicode_escape")
    except Exception:
        return text


def _arg_identifier_to_role_path(arg_name: str) -> str:
    """Convert constructor argument identifier back to role path."""
    return arg_name.replace("__", ".")
