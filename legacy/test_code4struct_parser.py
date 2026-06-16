"""Test whether a hardcoded string can pass code4struct IR parsing/compilation."""

from __future__ import annotations

import json

from ntust_thesis.models.components.ir_compiler import DeterministicIRCompiler

# Edit these values directly before running this script.
INPUT_MODE = "json_payload"  # one of: ir_text, json_payload
INPUT_TEXT = r"spy=[Entity(\"Russia\"),],\nobservedentity=[Entity(\"Clinton\"),],\n)"
JSON_PAYLOAD = r'{"ir_text": "spy=[Entity(\"Russia\"),],\nobservedentity=[Entity(\"Clinton\"),],\n)"}'
DECODE_ESCAPES = True
EVENT_TYPE = "debug.event"


def _read_input_text() -> str:
    """Resolve configured source into IR text."""
    if INPUT_MODE == "ir_text":
        return INPUT_TEXT
    if INPUT_MODE == "json_payload":
        payload = json.loads(JSON_PAYLOAD)
        ir_text = payload.get("ir_text")
        if not isinstance(ir_text, str):
            msg = "JSON payload must contain string field 'ir_text'."
            raise ValueError(msg)
        return ir_text

    msg = f"Unsupported INPUT_MODE: {INPUT_MODE}"
    raise ValueError(msg)


def _decode_escapes_if_needed(text: str, decode_escapes: bool) -> str:
    """Optionally decode escape sequences into actual characters."""
    if not decode_escapes:
        return text
    return bytes(text, "utf-8").decode("unicode_escape")


def main() -> int:
    """Run parsing/compilation test."""
    try:
        original_text = _read_input_text()
        ir_text = _decode_escapes_if_needed(original_text, DECODE_ESCAPES)
        compiler = DeterministicIRCompiler(ir_grammar="code4struct_ir")
        compiled = compiler.compile(
            ir_text=ir_text,
            event_type=EVENT_TYPE,
        )
    except Exception as exc:
        print("RESULT: FAIL")  # noqa: T201
        print(f"ERROR: {exc}")  # noqa: T201
        return 1

    print("RESULT: PASS")  # noqa: T201
    print(json.dumps(compiled.model_dump(), ensure_ascii=False, indent=2))  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
