"""IR generation stage components."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ntust_thesis.core.interfaces import LLMClient


class IRGenerator:
    """Convert extraction text into dot-notation IR."""

    def generate(self, extraction_text: str) -> str:
        """Generate IR text from extraction text."""
        raise NotImplementedError


class MockIRGenerator(IRGenerator):
    """Deterministic mock IR generator."""

    def generate(self, extraction_text: str) -> str:
        """Convert 'event_type: X' to minimal IR."""
        event_type = "unknown.event"
        for line in extraction_text.splitlines():
            stripped = line.strip()
            if stripped.startswith("event_type:"):
                event_type = stripped.split(":", 1)[1].strip()
                break
        return f"event.type = {event_type}"


class GeminiIRGenerator(IRGenerator):
    """Gemini-based IR generation stage."""

    def __init__(self, llm: LLMClient, temperature: float) -> None:
        """Initialize IR generator with LLM backend."""
        self._llm = llm
        self._temperature = temperature

    def generate(self, extraction_text: str) -> str:
        """Generate strict dot-notation IR lines."""
        # TODO: optimize prompt
        prompt = (
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
        return self._llm.generate(prompt, temperature=self._temperature)
