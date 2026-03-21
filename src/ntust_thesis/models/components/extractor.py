"""Extraction stage components for IR pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ntust_thesis.core.interfaces import LLMClient


class Extractor:
    """Produce extraction text from raw input text."""

    def extract(self, input_text: str) -> str:
        """Return extraction stage output text."""
        raise NotImplementedError


class MockExtractor(Extractor):
    """Deterministic mock extractor for local tests."""

    def extract(self, input_text: str) -> str:
        """Create simple extraction text from keywords."""
        lower = input_text.lower()
        if "attack" in lower or "attacked" in lower:
            event_type = "conflict.attack.selfdirectedbattle"
        elif "killed" in lower or "death" in lower or "die" in lower:
            event_type = "life.die.deathcausedbyviolentevents"
        else:
            event_type = "unknown.event"
        return f"event_type: {event_type}\narguments: none"


class GeminiExtractor(Extractor):
    """Gemini-based extraction stage."""

    def __init__(self, llm: LLMClient, temperature: float) -> None:
        """Initialize extractor with LLM backend."""
        self._llm = llm
        self._temperature = temperature

    def extract(self, input_text: str) -> str:
        """Ask Gemini to produce extraction notes."""
        prompt = (
            "Extract event information from the text. "
            "Return concise plain text with two sections:\n"
            "1) event_type: <event type>\n"
            "2) arguments: one argument per line as '<role>: <text>' or 'none'.\n"
            "Do not return JSON.\n"
            f"Text: {input_text}"
        )
        return self._llm.generate(prompt, temperature=self._temperature)
