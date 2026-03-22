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


class GeminiExtractor(Extractor):
    """Gemini-based extraction stage."""

    def __init__(self, llm: LLMClient, temperature: float) -> None:
        """Initialize extractor with LLM backend."""
        self._llm = llm
        self._temperature = temperature

    def extract(self, input_text: str) -> str:
        """Ask Gemini to produce extraction notes."""
        # TODO: optimize prompt
        prompt = (
            "Extract event information from the text. "
            "Return concise plain text with two sections:\n"
            "1) event_type: <event type>\n"
            "2) arguments: one argument per line as '<role>: <text>' or 'none'.\n"
            "Do not return JSON.\n"
            f"Text: {input_text}"
        )
        return self._llm.generate(prompt, temperature=self._temperature)
