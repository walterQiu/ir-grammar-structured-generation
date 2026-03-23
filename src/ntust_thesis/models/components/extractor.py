"""Extraction stage components for IR pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ntust_thesis.prompts import build_ir_extraction_prompt

if TYPE_CHECKING:
    from ntust_thesis.core.interfaces import LLMClient


class Extractor:
    """Produce extraction text from raw input text."""

    def extract(
        self,
        sentence: str,
        event_type: str | None = None,
        legal_roles: list[str] | None = None,
    ) -> str:
        """Return extraction stage output text."""
        raise NotImplementedError


class GeminiExtractor(Extractor):
    """Gemini-based extraction stage."""

    def __init__(self, llm: LLMClient, temperature: float) -> None:
        """Initialize extractor with LLM backend."""
        self._llm = llm
        self._temperature = temperature

    def extract(
        self,
        sentence: str,
        event_type: str | None = None,
        legal_roles: list[str] | None = None,
    ) -> str:
        """Ask Gemini to produce extraction notes."""
        prompt = build_ir_extraction_prompt(
            sentence=sentence,
            event_type=event_type,
            legal_roles=legal_roles,
        )
        return self._llm.generate(prompt, temperature=self._temperature)
