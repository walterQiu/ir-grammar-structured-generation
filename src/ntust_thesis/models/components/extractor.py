"""Extraction stage components for IR pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ntust_thesis.prompts import build_two_stage_extraction_prompt

if TYPE_CHECKING:
    from ntust_thesis.core.interfaces import LLMClient


class Extractor:
    """Produce extraction text from raw input text."""

    def extract(
        self,
        sentence: str,
        event_type: str | None = None,
        role_multiplicities: dict[str, int] | None = None,
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
        role_multiplicities: dict[str, int] | None = None,
    ) -> str:
        """Ask Gemini to produce extraction notes."""
        system_prompt, user_prompt = build_two_stage_extraction_prompt(
            sentence=sentence,
            event_type=event_type,
            role_multiplicities=role_multiplicities,
        )
        return self._llm.generate(
            system_prompt,
            user_prompt,
            self._temperature,
        )
