"""IR generation stage components."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ntust_thesis.prompts import build_ir_generation_prompt

if TYPE_CHECKING:
    from ntust_thesis.core.interfaces import LLMClient


class IRGenerator:
    """Convert extraction text into dot-notation IR."""

    def generate(
        self,
        extraction_text: str,
        role_multiplicities: dict[str, int] | None = None,
    ) -> str:
        """Generate IR text from extraction text."""
        raise NotImplementedError


class GeminiIRGenerator(IRGenerator):
    """Gemini-based IR generation stage."""

    def __init__(self, llm: LLMClient, temperature: float) -> None:
        """Initialize IR generator with LLM backend."""
        self._llm = llm
        self._temperature = temperature

    def generate(
        self,
        extraction_text: str,
        role_multiplicities: dict[str, int] | None = None,
    ) -> str:
        """Generate strict dot-notation IR lines."""
        prompt = build_ir_generation_prompt(
            extraction_text=extraction_text,
            role_multiplicities=role_multiplicities,
        )
        return self._llm.generate(
            prompt,
            temperature=self._temperature,
            allow_empty=True,
        )
