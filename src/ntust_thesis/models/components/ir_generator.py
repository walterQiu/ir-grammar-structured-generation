"""IR generation stage components."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ntust_thesis.prompts import build_two_stage_ir_prompt

if TYPE_CHECKING:
    from ntust_thesis.core.interfaces import LLMClient


class IRGenerator:
    """Convert extraction text into dot-notation IR."""

    def generate(
        self,
        extraction_text: str,
        event_type: str,
        role_multiplicities: dict[str, int],
    ) -> str:
        """Generate IR text from extraction text."""
        raise NotImplementedError


class GeminiIRGenerator(IRGenerator):
    """Gemini-based IR generation stage."""

    def __init__(self, llm: LLMClient, temperature: float, ir_grammar: str) -> None:
        """Initialize IR generator with LLM backend."""
        self._llm = llm
        self._temperature = temperature
        self._ir_grammar = ir_grammar

    def generate(
        self,
        extraction_text: str,
        event_type: str,
        role_multiplicities: dict[str, int],
    ) -> str:
        """Generate IR text for configured grammar."""
        system_prompt, user_prompt = build_two_stage_ir_prompt(
            extraction_text=extraction_text,
            event_type=event_type,
            role_multiplicities=role_multiplicities,
            ir_grammar=self._ir_grammar,
        )
        return self._llm.generate(
            system_prompt,
            user_prompt,
            self._temperature,
            allow_empty=True,
        )
