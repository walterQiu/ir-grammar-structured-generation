"""Schema generation stage components."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ntust_thesis.prompts import build_json_generation_prompt

if TYPE_CHECKING:
    from ntust_thesis.core.interfaces import LLMClient


class SchemaGenerator:
    """Convert extraction text into final JSON text."""

    def generate(
        self,
        extraction_text: str,
        role_multiplicities: dict[str, int] | None = None,
    ) -> str:
        """Generate final JSON text from extraction notes."""
        raise NotImplementedError


class LLMSchemaGenerator(SchemaGenerator):
    """LLM-backed schema generation stage."""

    def __init__(self, llm: LLMClient, temperature: float) -> None:
        """Initialize schema generator with LLM backend."""
        self._llm = llm
        self._temperature = temperature

    def generate(
        self,
        extraction_text: str,
        role_multiplicities: dict[str, int] | None = None,
    ) -> str:
        """Generate final JSON string from extraction text."""
        system_prompt, user_prompt = build_json_generation_prompt(
            extraction_text=extraction_text,
            role_multiplicities=role_multiplicities,
        )
        return self._llm.generate(
            system_prompt,
            user_prompt,
            self._temperature,
        )
