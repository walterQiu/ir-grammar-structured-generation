"""Common types and interfaces for IR grammars."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class IRValidationResult:
    """Validation result for one IR text."""

    is_valid: bool
    error_message: str | None = None
    error_line_no: int | None = None
    error_line_text: str | None = None


class IRGrammarValidator(ABC):
    """Interface for pluggable IR grammar validators."""

    @abstractmethod
    def name(self) -> str:
        """Return grammar identifier."""

    @abstractmethod
    def validate(self, ir_text: str) -> IRValidationResult:
        """Validate IR text and return structured result."""


IRParser = Callable[[str], list[tuple[str, str]]]
