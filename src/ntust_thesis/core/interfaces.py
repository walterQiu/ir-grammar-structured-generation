"""Abstract interfaces for pluggable benchmark components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ntust_thesis.core.schemas import (
        EvaluationRow,
        EventOutput,
        OutputSchema,
        Prediction,
        Sample,
    )


class Dataset(ABC):
    """Interface for benchmark datasets."""

    @abstractmethod
    def name(self) -> str:
        """Return dataset identifier."""

    @abstractmethod
    def load(self) -> list[Sample]:
        """Load and return dataset samples."""


class Model(ABC):
    """Interface for model inference."""

    @abstractmethod
    def name(self) -> str:
        """Return model identifier."""

    @abstractmethod
    def predict(self, sample: Sample) -> Prediction:
        """Generate prediction for one sample."""


class Validator(ABC):
    """Interface for strict validators."""

    @abstractmethod
    def validate(self, prediction: Prediction, sample: Sample) -> dict[str, object]:
        """Return validation flags/details for one sample."""


class Metric(ABC):
    """Interface for aggregate metric calculators."""

    @abstractmethod
    def name(self) -> str:
        """Return metric identifier."""

    @abstractmethod
    def compute(self, rows: list[EvaluationRow]) -> dict[str, object]:
        """Aggregate row-level outputs into metric values."""


class Compiler(ABC):
    """Optional interface for IR compiler components."""

    @abstractmethod
    def compile(
        self,
        ir_text: str,
        output_schema: OutputSchema,
        source_sentence: str,
        event_type: str,
    ) -> EventOutput:
        """Compile IR text into JSON output."""


class LLMClient(ABC):
    """Interface for interchangeable LLM backends."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs: object) -> str:
        """Generate text from prompt."""
