"""Shared typed domain models across the benchmark pipeline."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Argument(BaseModel):
    """Event argument structure."""

    model_config = ConfigDict(extra="forbid")

    role: str | dict[str, Any]
    span: str


class EventOutput(BaseModel):
    """Structured event output used for gold and predictions."""

    model_config = ConfigDict(extra="forbid")

    event_type: str
    arguments: list[Argument] = Field(default_factory=list)


class OutputSchema(BaseModel):
    """Schema spec used by strict validation."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["object"] = "object"
    required: tuple[str, ...] = ("event_type", "arguments")


class SampleMetadata(BaseModel):
    """Metadata for one dataset sample."""

    model_config = ConfigDict(extra="allow")
    sentence_text: str | None = None
    marked_sentence: str | None = None
    event_type: str
    candidate_roles: list[str] | None = None
    role_multiplicities: dict[str, int]


class PredictionMetadata(BaseModel):
    """Metadata for one model prediction."""

    model_config = ConfigDict(extra="allow")

    model: str
    backend: str | None = None
    extraction_backend: str | None = None
    ir_backend: str | None = None
    extraction_text: str | None = None
    ir_text: str | None = None
    compile_error: str | None = None
    model_input: str | dict[str, str] | None = None


class Sample(BaseModel):
    """Single benchmark sample."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    raw_sentence: str
    output_schema: OutputSchema
    gold: EventOutput
    metadata: SampleMetadata


class Prediction(BaseModel):
    """Model prediction for one sample."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    raw_output: str
    parsed_output: EventOutput | None
    metadata: PredictionMetadata


class EvaluationRow(BaseModel):
    """Row-level record used for metrics and artifact export."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    raw_output: str
    parsed_output: EventOutput | None
    gold: EventOutput
    prediction_metadata: PredictionMetadata
    sample_metadata: SampleMetadata
