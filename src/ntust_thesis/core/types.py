"""Shared typed domain models across the benchmark pipeline."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Argument(BaseModel):
    """Event argument structure."""

    model_config = ConfigDict(extra="forbid")

    role: str
    text: str
    span: tuple[int, int]


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
    additional_properties: bool = Field(
        default=False,
        alias="additionalProperties",
        serialization_alias="additionalProperties",
    )


class SampleMetadata(BaseModel):
    """Metadata for one dataset sample."""

    model_config = ConfigDict(extra="allow")


class PredictionMetadata(BaseModel):
    """Metadata for one model prediction."""

    model_config = ConfigDict(extra="allow")

    model: str


class Sample(BaseModel):
    """Single benchmark sample."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    sample_id: str
    input_text: str
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
