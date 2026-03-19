"""Shared data types across the benchmark pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

JSONDict = dict[str, Any]


@dataclass(slots=True)
class Sample:
    """Single benchmark sample."""

    sample_id: str
    input_text: str
    schema: JSONDict
    gold: JSONDict
    metadata: JSONDict


@dataclass(slots=True)
class Prediction:
    """Model prediction for one sample."""

    sample_id: str
    raw_output: str
    parsed_output: JSONDict | None
    metadata: JSONDict
