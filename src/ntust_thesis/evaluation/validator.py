"""Strict validation utilities and validator implementation."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from ntust_thesis.core.interfaces import Validator
from ntust_thesis.core.registry import VALIDATOR_REGISTRY

if TYPE_CHECKING:
    from ntust_thesis.core.schemas import EventOutput, OutputSchema, Prediction, Sample


def is_valid_json(output_str: str) -> bool:
    """Check whether output is valid JSON."""
    try:
        json.loads(output_str)
    except json.JSONDecodeError:
        return False
    return True


def validate_schema(output: EventOutput, output_schema: OutputSchema) -> bool:
    """Minimal schema validation for required keys and extra keys."""
    obj = output.model_dump()
    if output_schema.type != "object":
        return False

    required = output_schema.required
    if any(key not in obj for key in required):
        return False

    allowed = set(required)
    return set(obj.keys()).issubset(allowed)


def exact_match(pred: EventOutput, gold: EventOutput) -> bool:
    """Strict structural equality check."""
    return pred.model_dump() == gold.model_dump()


class StrictValidator(Validator):
    """Validator for strict structured generation metrics."""

    def validate(self, prediction: Prediction, sample: Sample) -> dict[str, object]:
        """Validate JSON validity, schema validity, and exact match."""
        parsed = prediction.parsed_output
        json_valid = is_valid_json(prediction.raw_output)
        schema_valid = parsed is not None and validate_schema(
            parsed, sample.output_schema
        )
        exact = parsed is not None and exact_match(parsed, sample.gold)
        return {
            "json_valid": json_valid,
            "schema_valid": schema_valid,
            "exact_match": exact,
        }


def register() -> None:
    """Register built-in strict validator."""
    VALIDATOR_REGISTRY.register("strict", StrictValidator)
