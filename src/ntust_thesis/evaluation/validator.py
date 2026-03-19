"""Strict validation utilities and validator implementation."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from ntust_thesis.core.interfaces import Validator
from ntust_thesis.core.registry import VALIDATOR_REGISTRY

if TYPE_CHECKING:
    from ntust_thesis.core.types import JSONDict, Prediction, Sample


def is_valid_json(output_str: str) -> bool:
    """Check whether output is valid JSON."""
    try:
        json.loads(output_str)
    except json.JSONDecodeError:
        return False
    return True


def validate_schema(json_obj: JSONDict, schema: JSONDict) -> bool:
    """Minimal schema validation for required keys and extra keys."""
    if schema.get("type") != "object" or not isinstance(json_obj, dict):
        return False

    required = schema.get("required", [])
    if any(key not in json_obj for key in required):
        return False

    if schema.get("additionalProperties") is False:
        allowed = set(required)
        return set(json_obj.keys()).issubset(allowed)
    return True


def exact_match(pred: JSONDict, gold: JSONDict) -> bool:
    """Strict structural equality check."""
    return pred == gold


class StrictValidator(Validator):
    """Validator for strict structured generation metrics."""

    def validate(self, prediction: Prediction, sample: Sample) -> JSONDict:
        """Validate JSON validity, schema validity, and exact match."""
        parsed = prediction.parsed_output
        json_valid = is_valid_json(prediction.raw_output)
        schema_valid = isinstance(parsed, dict) and validate_schema(
            parsed, sample.schema
        )
        exact = isinstance(parsed, dict) and exact_match(parsed, sample.gold)
        return {
            "json_valid": json_valid,
            "schema_valid": schema_valid,
            "exact_match": exact,
        }


def register() -> None:
    """Register built-in strict validator."""
    VALIDATOR_REGISTRY.register("strict", StrictValidator)
