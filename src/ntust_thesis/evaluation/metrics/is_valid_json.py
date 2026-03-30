"""Atomic strict metric: is_valid_json."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from ntust_thesis.core.interfaces import Metric
from ntust_thesis.evaluation.metrics.common import safe_divide

if TYPE_CHECKING:
    from ntust_thesis.core.schemas import EvaluationRow


class IsValidJsonMetric(Metric):
    """Rate of predictions that are valid JSON text."""

    def name(self) -> str:
        """Return metric key."""
        return "is_valid_json"

    def compute(self, rows: list[EvaluationRow]) -> dict[str, float]:
        """Compute valid-JSON rate over rows."""
        total = len(rows)
        if total == 0:
            return {"is_valid_json": 0.0}
        valid = sum(1 for row in rows if _is_valid_json(row.raw_output))
        return {"is_valid_json": safe_divide(valid, total)}


def _is_valid_json(output_str: str) -> bool:
    """Return whether raw output string is valid JSON."""
    try:
        json.loads(output_str)
    except json.JSONDecodeError:
        return False
    return True
