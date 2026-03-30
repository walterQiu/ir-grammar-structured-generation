"""Atomic strict metric: exact_match."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ntust_thesis.core.interfaces import Metric
from ntust_thesis.evaluation.metrics.common import safe_divide

if TYPE_CHECKING:
    from ntust_thesis.core.schemas import EvaluationRow, EventOutput


class ExactMatchMetric(Metric):
    """Rate of exact argument matches between prediction and gold."""

    def name(self) -> str:
        """Return metric key."""
        return "exact_match"

    def compute(self, rows: list[EvaluationRow]) -> dict[str, float]:
        """Compute exact-match rate over rows."""
        total = len(rows)
        if total == 0:
            return {"exact_match": 0.0}
        matched = sum(
            1
            for row in rows
            if row.parsed_output is not None
            and _exact_match(row.parsed_output, row.gold)
        )
        return {"exact_match": safe_divide(matched, total)}


def _exact_match(pred: EventOutput, gold: EventOutput) -> bool:
    """Strict argument-level equality by role+text."""
    pred_args = sorted((arg.role, arg.text) for arg in pred.arguments)
    gold_args = sorted((arg.role, arg.text) for arg in gold.arguments)
    return pred_args == gold_args
