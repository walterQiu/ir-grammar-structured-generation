"""Atomic strict metric: schema_exact_match."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from ntust_thesis.core.interfaces import Metric
from ntust_thesis.evaluation.metrics.common import safe_divide

if TYPE_CHECKING:
    from ntust_thesis.core.schemas import EvaluationRow, EventOutput


class SchemaExactMatchMetric(Metric):
    """Rate of predictions matching expected output schema shape."""

    def name(self) -> str:
        """Return metric key."""
        return "schema_exact_match"

    def compute(self, rows: list[EvaluationRow]) -> dict[str, float]:
        """Compute schema-match rate over rows."""
        total = len(rows)
        if total == 0:
            return {"schema_exact_match": 0.0}
        matched = sum(
            1
            for row in rows
            if row.parsed_output is not None
            and _schema_exact_match(row.parsed_output, row.gold)
        )
        return {"schema_exact_match": safe_divide(matched, total)}


def _schema_exact_match(pred: EventOutput, gold: EventOutput) -> bool:
    """Return whether prediction matches gold JSON structure exactly.

    Structure here ignores argument text span content, but requires:
    1. same top-level event type
    2. identical argument-role multiset (including multiplicity)
    """
    if pred.event_type != gold.event_type:
        return False
    pred_roles = Counter(arg.role for arg in pred.arguments)
    gold_roles = Counter(arg.role for arg in gold.arguments)
    return pred_roles == gold_roles
