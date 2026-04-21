"""Atomic strict metric: exact_match."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from ntust_thesis.core.interfaces import Metric
from ntust_thesis.core.role_path import role_to_path
from ntust_thesis.evaluation.metrics.common import f1, safe_divide

if TYPE_CHECKING:
    from ntust_thesis.core.schemas import EvaluationRow


class ExactMatchMetric(Metric):
    """Exact pair-match metric over (role, span) with micro P/R/F1."""

    def name(self) -> str:
        """Return metric key."""
        return "exact_match"

    def compute(self, rows: list[EvaluationRow]) -> dict[str, float]:
        """Compute micro P/R/F1 over exact (role, span) pairs."""
        tp = 0
        pred_total = 0
        gold_total = 0

        for row in rows:
            pred_args = (
                row.parsed_output.arguments if row.parsed_output is not None else []
            )
            gold_args = row.gold.arguments

            pred_items = Counter(
                ((role_to_path(arg.role) or "unknown_role"), arg.span.strip())
                for arg in pred_args
            )
            gold_items = Counter(
                ((role_to_path(arg.role) or "unknown_role"), arg.span.strip())
                for arg in gold_args
            )

            pred_total += sum(pred_items.values())
            gold_total += sum(gold_items.values())
            tp += sum((pred_items & gold_items).values())

        precision = safe_divide(tp, pred_total)
        recall = safe_divide(tp, gold_total)
        return {
            "exact_match_precision": precision,
            "exact_match_recall": recall,
            "exact_match": f1(precision, recall),
        }
