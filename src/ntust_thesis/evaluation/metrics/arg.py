"""Argument-level F1 metrics."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from ntust_thesis.core.interfaces import Metric
from ntust_thesis.evaluation.metrics.common import f1, safe_divide

if TYPE_CHECKING:
    from ntust_thesis.core.schemas import EvaluationRow


class ArgIF1Metric(Metric):
    """Argument Identification F1: match by argument span only."""

    def name(self) -> str:
        """Return metric key."""
        return "arg_i_f1"

    def compute(self, rows: list[EvaluationRow]) -> dict[str, float]:
        """Compute micro Precision/Recall/F1 for span-only matching."""
        tp = 0
        pred_total = 0
        gold_total = 0

        for row in rows:
            pred_args = (
                row.parsed_output.arguments if row.parsed_output is not None else []
            )
            gold_args = row.gold.arguments

            pred_spans = Counter(arg.span.strip() for arg in pred_args)
            gold_spans = Counter(arg.span.strip() for arg in gold_args)

            pred_total += sum(pred_spans.values())
            gold_total += sum(gold_spans.values())
            tp += sum((pred_spans & gold_spans).values())

        precision = safe_divide(tp, pred_total)
        recall = safe_divide(tp, gold_total)
        return {
            "arg_i_precision": precision,
            "arg_i_recall": recall,
            "arg_i_f1": f1(precision, recall),
        }


class ArgCF1Metric(Metric):
    """Argument Classification F1: match by role and argument span."""

    def name(self) -> str:
        """Return metric key."""
        return "arg_c_f1"

    def compute(self, rows: list[EvaluationRow]) -> dict[str, float]:
        """Compute micro Precision/Recall/F1 for (role, span) matching."""
        tp = 0
        pred_total = 0
        gold_total = 0

        for row in rows:
            pred_args = (
                row.parsed_output.arguments if row.parsed_output is not None else []
            )
            gold_args = row.gold.arguments

            pred_items = Counter((arg.role, arg.span.strip()) for arg in pred_args)
            gold_items = Counter((arg.role, arg.span.strip()) for arg in gold_args)

            pred_total += sum(pred_items.values())
            gold_total += sum(gold_items.values())
            tp += sum((pred_items & gold_items).values())

        precision = safe_divide(tp, pred_total)
        recall = safe_divide(tp, gold_total)
        return {
            "arg_c_precision": precision,
            "arg_c_recall": recall,
            "arg_c_f1": f1(precision, recall),
        }
