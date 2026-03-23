"""Aggregate metrics for strict evaluation."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from ntust_thesis.core.interfaces import Metric
from ntust_thesis.core.registry import METRIC_REGISTRY

if TYPE_CHECKING:
    from ntust_thesis.core.schemas import EvaluationRow


class StrictRatesMetric(Metric):
    """Compute strict benchmark rates from row-level flags."""

    def name(self) -> str:
        """Return metric key."""
        return "strict_rates"

    def compute(self, rows: list[EvaluationRow]) -> dict[str, float]:
        """Compute json/schema/exact-match rates."""
        total = len(rows)
        if total == 0:
            return {
                "json_valid_rate": 0.0,
                "schema_valid_rate": 0.0,
                "exact_match_rate": 0.0,
            }

        json_valid = sum(1 for row in rows if row.json_valid)
        schema_valid = sum(1 for row in rows if row.schema_valid)
        exact = sum(1 for row in rows if row.exact_match)
        return {
            "json_valid_rate": json_valid / total,
            "schema_valid_rate": schema_valid / total,
            "exact_match_rate": exact / total,
        }


def _safe_divide(numerator: int, denominator: int) -> float:
    """Return safe division result under zero-denominator."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _f1(precision: float, recall: float) -> float:
    """Compute harmonic mean under zero edge case."""
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


class ArgIF1Metric(Metric):
    """Argument Identification F1: match by argument text only."""

    def name(self) -> str:
        """Return metric key."""
        return "arg_i_f1"

    def compute(self, rows: list[EvaluationRow]) -> dict[str, float]:
        """Compute micro Precision/Recall/F1 for text-only matching."""
        tp = 0
        pred_total = 0
        gold_total = 0

        for row in rows:
            pred_args = (
                row.parsed_output.arguments if row.parsed_output is not None else []
            )
            gold_args = row.gold.arguments

            pred_texts = Counter(arg.text.strip() for arg in pred_args)
            gold_texts = Counter(arg.text.strip() for arg in gold_args)

            pred_total += sum(pred_texts.values())
            gold_total += sum(gold_texts.values())
            tp += sum((pred_texts & gold_texts).values())

        precision = _safe_divide(tp, pred_total)
        recall = _safe_divide(tp, gold_total)
        return {
            "arg_i_precision": precision,
            "arg_i_recall": recall,
            "arg_i_f1": _f1(precision, recall),
        }


class ArgCF1Metric(Metric):
    """Argument Classification F1: match by role and argument text."""

    def name(self) -> str:
        """Return metric key."""
        return "arg_c_f1"

    def compute(self, rows: list[EvaluationRow]) -> dict[str, float]:
        """Compute micro Precision/Recall/F1 for (role, text) matching."""
        tp = 0
        pred_total = 0
        gold_total = 0

        for row in rows:
            pred_args = (
                row.parsed_output.arguments if row.parsed_output is not None else []
            )
            gold_args = row.gold.arguments

            pred_items = Counter((arg.role, arg.text.strip()) for arg in pred_args)
            gold_items = Counter((arg.role, arg.text.strip()) for arg in gold_args)

            pred_total += sum(pred_items.values())
            gold_total += sum(gold_items.values())
            tp += sum((pred_items & gold_items).values())

        precision = _safe_divide(tp, pred_total)
        recall = _safe_divide(tp, gold_total)
        return {
            "arg_c_precision": precision,
            "arg_c_recall": recall,
            "arg_c_f1": _f1(precision, recall),
        }


def register() -> None:
    """Register built-in strict rates metric."""
    METRIC_REGISTRY.register("strict_rates", StrictRatesMetric)
    METRIC_REGISTRY.register("arg_i_f1", ArgIF1Metric)
    METRIC_REGISTRY.register("arg_c_f1", ArgCF1Metric)
