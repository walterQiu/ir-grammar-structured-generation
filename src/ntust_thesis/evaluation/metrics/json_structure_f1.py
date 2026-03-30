"""JSON structure matching metric (path-only, value-agnostic)."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

from ntust_thesis.core.interfaces import Metric
from ntust_thesis.evaluation.metrics.common import f1, safe_divide

if TYPE_CHECKING:
    from ntust_thesis.core.schemas import EvaluationRow


class JsonStructureF1Metric(Metric):
    """Compare prediction/gold JSON structure using path-count overlap."""

    def name(self) -> str:
        """Return metric key."""
        return "json_structure_f1"

    def compute(self, rows: list[EvaluationRow]) -> dict[str, float]:
        """Compute micro P/R/F1 over flattened JSON path multisets."""
        tp = 0
        fp = 0
        fn = 0
        pred_total = 0
        gold_total = 0

        for row in rows:
            pred_arguments = _extract_arguments_field(row.parsed_output)
            gold_arguments = _extract_arguments_field(row.gold)

            pred_counts = Counter(_flatten_argument_role_paths(pred_arguments))
            gold_counts = Counter(_flatten_argument_role_paths(gold_arguments))

            all_paths = set(pred_counts) | set(gold_counts)
            for path in all_paths:
                pred_count = pred_counts.get(path, 0)
                gold_count = gold_counts.get(path, 0)
                tp += min(pred_count, gold_count)
                fp += max(pred_count - gold_count, 0)
                fn += max(gold_count - pred_count, 0)

            pred_total += sum(pred_counts.values())
            gold_total += sum(gold_counts.values())

        # Edge case: all prediction/gold path sets are empty -> perfect structural match.
        if pred_total == 0 and gold_total == 0:
            return {
                "json_structure_precision": 1.0,
                "json_structure_recall": 1.0,
                "json_structure_f1": 1.0,
            }

        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        return {
            "json_structure_precision": precision,
            "json_structure_recall": recall,
            "json_structure_f1": f1(precision, recall),
        }


def _extract_arguments_field(output: Any) -> list[Any]:  # noqa: ANN401
    """Extract arguments list from EventOutput-like object."""
    if output is None:
        return []
    if hasattr(output, "model_dump"):
        payload = output.model_dump()
    elif isinstance(output, dict):
        payload = output
    else:
        return []
    arguments = payload.get("arguments")
    if isinstance(arguments, list):
        return arguments
    return []


def _flatten_argument_role_paths(arguments: list[Any]) -> list[str]:
    """Flatten only argument-role semantic structure with duplicates."""
    paths: list[str] = []
    for item in arguments:
        if not isinstance(item, dict):
            continue
        raw_role = item.get("role")
        if not isinstance(raw_role, str):
            continue
        role = raw_role.strip()
        if not role:
            continue
        paths.append(f"arguments.{role}")
    return paths
