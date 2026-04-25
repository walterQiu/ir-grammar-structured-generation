"""Empty-case abstention metric (ECAR)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ntust_thesis.core.interfaces import Metric
from ntust_thesis.evaluation.metrics.common import safe_divide

if TYPE_CHECKING:
    from ntust_thesis.core.schemas import EvaluationRow


class ECARMetric(Metric):
    """Rate of correctly abstained valid predictions on gold-empty samples."""

    def name(self) -> str:
        """Return metric key."""
        return "ecar"

    def compute(self, rows: list[EvaluationRow]) -> dict[str, float]:
        """Compute ECAR over rows whose gold arguments are empty."""
        gold_empty_rows = [row for row in rows if len(row.gold.arguments) == 0]
        total_gold_empty = len(gold_empty_rows)
        if total_gold_empty == 0:
            return {"ecar": 0.0}

        correct_abstained = sum(
            1 for row in gold_empty_rows if _is_valid_abstention(row)
        )
        return {
            "ecar": safe_divide(
                correct_abstained,
                total_gold_empty,
            )
        }


def _is_valid_abstention(row: EvaluationRow) -> bool:
    """Return whether a gold-empty row was validly predicted as empty."""
    if row.prediction_metadata.compile_error is not None:
        return False
    return row.parsed_output is not None and len(row.parsed_output.arguments) == 0
