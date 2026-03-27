"""Strict validity-rate metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ntust_thesis.core.interfaces import Metric

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
