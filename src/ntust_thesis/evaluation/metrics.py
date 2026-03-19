"""Aggregate metrics for strict evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ntust_thesis.core.interfaces import Metric
from ntust_thesis.core.registry import METRIC_REGISTRY

if TYPE_CHECKING:
    from ntust_thesis.core.types import JSONDict


class StrictRatesMetric(Metric):
    """Compute strict benchmark rates from row-level flags."""

    def name(self) -> str:
        """Return metric key."""
        return "strict_rates"

    def compute(self, rows: list[JSONDict]) -> JSONDict:
        """Compute json/schema/exact-match rates."""
        total = len(rows)
        if total == 0:
            return {
                "json_valid_rate": 0.0,
                "schema_valid_rate": 0.0,
                "exact_match_rate": 0.0,
            }

        json_valid = sum(1 for row in rows if row.get("json_valid", False))
        schema_valid = sum(1 for row in rows if row.get("schema_valid", False))
        exact = sum(1 for row in rows if row.get("exact_match", False))
        return {
            "json_valid_rate": json_valid / total,
            "schema_valid_rate": schema_valid / total,
            "exact_match_rate": exact / total,
        }


def register() -> None:
    """Register built-in strict rates metric."""
    METRIC_REGISTRY.register("strict_rates", StrictRatesMetric)
