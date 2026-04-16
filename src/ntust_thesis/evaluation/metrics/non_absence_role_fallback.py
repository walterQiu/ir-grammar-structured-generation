"""Fallback metrics that drop non-absence placeholder role values first."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ntust_thesis.core.interfaces import Metric
from ntust_thesis.core.schemas import EventOutput
from ntust_thesis.evaluation.metrics.arg import ArgCF1Metric
from ntust_thesis.evaluation.metrics.bemeae import BEMEAEMetric

if TYPE_CHECKING:
    from ntust_thesis.core.schemas import EvaluationRow

_ABSENCE_MARKERS = frozenset({"none", "null", "not specified", "not specific"})
_MIN_QUOTED_LENGTH = 2


class NonAbsenceRoleFallbackJsonMetric(Metric):
    """Apply role-fallback cleanup before Arg-C/BEMEAE for JSON IR."""

    def __init__(self, ir_grammar: str) -> None:
        """Initialize fallback metric with configured experiment grammar."""
        self._inner = _NonAbsenceRoleFallbackMetricBase(
            metric_name="non_absence_role_fallback_json",
            target_grammar="json",
            ir_grammar=ir_grammar,
        )

    def name(self) -> str:
        """Return metric key."""
        return self._inner.name()

    def compute(self, rows: list[EvaluationRow]) -> dict[str, float]:
        """Compute cleaned sub-metrics for JSON IR."""
        return self._inner.compute(rows)


class NonAbsenceRoleFallbackDotNotationMetric(Metric):
    """Apply role-fallback cleanup before Arg-C/BEMEAE for dot IR."""

    def __init__(self, ir_grammar: str) -> None:
        """Initialize fallback metric with configured experiment grammar."""
        self._inner = _NonAbsenceRoleFallbackMetricBase(
            metric_name="non_absence_role_fallback_dot_notation",
            target_grammar="dot_notation_ir",
            ir_grammar=ir_grammar,
        )

    def name(self) -> str:
        """Return metric key."""
        return self._inner.name()

    def compute(self, rows: list[EvaluationRow]) -> dict[str, float]:
        """Compute cleaned sub-metrics for dot-notation IR."""
        return self._inner.compute(rows)


class _NonAbsenceRoleFallbackMetricBase(Metric):
    """Shared implementation for grammar-specific fallback metrics."""

    def __init__(
        self,
        *,
        metric_name: str,
        target_grammar: str,
        ir_grammar: str,
    ) -> None:
        """Build one grammar-scoped fallback metric."""
        self._metric_name = metric_name
        self._target_grammar = target_grammar
        self._enabled = ir_grammar == target_grammar
        self._arg_c_metric = ArgCF1Metric()
        self._bemeae_metric = BEMEAEMetric()

    def name(self) -> str:
        """Return metric key."""
        return self._metric_name

    def compute(self, rows: list[EvaluationRow]) -> dict[str, float]:
        """Compute sub-metrics after removing placeholder absence values."""
        keys = {
            f"{self._metric_name}_arg_c_f1": 0.0,
            f"{self._metric_name}_bemeae": 0.0,
        }
        if not self._enabled:
            return keys

        cleaned_rows = [_sanitize_row_arguments(row) for row in rows]
        arg_c = self._arg_c_metric.compute(cleaned_rows)
        bemeae = self._bemeae_metric.compute(cleaned_rows)
        return {
            f"{self._metric_name}_arg_c_f1": float(arg_c.get("arg_c_f1", 0.0)),
            f"{self._metric_name}_bemeae": float(bemeae.get("bemeae", 0.0)),
        }


def _sanitize_row_arguments(row: EvaluationRow) -> EvaluationRow:
    """Return row with placeholder-absence arguments removed from prediction."""
    pred = row.parsed_output
    if pred is None:
        return row

    kept_args = [arg for arg in pred.arguments if not _is_absence_value(arg.span)]
    if len(kept_args) == len(pred.arguments):
        return row

    sanitized_pred = EventOutput(
        event_type=pred.event_type,
        arguments=kept_args,
    )
    return row.model_copy(update={"parsed_output": sanitized_pred})


def _is_absence_value(text: str) -> bool:
    """Return whether text is a configured non-absence fallback marker."""
    value = text.strip()
    while (
        len(value) >= _MIN_QUOTED_LENGTH
        and value[0] == value[-1]
        and value[0] in {"'", '"'}
    ):
        value = value[1:-1].strip()
    normalized = " ".join(value.split()).casefold()
    return normalized in _ABSENCE_MARKERS
