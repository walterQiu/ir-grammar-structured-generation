"""Fallback metrics that drop non-absence placeholder role values first."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from ntust_thesis.core.interfaces import Metric
from ntust_thesis.core.schemas import EventOutput
from ntust_thesis.evaluation.metrics.arg import ArgCF1Metric
from ntust_thesis.evaluation.metrics.bemeae import BEMEAEMetric
from ntust_thesis.evaluation.metrics.common import safe_divide
from ntust_thesis.ir import get_ir_grammar_validator

if TYPE_CHECKING:
    from ntust_thesis.core.schemas import EvaluationRow

_ABSENCE_MARKERS = frozenset({"none", "null", "not specified", "not specific"})
_MIN_QUOTED_LENGTH = 2


class NonAbsenceRoleFallbackJsonMetric(Metric):
    """Apply role-fallback cleanup before Arg-C/BEMEAE/is_valid_ir for JSON IR."""

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
    """Apply role-fallback cleanup before Arg-C/BEMEAE/is_valid_ir for dot IR."""

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
        self._validator = get_ir_grammar_validator(target_grammar)

    def name(self) -> str:
        """Return metric key."""
        return self._metric_name

    def compute(self, rows: list[EvaluationRow]) -> dict[str, float]:
        """Compute sub-metrics after removing placeholder absence values."""
        keys = {
            f"{self._metric_name}_arg_c_f1": 0.0,
            f"{self._metric_name}_bemeae": 0.0,
            f"{self._metric_name}_is_valid_ir": 0.0,
        }
        if not self._enabled:
            return keys

        cleaned_rows = [_sanitize_row_arguments(row) for row in rows]
        arg_c = self._arg_c_metric.compute(cleaned_rows)
        bemeae = self._bemeae_metric.compute(cleaned_rows)
        is_valid = self._compute_cleaned_is_valid_ir(rows)

        return {
            f"{self._metric_name}_arg_c_f1": float(arg_c.get("arg_c_f1", 0.0)),
            f"{self._metric_name}_bemeae": float(bemeae.get("bemeae", 0.0)),
            f"{self._metric_name}_is_valid_ir": is_valid,
        }

    def _compute_cleaned_is_valid_ir(self, rows: list[EvaluationRow]) -> float:
        """Compute IR validity after grammar-aware fallback cleanup."""
        ir_rows = [
            row
            for row in rows
            if getattr(row.prediction_metadata, "ir_text", None) is not None
        ]
        if not ir_rows:
            return 0.0

        valid = 0
        for row in ir_rows:
            raw_ir = row.prediction_metadata.ir_text or ""
            cleaned_ir = _sanitize_ir_text_by_grammar(
                raw_ir,
                grammar=self._target_grammar,
            )
            if self._validator.validate(cleaned_ir).is_valid:
                valid += 1
        return safe_divide(valid, len(ir_rows))


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


def _sanitize_ir_text_by_grammar(ir_text: str, *, grammar: str) -> str:
    """Drop placeholder-absence role entries from IR text per grammar."""
    if grammar == "json":
        return _sanitize_json_ir_text(ir_text)
    if grammar == "dot_notation_ir":
        return _sanitize_dot_notation_ir_text(ir_text)
    return ir_text


def _sanitize_json_ir_text(ir_text: str) -> str:
    """Remove JSON arguments whose span is non-absence fallback marker."""
    try:
        payload = json.loads(ir_text.strip())
    except Exception:
        return ir_text

    if not isinstance(payload, dict):
        return ir_text
    arguments = payload.get("arguments")
    if not isinstance(arguments, list):
        return ir_text

    filtered = []
    for item in arguments:
        if not isinstance(item, dict):
            filtered.append(item)
            continue
        span = item.get("span")
        if isinstance(span, str) and _is_absence_value(span):
            continue
        filtered.append(item)

    payload["arguments"] = filtered
    return json.dumps(payload, ensure_ascii=False)


def _sanitize_dot_notation_ir_text(ir_text: str) -> str:
    """Remove dot-notation lines whose rhs value is absence marker."""
    kept_lines: list[str] = []
    for line in ir_text.splitlines():
        stripped = line.strip()
        if not stripped:
            kept_lines.append(line)
            continue

        rhs: str | None = None
        if "+=" in stripped:
            rhs = stripped.split("+=", 1)[1].strip()
        elif "=" in stripped:
            rhs = stripped.split("=", 1)[1].strip()

        if rhs is not None and _is_absence_value(rhs):
            continue
        kept_lines.append(line)
    return "\n".join(kept_lines)


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
