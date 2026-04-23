"""IR grammar validity metric for IR-producing pipelines."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ntust_thesis.core.interfaces import Metric
from ntust_thesis.evaluation.metrics.common import safe_divide
from ntust_thesis.ir import get_ir_grammar_validator

if TYPE_CHECKING:
    from ntust_thesis.core.schemas import EvaluationRow


class IsValidIRMetric(Metric):
    """Rate of rows whose IR text conforms to target grammar."""

    def __init__(self, ir_grammar: str) -> None:
        """Initialize metric with target IR grammar."""
        self._validator = get_ir_grammar_validator(ir_grammar)

    def name(self) -> str:
        """Return metric key."""
        return "is_valid_ir"

    def compute(self, rows: list[EvaluationRow]) -> dict[str, float]:
        """Compute validity rate over all rows."""
        if not rows:
            return {"is_valid_ir": 0.0}

        valid = sum(
            1
            for row in rows
            if self._is_row_valid(row.prediction_metadata.ir_text or "")
        )
        return {"is_valid_ir": safe_divide(valid, len(rows))}

    def _is_row_valid(self, ir_text: str) -> bool:
        """Return validator result; treat unexpected validator errors as invalid."""
        try:
            return self._validator.validate(ir_text).is_valid
        except Exception:
            return False
