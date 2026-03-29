"""Bootstrap registration for built-in components."""

from __future__ import annotations

from ntust_thesis.datasets.rams import register as register_datasets
from ntust_thesis.evaluation.metrics import register as register_metrics
from ntust_thesis.evaluation.validator import register as register_validators
from ntust_thesis.models.baseline import register as register_baseline
from ntust_thesis.models.ir_pipeline import register as register_ir
from ntust_thesis.models.two_stage_baseline import (
    register as register_two_stage_baseline,
)


def register_builtin_components() -> None:
    """Register all built-in components once."""
    if getattr(register_builtin_components, "_done", False):
        return
    register_datasets()
    register_baseline()
    register_ir()
    register_two_stage_baseline()
    register_validators()
    register_metrics()
    register_builtin_components._done = True
