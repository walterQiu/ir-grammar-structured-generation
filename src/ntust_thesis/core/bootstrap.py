"""Bootstrap registration for built-in components."""

from __future__ import annotations

from ntust_thesis.datasets.rams import register as register_datasets
from ntust_thesis.evaluation.metrics import register as register_metrics
from ntust_thesis.models.one_stage_ir import register as register_one_stage_ir
from ntust_thesis.models.two_stage_ir import register as register_two_stage_ir


def register_builtin_components() -> None:
    """Register all built-in components once."""
    if getattr(register_builtin_components, "_done", False):
        return
    register_datasets()
    register_one_stage_ir()
    register_two_stage_ir()
    register_metrics()
    register_builtin_components._done = True
