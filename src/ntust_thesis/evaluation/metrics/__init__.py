"""Metric registry entrypoint."""

from ntust_thesis.core.registry import METRIC_REGISTRY
from ntust_thesis.evaluation.metrics.arg import ArgCF1Metric, ArgIF1Metric
from ntust_thesis.evaluation.metrics.bemeae import BEMEAEMetric
from ntust_thesis.evaluation.metrics.content_similarity_sbert import (
    ContentSimilaritySBERTMetric,
)
from ntust_thesis.evaluation.metrics.is_valid_ir import IsValidIRMetric
from ntust_thesis.evaluation.metrics.non_absence_role_fallback import (
    NonAbsenceRoleFallbackIncrementalAssignmentMetric,
    NonAbsenceRoleFallbackJsonMetric,
)


def register() -> None:
    """Register built-in metrics."""
    METRIC_REGISTRY.register("is_valid_ir", IsValidIRMetric)
    METRIC_REGISTRY.register("arg_i_f1", ArgIF1Metric)
    METRIC_REGISTRY.register("arg_c_f1", ArgCF1Metric)
    METRIC_REGISTRY.register("content_similarity_sbert", ContentSimilaritySBERTMetric)
    METRIC_REGISTRY.register("bemeae", BEMEAEMetric)
    METRIC_REGISTRY.register(
        "non_absence_role_fallback_json",
        NonAbsenceRoleFallbackJsonMetric,
    )
    METRIC_REGISTRY.register(
        "non_absence_role_fallback_incremental_assignment",
        NonAbsenceRoleFallbackIncrementalAssignmentMetric,
    )
