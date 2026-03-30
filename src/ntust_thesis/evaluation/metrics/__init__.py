"""Metric registry entrypoint."""

from ntust_thesis.core.registry import METRIC_REGISTRY
from ntust_thesis.evaluation.metrics.arg import ArgCF1Metric, ArgIF1Metric
from ntust_thesis.evaluation.metrics.bemeae import BEMEAEMetric
from ntust_thesis.evaluation.metrics.content_similarity_sbert import (
    ContentSimilaritySBERTMetric,
)
from ntust_thesis.evaluation.metrics.exact_match import ExactMatchMetric
from ntust_thesis.evaluation.metrics.is_valid_json import IsValidJsonMetric
from ntust_thesis.evaluation.metrics.json_structure_f1 import JsonStructureF1Metric
from ntust_thesis.evaluation.metrics.schema_exact_match import SchemaExactMatchMetric


def register() -> None:
    """Register built-in metrics."""
    METRIC_REGISTRY.register("is_valid_json", IsValidJsonMetric)
    METRIC_REGISTRY.register("schema_exact_match", SchemaExactMatchMetric)
    METRIC_REGISTRY.register("exact_match", ExactMatchMetric)
    METRIC_REGISTRY.register("arg_i_f1", ArgIF1Metric)
    METRIC_REGISTRY.register("arg_c_f1", ArgCF1Metric)
    METRIC_REGISTRY.register("content_similarity_sbert", ContentSimilaritySBERTMetric)
    METRIC_REGISTRY.register("bemeae", BEMEAEMetric)
    METRIC_REGISTRY.register("json_structure_f1", JsonStructureF1Metric)
