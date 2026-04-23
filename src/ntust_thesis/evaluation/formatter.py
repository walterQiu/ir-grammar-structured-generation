"""Metric output formatting helpers."""

from __future__ import annotations

from typing import Any

METRIC_DECIMAL_PLACES = 5
METRIC_OUTPUT_FIELDS: dict[str, tuple[str, list[str]]] = {
    "is_valid_ir": ("is_valid_ir", []),
    "arg_i_f1": ("arg_i_f1", ["arg_i_precision", "arg_i_recall"]),
    "arg_c_f1": ("arg_c_f1", ["arg_c_precision", "arg_c_recall"]),
    "content_similarity_sbert": (
        "sbert_cossim_f1",
        ["sbert_cossim_precision", "sbert_cossim_recall"],
    ),
    "bemeae": ("bemeae", ["bemeae_soft_precision", "bemeae_soft_recall"]),
    "ecar": ("ecar", []),
}


def round_metric_values(metric_values: dict[str, object]) -> dict[str, object]:
    """Round all float metric values to fixed decimal places."""
    rounded: dict[str, object] = {}
    for key, value in metric_values.items():
        if isinstance(value, float):
            rounded[key] = round(value, METRIC_DECIMAL_PLACES)
        else:
            rounded[key] = value
    return rounded


def format_metrics_payload(
    overall_metrics: dict[str, Any],
    difficulty_metrics: dict[
        int, dict[int, dict[str, Any]]
    ],  # [role_multiplicities_total, [gold_span_total, [metrics]]]
    main_metric_keys: list[str],
    secondary_metric_keys: list[str],
) -> dict[str, Any]:
    """Build grouped metric payload for overall and difficulty slices."""
    return {
        "main": _format_metric_block(overall_metrics, main_metric_keys),
        "secondary": _format_metric_block(overall_metrics, secondary_metric_keys),
        "sample_difficulty": _format_sample_difficulty_block(
            difficulty_metrics,
            main_metric_keys,
            secondary_metric_keys,
        ),
    }


def _format_metric_block(
    metric_values: dict[str, Any],
    metric_keys: list[str],
) -> dict[str, Any]:
    """Format one metric block with F1 first and precision/recall after."""
    block: dict[str, Any] = {}
    for metric_key in metric_keys:
        primary, details = METRIC_OUTPUT_FIELDS.get(metric_key, (metric_key, []))
        if primary in metric_values:
            block[primary] = metric_values[primary]
        for detail_key in details:
            if detail_key in metric_values:
                block[detail_key] = metric_values[detail_key]
    return block


def _format_sample_difficulty_block(
    difficulty_metrics: dict[int, dict[int, dict[str, Any]]],
    main_metric_keys: list[str],
    secondary_metric_keys: list[str],
) -> dict[str, Any]:
    """Format role_multiplicity_total -> gold_role_count -> metric blocks."""
    formatted: dict[str, Any] = {}
    for multiplicity_total in sorted(difficulty_metrics):
        by_gold_count = difficulty_metrics[multiplicity_total]
        formatted_by_gold_count: dict[str, Any] = {}
        for role_count in sorted(by_gold_count):
            metric_values = by_gold_count[role_count]
            formatted_by_gold_count[str(role_count)] = {
                "sample_count": int(metric_values.get("sample_count", 0)),
                "main": _format_metric_block(metric_values, main_metric_keys),
                "secondary": _format_metric_block(
                    metric_values,
                    secondary_metric_keys,
                ),
            }
        formatted[str(multiplicity_total)] = formatted_by_gold_count
    return formatted
