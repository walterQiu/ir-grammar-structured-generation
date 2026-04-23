"""Metric output formatting helpers."""

from __future__ import annotations

from typing import Any

from ntust_thesis.evaluation.difficulty import DIFFICULTY_LEVELS

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
    difficulty_metrics: dict[str, dict[str, Any]],
    main_metric_keys: list[str],
    secondary_metric_keys: list[str],
) -> dict[str, Any]:
    """Build grouped metric payload for overall and difficulty slices."""
    return {
        "main": format_metric_block(overall_metrics, main_metric_keys),
        "secondary": format_metric_block(overall_metrics, secondary_metric_keys),
        "sample_difficulty": {
            level: {
                "sample_count": int(
                    difficulty_metrics.get(level, {}).get("sample_count", 0)
                ),
                "main": format_metric_block(
                    difficulty_metrics.get(level, {}),
                    main_metric_keys,
                ),
                "secondary": format_metric_block(
                    difficulty_metrics.get(level, {}),
                    secondary_metric_keys,
                ),
            }
            for level in DIFFICULTY_LEVELS
        },
    }


def format_metric_block(
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
