"""Experiment pipeline orchestrator."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ntust_thesis.core.registry import (
    DATASET_REGISTRY,
    METRIC_REGISTRY,
    MODEL_REGISTRY,
)

if TYPE_CHECKING:
    from ntust_thesis.core.config_models import ExperimentConfig

from ntust_thesis.core.schemas import EvaluationRow

_METRIC_DECIMAL_PLACES = 5
_DIFFICULTY_LEVELS = ("easy", "medium", "hard")
_METRIC_OUTPUT_FIELDS: dict[str, tuple[str, list[str]]] = {
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


@dataclass(slots=True)
class PipelineResult:
    """Minimal pipeline output container."""

    rows: list[dict[str, Any]]
    metrics: dict[str, Any]
    failed_samples: list[dict[str, Any]]


class ExperimentPipeline:
    """Coordinates dataset -> model -> metrics."""

    def __init__(self, config: ExperimentConfig) -> None:
        """Initialize pipeline with resolved experiment config."""
        self._config = config

    def run(self) -> PipelineResult:
        """Execute dataset -> model -> metric flow."""
        dataset_key = self._config.dataset.name
        dataset_config = self._config.dataset.model_dump()
        model_key = self._config.model.name
        model_config = self._config.model.model_dump()
        main_metric_keys = self._config.evaluation.metrics.main
        secondary_metric_keys = self._config.evaluation.metrics.secondary
        metric_keys = _flatten_metric_keys(
            main_metric_keys,
            secondary_metric_keys,
        )

        dataset = DATASET_REGISTRY.create(dataset_key, config=dataset_config)
        model = MODEL_REGISTRY.create(model_key, config=model_config)
        ir_grammar = self._config.model.ir_grammar
        metrics = [
            METRIC_REGISTRY.create(key, ir_grammar=ir_grammar)
            if key
            in {
                "is_valid_ir",
            }
            else METRIC_REGISTRY.create(key)
            for key in metric_keys
        ]
        samples = dataset.load()
        total = len(samples)

        metric_rows: list[EvaluationRow] = []
        rows: list[dict[str, Any]] = []
        failed_samples: list[dict[str, Any]] = []
        for idx, sample in enumerate(samples, start=1):
            try:
                prediction = model.predict(sample)
                row_data: dict[str, Any] = {
                    "sample_id": sample.sample_id,
                    "raw_output": prediction.raw_output,
                    "parsed_output": prediction.parsed_output,
                    "gold": sample.gold,
                    "prediction_metadata": prediction.metadata,
                    "sample_metadata": sample.metadata,
                }
                row_model = EvaluationRow.model_validate(row_data)
                metric_rows.append(row_model)
                rows.append(row_model.model_dump())
            except Exception as exc:
                failed_samples.append(
                    {
                        "sample_index": idx,
                        "sample_id": sample.sample_id,
                        "error": str(exc),
                    }
                )
                sys.stdout.write(
                    f"[warning] failed sample {idx}/{total}: {sample.sample_id}\n"
                )
                sys.stdout.flush()

            if idx % 10 == 0 or idx == total:
                sys.stdout.write(f"[progress] processed {idx}/{total} samples\n")
                sys.stdout.flush()

        aggregated: dict[str, Any] = {}
        for metric in metrics:
            aggregated.update(_round_metric_values(metric.compute(metric_rows)))

        difficulty_aggregated: dict[str, dict[str, Any]] = {}
        for level in _DIFFICULTY_LEVELS:
            rows_in_level = [
                row for row in metric_rows if _schema_difficulty(row) == level
            ]
            level_metrics: dict[str, Any] = {"sample_count": len(rows_in_level)}
            for metric in metrics:
                level_metrics.update(
                    _round_metric_values(metric.compute(rows_in_level))
                )
            difficulty_aggregated[level] = level_metrics

        formatted_metrics = _format_metrics_payload(
            aggregated,
            difficulty_aggregated,
            main_metric_keys,
            secondary_metric_keys,
        )

        return PipelineResult(
            rows=rows,
            metrics=formatted_metrics,
            failed_samples=failed_samples,
        )


def _schema_difficulty(row: EvaluationRow) -> str:
    """Return schema difficulty bucket from gold argument count."""
    n_roles = len(row.gold.arguments)
    if n_roles <= 2:  # noqa: PLR2004
        return "easy"
    if n_roles == 3:  # noqa: PLR2004
        return "medium"
    return "hard"


def _round_metric_values(metric_values: dict[str, object]) -> dict[str, object]:
    """Round all float metric values to fixed decimal places."""
    rounded: dict[str, object] = {}
    for key, value in metric_values.items():
        if isinstance(value, float):
            rounded[key] = round(value, _METRIC_DECIMAL_PLACES)
        else:
            rounded[key] = value
    return rounded


def _flatten_metric_keys(main: list[str], secondary: list[str]) -> list[str]:
    """Merge main/secondary metric keys while preserving first-seen order."""
    ordered_keys: list[str] = []
    seen: set[str] = set()
    for key in [*main, *secondary]:
        if key in seen:
            continue
        seen.add(key)
        ordered_keys.append(key)
    return ordered_keys


def _format_metrics_payload(
    overall_metrics: dict[str, Any],
    difficulty_metrics: dict[str, dict[str, Any]],
    main_metric_keys: list[str],
    secondary_metric_keys: list[str],
) -> dict[str, Any]:
    """Build beautified metric payload grouped by main/secondary and difficulty."""
    return {
        "main": _format_metric_block(overall_metrics, main_metric_keys),
        "secondary": _format_metric_block(overall_metrics, secondary_metric_keys),
        "sample_difficulty": {
            level: {
                "sample_count": int(
                    difficulty_metrics.get(level, {}).get("sample_count", 0)
                ),
                "main": _format_metric_block(
                    difficulty_metrics.get(level, {}),
                    main_metric_keys,
                ),
                "secondary": _format_metric_block(
                    difficulty_metrics.get(level, {}),
                    secondary_metric_keys,
                ),
            }
            for level in _DIFFICULTY_LEVELS
        },
    }


def _format_metric_block(
    metric_values: dict[str, Any],
    metric_keys: list[str],
) -> dict[str, Any]:
    """Format one metric block with F1 first and precision/recall after."""
    block: dict[str, Any] = {}
    for metric_key in metric_keys:
        primary, details = _METRIC_OUTPUT_FIELDS.get(metric_key, (metric_key, []))
        if primary in metric_values:
            block[primary] = metric_values[primary]
        for detail_key in details:
            if detail_key in metric_values:
                block[detail_key] = metric_values[detail_key]
    return block
