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
from ntust_thesis.evaluation.difficulty import (
    DIFFICULTY_LEVELS,
    group_rows_by_schema_difficulty,
)
from ntust_thesis.evaluation.formatter import (
    format_metrics_payload,
    round_metric_values,
)

if TYPE_CHECKING:
    from ntust_thesis.core.config_models import ExperimentConfig

from ntust_thesis.core.schemas import EvaluationRow


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
            aggregated.update(round_metric_values(metric.compute(metric_rows)))

        rows_by_difficulty = group_rows_by_schema_difficulty(metric_rows)
        difficulty_aggregated: dict[str, dict[str, Any]] = {}
        for level in DIFFICULTY_LEVELS:
            rows_in_level = rows_by_difficulty[level]
            level_metrics: dict[str, Any] = {"sample_count": len(rows_in_level)}
            for metric in metrics:
                level_metrics.update(round_metric_values(metric.compute(rows_in_level)))
            difficulty_aggregated[level] = level_metrics

        formatted_metrics = format_metrics_payload(
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
