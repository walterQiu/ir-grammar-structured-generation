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
        metric_keys = self._config.evaluation.metrics

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

        difficulty_rows = {
            "easy": [row for row in metric_rows if _schema_difficulty(row) == "easy"],
            "medium": [
                row for row in metric_rows if _schema_difficulty(row) == "medium"
            ],
            "hard": [row for row in metric_rows if _schema_difficulty(row) == "hard"],
        }
        for level, rows_in_level in difficulty_rows.items():
            aggregated[f"{level}_sample_count"] = len(rows_in_level)
            for metric in metrics:
                metric_result = _round_metric_values(metric.compute(rows_in_level))
                for key, value in metric_result.items():
                    aggregated[f"{level}_{key}"] = value

        return PipelineResult(
            rows=rows,
            metrics=aggregated,
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
