"""Experiment pipeline orchestrator."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ntust_thesis.core.registry import (
    DATASET_REGISTRY,
    METRIC_REGISTRY,
    MODEL_REGISTRY,
    VALIDATOR_REGISTRY,
)

if TYPE_CHECKING:
    from ntust_thesis.core.config_models import ExperimentConfig

from ntust_thesis.core.schemas import EvaluationRow


@dataclass(slots=True)
class PipelineResult:
    """Minimal pipeline output container."""

    rows: list[dict[str, Any]]
    metrics: dict[str, Any]


class ExperimentPipeline:
    """Coordinates dataset -> model -> validation -> metrics."""

    def __init__(self, config: ExperimentConfig) -> None:
        """Initialize pipeline with resolved experiment config."""
        self._config = config

    def run(self) -> PipelineResult:
        """Execute dataset -> model -> validator -> metric flow."""
        dataset_key = self._config.dataset.name
        dataset_config = self._config.dataset.model_dump()
        model_key = self._config.model.name
        model_config = self._config.model.model_dump()
        validator_keys = self._config.evaluation.validators
        metric_keys = self._config.evaluation.metrics

        dataset = DATASET_REGISTRY.create(dataset_key, config=dataset_config)
        model = MODEL_REGISTRY.create(model_key, config=model_config)
        validators = [VALIDATOR_REGISTRY.create(key) for key in validator_keys]
        metrics = [METRIC_REGISTRY.create(key) for key in metric_keys]
        samples = dataset.load()
        total = len(samples)

        metric_rows: list[EvaluationRow] = []
        rows: list[dict[str, Any]] = []
        for idx, sample in enumerate(samples, start=1):
            prediction = model.predict(sample)
            row_data: dict[str, Any] = {
                "sample_id": sample.sample_id,
                "raw_output": prediction.raw_output,
                "parsed_output": prediction.parsed_output,
                "gold": sample.gold,
                "prediction_metadata": prediction.metadata,
                "sample_metadata": sample.metadata,
            }
            for validator in validators:
                row_data.update(validator.validate(prediction, sample))
            row_model = EvaluationRow.model_validate(row_data)
            metric_rows.append(row_model)
            rows.append(row_model.model_dump())

            if idx % 10 == 0 or idx == total:
                sys.stdout.write(f"[progress] processed {idx}/{total} samples\n")
                sys.stdout.flush()

        aggregated: dict[str, Any] = {}
        for metric in metrics:
            aggregated.update(metric.compute(metric_rows))
        return PipelineResult(rows=rows, metrics=aggregated)
