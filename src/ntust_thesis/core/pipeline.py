"""Experiment pipeline orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ntust_thesis.core.registry import (
    DATASET_REGISTRY,
    METRIC_REGISTRY,
    MODEL_REGISTRY,
    VALIDATOR_REGISTRY,
)


@dataclass(slots=True)
class PipelineResult:
    """Minimal pipeline output container."""

    rows: list[dict[str, Any]]
    metrics: dict[str, Any]


class ExperimentPipeline:
    """Coordinates dataset -> model -> validation -> metrics."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize pipeline with resolved experiment config."""
        self._config = config

    def run(self) -> PipelineResult:
        """Execute dataset -> model -> validator -> metric flow."""
        dataset_key = self._config["dataset"]["name"]
        dataset_config = self._config["dataset"]
        model_key = self._config["model"]["name"]
        validator_keys = self._config["evaluation"].get("validators", ["strict"])
        metric_keys = self._config["evaluation"].get("metrics", ["strict_rates"])

        dataset = DATASET_REGISTRY.create(dataset_key, config=dataset_config)
        model = MODEL_REGISTRY.create(model_key)
        validators = [VALIDATOR_REGISTRY.create(key) for key in validator_keys]
        metrics = [METRIC_REGISTRY.create(key) for key in metric_keys]

        rows: list[dict[str, Any]] = []
        for sample in dataset.load():
            prediction = model.predict(sample)
            row: dict[str, Any] = {"sample_id": sample.sample_id}
            for validator in validators:
                row.update(validator.validate(prediction, sample))
            rows.append(row)

        aggregated: dict[str, Any] = {}
        for metric in metrics:
            aggregated.update(metric.compute(rows))
        return PipelineResult(rows=rows, metrics=aggregated)
