"""Stub IR pipeline model."""

from __future__ import annotations

import json

from ntust_thesis.core.interfaces import Model
from ntust_thesis.core.registry import MODEL_REGISTRY
from ntust_thesis.core.types import Prediction, Sample


class IRPipelineModel(Model):
    """Temporary IR model that returns deterministic stub output."""

    def name(self) -> str:
        """Return model key."""
        return "ir_pipeline"

    def predict(self, sample: Sample) -> Prediction:
        """Generate deterministic compiled output for smoke tests."""
        compiled = sample.gold
        return Prediction(
            sample_id=sample.sample_id,
            raw_output=json.dumps(compiled, ensure_ascii=False),
            parsed_output=compiled,
            metadata={"model": self.name(), "stage": "stub_compiler"},
        )


def register() -> None:
    """Register built-in IR pipeline model."""
    MODEL_REGISTRY.register("ir_pipeline", IRPipelineModel)
