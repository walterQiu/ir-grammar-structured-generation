"""Stub baseline model."""

from __future__ import annotations

import json

from ntust_thesis.core.interfaces import Model
from ntust_thesis.core.registry import MODEL_REGISTRY
from ntust_thesis.core.types import Prediction, Sample


class BaselineModel(Model):
    """Simple baseline that echoes gold as a deterministic stub."""

    def name(self) -> str:
        """Return model key."""
        return "baseline"

    def predict(self, sample: Sample) -> Prediction:
        """Generate deterministic stub prediction."""
        return Prediction(
            sample_id=sample.sample_id,
            raw_output=json.dumps(sample.gold, ensure_ascii=False),
            parsed_output=sample.gold,
            metadata={"model": self.name()},
        )


def register() -> None:
    """Register built-in baseline model."""
    MODEL_REGISTRY.register("baseline", BaselineModel)
