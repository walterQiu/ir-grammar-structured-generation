"""Experiment pipeline orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class PipelineResult:
    """Minimal pipeline output container."""

    rows: list[dict[str, Any]]
    metrics: dict[str, Any]


class ExperimentPipeline:
    """Coordinates dataset -> model -> validation -> metrics."""

    def run(self) -> PipelineResult:
        """Execute experiment flow.

        Detailed implementation will be added step by step.
        """
        return PipelineResult(rows=[], metrics={})
