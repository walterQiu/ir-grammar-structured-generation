"""Stub datasets for pipeline wiring and smoke tests."""

from __future__ import annotations

from ntust_thesis.core.interfaces import Dataset
from ntust_thesis.core.registry import DATASET_REGISTRY
from ntust_thesis.core.types import Sample


class RAMSStubDataset(Dataset):
    """Small in-memory dataset placeholder for RAMS experiments."""

    def name(self) -> str:
        """Dataset name."""
        return "rams"

    def load(self) -> list[Sample]:
        """Load dataset samples."""
        schema = {
            "type": "object",
            "required": ["event_type", "arguments"],
            "additionalProperties": False,
        }
        gold = {"event_type": "Attack", "arguments": {"agent": "rebels"}}
        return [
            Sample(
                sample_id="rams_stub_001",
                input_text="Rebels attacked a checkpoint at dawn.",
                schema=schema,
                gold=gold,
                metadata={"source": "stub"},
            )
        ]


def register() -> None:
    """Register built-in stub datasets."""
    DATASET_REGISTRY.register("rams", RAMSStubDataset)
