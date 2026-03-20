"""RAMS dataset loader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ntust_thesis.core.interfaces import Dataset
from ntust_thesis.core.registry import DATASET_REGISTRY
from ntust_thesis.core.types import JSONDict, Sample


class RAMSDataset(Dataset):
    """Load RAMS jsonlines into benchmark samples."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize dataset loader from config."""
        cfg = config or {}
        self._data_dir = Path(cfg.get("data_dir", "datasets/RAMS/data"))
        self._split = str(cfg.get("split", "train"))
        max_samples = cfg.get("max_samples")
        self._max_samples = int(max_samples) if max_samples is not None else None

    def name(self) -> str:
        """Return dataset key."""
        return "rams"

    def load(self) -> list[Sample]:
        """Load RAMS split as benchmark samples."""
        file_path = self._data_dir / f"{self._split}.jsonlines"
        if not file_path.exists():
            msg = f"RAMS split file not found: {file_path}"
            raise FileNotFoundError(msg)

        samples: list[Sample] = []
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                sample = self._to_sample(row)
                if sample is None:
                    continue
                samples.append(sample)
                if self._max_samples is not None and len(samples) >= self._max_samples:
                    break
        return samples

    def _to_sample(self, row: JSONDict) -> Sample | None:
        """Convert one RAMS row to a Sample object."""
        triggers = row.get("evt_triggers", [])
        if not triggers:
            return None

        trigger = triggers[0]
        trigger_span = trigger[0:2]
        trigger_info = trigger[2]
        event_type = trigger_info[0][0]

        tokens = self._flatten_tokens(row["sentences"])
        arguments = []
        for link in row.get("gold_evt_links", []):
            arg_span = link[1]
            role = link[2]
            text = self._span_to_text(tokens, arg_span[0], arg_span[1])
            arguments.append(
                {
                    "role": role,
                    "text": text,
                    "span": [arg_span[0], arg_span[1]],
                }
            )

        gold = {"event_type": event_type, "arguments": arguments}
        schema = {
            "type": "object",
            "required": ["event_type", "arguments"],
            "additionalProperties": False,
        }
        input_text = " ".join(tokens)
        metadata = {
            "doc_key": row.get("doc_key"),
            "split": row.get("split", self._split),
            "trigger_span": trigger_span,
            "source_url": row.get("source_url"),
        }
        return Sample(
            sample_id=str(row.get("doc_key")),
            input_text=input_text,
            schema=schema,
            gold=gold,
            metadata=metadata,
        )

    @staticmethod
    def _flatten_tokens(sentences: list[list[str]]) -> list[str]:
        """Flatten nested sentence tokens into document tokens."""
        return [token for sent in sentences for token in sent]

    @staticmethod
    def _span_to_text(tokens: list[str], start: int, end: int) -> str:
        """Convert inclusive token span to display text."""
        if start < 0 or end >= len(tokens) or start > end:
            return ""
        return " ".join(tokens[start : end + 1])


def register() -> None:
    """Register RAMS dataset loader."""
    DATASET_REGISTRY.register("rams", RAMSDataset)
