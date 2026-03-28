"""RAMS dataset loader."""

from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import ValidationError

from ntust_thesis.core.interfaces import Dataset
from ntust_thesis.core.registry import DATASET_REGISTRY
from ntust_thesis.core.schemas import EventOutput, OutputSchema, Sample, SampleMetadata
from ntust_thesis.datasets.rams_models import (
    OntologyEventRoles,
    RAMSDatasetConfig,
    RAMSOntology,
    RAMSRow,
)

_ONTOLOGY_MIN_PARTS = 3
_ONTOLOGY_PAIR_STEP = 2


class RAMSDataset(Dataset):
    """Load RAMS jsonlines into benchmark samples."""

    def __init__(self, config: RAMSDatasetConfig) -> None:
        """Initialize dataset loader from config."""
        self._data_dir = Path(config.data_dir)
        self._ontology_path = Path(config.ontology_path)
        self._split = config.split
        self._max_samples = config.max_samples
        self._ontology = self._load_ontology(self._ontology_path)

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
            for line_no, line in enumerate(f, start=1):
                raw_row = json.loads(line)
                try:
                    row = RAMSRow.model_validate(raw_row)
                except ValidationError as exc:
                    msg = f"Invalid RAMS row at {file_path}:{line_no}: {exc}"
                    raise ValueError(msg) from exc
                sample = self._to_sample(row)
                if sample is None:
                    continue
                samples.append(sample)
                if self._max_samples is not None and len(samples) >= self._max_samples:
                    break
        return samples

    def _to_sample(self, row: RAMSRow) -> Sample | None:
        """Convert one RAMS row to a Sample object."""
        if not row.evt_triggers:
            return None

        trigger = row.evt_triggers[0]
        trigger_span = trigger[0:2]
        trigger_info = trigger[2]
        if not trigger_info:
            return None
        event_type = trigger_info[0][0]
        event_roles = self._ontology.events.get(event_type)
        if event_roles is None:
            msg = (
                f"Event type '{event_type}' not found in ontology "
                f"for doc_key={row.doc_key}."
            )
            raise ValueError(msg)

        tokens = self._flatten_tokens(row.sentences)
        arguments = []
        for link in row.gold_evt_links:
            arg_span = link[1]
            role = _normalize_role_name(link[2])
            text = self._span_to_text(tokens, arg_span[0], arg_span[1])
            arguments.append(
                {
                    "role": role,
                    "text": text,
                }
            )

        gold = EventOutput.model_validate(
            {"event_type": event_type, "arguments": arguments}
        )
        output_schema = OutputSchema()
        marked_sentence = self._mark_trigger(tokens, trigger_span[0], trigger_span[1])
        legal_roles = list(event_roles.roles.keys())
        role_multiplicities = dict(event_roles.roles)
        raw_sentence = marked_sentence
        metadata = SampleMetadata(
            sentence_text=" ".join(tokens),
            marked_sentence=marked_sentence,
            event_type=event_type,
            legal_roles=legal_roles,
            role_multiplicities=role_multiplicities,
        )
        return Sample(
            sample_id=row.doc_key,
            raw_sentence=raw_sentence,
            output_schema=output_schema,
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

    @staticmethod
    def _mark_trigger(tokens: list[str], start: int, end: int) -> str:
        """Mark trigger span with markdown-style double asterisks."""
        if start < 0 or end >= len(tokens) or start > end:
            return " ".join(tokens)
        marked = tokens[:]
        marked[start] = f"**{marked[start]}"
        marked[end] = f"{marked[end]}**"
        return " ".join(marked)

    @staticmethod
    def _load_ontology(path: Path) -> RAMSOntology:
        """Load event-role multiplicities from RAMS ontology file."""
        if not path.exists():
            msg = f"Ontology file not found: {path}"
            raise FileNotFoundError(msg)

        events: dict[str, OntologyEventRoles] = {}
        for line_no, raw_line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if (
                len(parts) < _ONTOLOGY_MIN_PARTS
                or len(parts) % _ONTOLOGY_PAIR_STEP == 0
            ):
                msg = f"Invalid ontology row format at {path}:{line_no}"
                raise ValueError(msg)

            event_type = parts[0]
            role_dict: dict[str, int] = {}
            for idx in range(1, len(parts), _ONTOLOGY_PAIR_STEP):
                role = parts[idx]
                count = int(parts[idx + 1])
                role_dict[role] = count
            event_roles = OntologyEventRoles(
                event_type=event_type,
                roles=role_dict,
            )
            events[event_type] = event_roles
        return RAMSOntology(events=events)


def register() -> None:
    """Register RAMS dataset loader."""
    DATASET_REGISTRY.register("rams", _build_rams_dataset)


def _build_rams_dataset(config: object) -> RAMSDataset:
    """Build RAMS dataset from untyped boundary input."""
    typed_config = RAMSDatasetConfig.model_validate(config)
    return RAMSDataset(config=typed_config)


_ROLE_PREFIX_PATTERN = re.compile(r"^evt\d+arg\d+")


def _normalize_role_name(raw_role: str) -> str:
    """Normalize RAMS role label by removing evt/arg prefix."""
    return _ROLE_PREFIX_PATTERN.sub("", raw_role)
