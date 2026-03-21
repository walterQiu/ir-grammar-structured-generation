"""Pydantic models for RAMS dataset config and raw rows."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

TriggerTypeScore = tuple[str, float]
TriggerEntry = tuple[int, int, list[TriggerTypeScore]]
Span = tuple[int, int]
GoldEventLink = tuple[Span, Span, str]


class RAMSDatasetConfig(BaseModel):
    """Validated runtime config for RAMS dataset loader."""

    model_config = ConfigDict(extra="ignore")

    data_dir: str = "datasets/RAMS/data"
    ontology_path: str = "datasets/RAMS/scorer/event_role_multiplicities.txt"
    split: Literal["train", "dev", "test"] = "train"
    max_samples: int | None = Field(default=None, ge=1)


class RAMSRow(BaseModel):
    """Validated RAMS JSONL row used by the loader."""

    model_config = ConfigDict(extra="ignore")

    doc_key: str
    sentences: list[list[str]]
    evt_triggers: list[TriggerEntry]
    gold_evt_links: list[GoldEventLink] = Field(default_factory=list)


class OntologyEventRoles(BaseModel):
    """Role multiplicities for one event type."""

    model_config = ConfigDict(extra="forbid")

    event_type: str
    roles: dict[str, int]  # dict[role name, max multiplicity]


class RAMSOntology(BaseModel):
    """Validated ontology container indexed by event type."""

    model_config = ConfigDict(extra="forbid")

    events: dict[str, OntologyEventRoles]  # dict[event type, OntologyEventRoles]
