"""Extraction-note cache for two-stage extraction models."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path


class ExtractionNotesCache:
    """Model-specific cache for extraction notes keyed by sample_id."""

    def __init__(
        self,
        *,
        extraction_backend: str,
        extraction_llm_name: str,
        extraction_api_base: str | None,
    ) -> None:
        """Initialize cache path and preload existing rows."""
        self._path = _resolve_extraction_cache_path(
            extraction_backend=extraction_backend,
            extraction_llm_name=extraction_llm_name,
            extraction_api_base=extraction_api_base,
        )
        self._items = _read_extraction_cache_jsonl(self._path)

    @property
    def path(self) -> Path:
        """Return underlying cache file path."""
        return self._path

    @property
    def size(self) -> int:
        """Return number of cached samples currently in memory."""
        return len(self._items)

    def get(self, sample_id: str) -> str | None:
        """Return cached extraction text by sample id when present."""
        return self._items.get(sample_id)

    def put(self, sample_id: str, extraction_text: str) -> None:
        """Append one extraction result to cache and update in-memory map."""
        _append_extraction_cache_row(
            path=self._path,
            sample_id=sample_id,
            extraction_text=extraction_text,
        )
        self._items[sample_id] = extraction_text


def _resolve_extraction_cache_path(
    *,
    extraction_backend: str,
    extraction_llm_name: str,
    extraction_api_base: str | None,
) -> Path:
    """Resolve extraction cache path from env override or model identity."""
    cache_path_str = os.getenv("EXTRACTION_NOTES_CACHE_PATH")
    if cache_path_str and cache_path_str.strip():
        return Path(cache_path_str.strip())

    backend_key = _cache_name_part(extraction_backend)
    llm_key = _cache_name_part(extraction_llm_name)
    parts = [backend_key, llm_key]
    if extraction_api_base and extraction_api_base.strip():
        parts.append(_cache_name_part(extraction_api_base))
    file_name = "__".join(parts) + ".jsonl"
    return Path("outputs/cache/extraction_notes") / file_name


def _read_extraction_cache_jsonl(path: Path) -> dict[str, str]:
    """Read extraction cache jsonl into sample_id->text map."""
    if not path.exists():
        return {}

    cache: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        sample_id = row.get("sample_id")
        extraction_text = row.get("extraction_text")
        if not isinstance(sample_id, str) or not sample_id.strip():
            continue
        if not isinstance(extraction_text, str):
            continue
        cache[sample_id] = extraction_text

    return cache


def _append_extraction_cache_row(
    *,
    path: Path,
    sample_id: str,
    extraction_text: str,
) -> None:
    """Append one extraction result into cache jsonl file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"sample_id": sample_id, "extraction_text": extraction_text}
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{json.dumps(row, ensure_ascii=False)}\n")


def _cache_name_part(raw: str) -> str:
    """Normalize one cache file-name segment."""
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", raw.strip())
    normalized = normalized.strip("-")
    return normalized or "unknown"
