"""SBERT-based content similarity metric."""

from __future__ import annotations

import os
from importlib import import_module
from typing import TYPE_CHECKING, cast

from ntust_thesis.core.interfaces import Metric
from ntust_thesis.evaluation.metrics.common import (
    SBERTEncoder,
    SBERTFactory,
    SBERTUtil,
    f1,
    group_argument_texts_by_role,
    hungarian_match_sum,
    safe_divide_float,
)

if TYPE_CHECKING:
    from ntust_thesis.core.schemas import EvaluationRow


class ContentSimilaritySBERTMetric(Metric):
    """Soft content similarity using SBERT cosine similarity."""

    def __init__(self) -> None:
        """Initialize lazy SBERT resources."""
        self._model_name = os.getenv("SBERT_MODEL_NAME", "all-mpnet-base-v2")
        self._model: SBERTEncoder | None = None
        self._util: SBERTUtil | None = None
        self._embedding_cache: dict[str, object] = {}

    def name(self) -> str:
        """Return metric key."""
        return "content_similarity_sbert"

    def compute(self, rows: list[EvaluationRow]) -> dict[str, float]:
        """Compute micro soft precision/recall/F1 with Hungarian matching."""
        matched_similarity_sum = 0.0
        pred_total = 0
        gold_total = 0

        for row in rows:
            pred_grouped = group_argument_texts_by_role(row.parsed_output)
            gold_grouped = group_argument_texts_by_role(row.gold)

            matched, pred_count, gold_count = self._match_row_hungarian(
                pred_grouped,
                gold_grouped,
            )
            matched_similarity_sum += matched
            pred_total += pred_count
            gold_total += gold_count

        soft_precision = safe_divide_float(matched_similarity_sum, pred_total)
        soft_recall = safe_divide_float(matched_similarity_sum, gold_total)
        return {
            "sbert_cossim_precision": soft_precision,
            "sbert_cossim_recall": soft_recall,
            "sbert_cossim_f1": f1(soft_precision, soft_recall),
        }

    def _match_row_hungarian(
        self,
        pred_grouped: dict[str, list[str]],
        gold_grouped: dict[str, list[str]],
    ) -> tuple[float, int, int]:
        """Compute one row soft-TP sum with Hungarian matching by role."""
        total_pred = sum(len(items) for items in pred_grouped.values())
        total_gold = sum(len(items) for items in gold_grouped.values())
        matched_sum = 0.0

        roles = set(pred_grouped) | set(gold_grouped)
        for role in roles:
            pred_texts = pred_grouped.get(role, [])
            gold_texts = gold_grouped.get(role, [])
            if not pred_texts or not gold_texts:
                continue
            matched_sum += hungarian_match_sum(
                pred_texts,
                gold_texts,
                self._semantic_similarity,
            )

        return matched_sum, total_pred, total_gold

    def _semantic_similarity(self, left: str, right: str) -> float:
        """Compute cosine similarity from SBERT embeddings."""
        self._ensure_model_loaded()
        if self._model is None or self._util is None:
            msg = "SBERT model is not loaded."
            raise RuntimeError(msg)

        emb_left = self._embed(left)
        emb_right = self._embed(right)
        return float(self._util.cos_sim(emb_left, emb_right).item())

    def _embed(self, text: str) -> object:
        """Encode text once and reuse cached embedding."""
        cached = self._embedding_cache.get(text)
        if cached is not None:
            return cached
        if self._model is None:
            msg = "SBERT model is not loaded."
            raise RuntimeError(msg)
        embedding = self._model.encode(
            text,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        self._embedding_cache[text] = embedding
        return embedding

    def _ensure_model_loaded(self) -> None:
        """Lazily import and initialize sentence-transformers model."""
        if self._model is not None and self._util is not None:
            return
        try:
            sentence_transformers = import_module("sentence_transformers")
            sentence_transformer_cls = cast(
                "SBERTFactory", sentence_transformers.SentenceTransformer
            )
            util_module = cast("SBERTUtil", sentence_transformers.util)
        except ImportError as exc:
            msg = (
                "sentence-transformers is required for content_similarity_sbert. "
                "Install dependencies and rerun."
            )
            raise RuntimeError(msg) from exc
        self._model = sentence_transformer_cls(self._model_name)
        self._util = util_module
