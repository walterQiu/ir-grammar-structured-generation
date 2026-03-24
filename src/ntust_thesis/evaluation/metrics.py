"""Aggregate metrics for strict evaluation."""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Callable
from importlib import import_module
from typing import TYPE_CHECKING, Protocol, cast

from ntust_thesis.core.interfaces import Metric
from ntust_thesis.core.registry import METRIC_REGISTRY

if TYPE_CHECKING:
    from ntust_thesis.core.schemas import EvaluationRow, EventOutput


class _CosineScore(Protocol):
    """Minimal cosine result protocol."""

    def item(self) -> float:
        """Return scalar value."""
        ...


class _SBERTUtil(Protocol):
    """Minimal sentence-transformers util protocol."""

    def cos_sim(self, left: object, right: object) -> _CosineScore:
        """Return cosine similarity scalar wrapper."""
        ...


class _SBERTEncoder(Protocol):
    """Minimal sentence-transformers encoder protocol."""

    def encode(
        self,
        text: str,
        *,
        convert_to_tensor: bool,
        normalize_embeddings: bool,
    ) -> object:
        """Encode text into embedding tensor."""
        ...


_SBERTFactory = Callable[[str], _SBERTEncoder]


class StrictRatesMetric(Metric):
    """Compute strict benchmark rates from row-level flags."""

    def name(self) -> str:
        """Return metric key."""
        return "strict_rates"

    def compute(self, rows: list[EvaluationRow]) -> dict[str, float]:
        """Compute json/schema/exact-match rates."""
        total = len(rows)
        if total == 0:
            return {
                "json_valid_rate": 0.0,
                "schema_valid_rate": 0.0,
                "exact_match_rate": 0.0,
            }

        json_valid = sum(1 for row in rows if row.json_valid)
        schema_valid = sum(1 for row in rows if row.schema_valid)
        exact = sum(1 for row in rows if row.exact_match)
        return {
            "json_valid_rate": json_valid / total,
            "schema_valid_rate": schema_valid / total,
            "exact_match_rate": exact / total,
        }


def _safe_divide(numerator: int, denominator: int) -> float:
    """Return safe division result under zero-denominator."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _f1(precision: float, recall: float) -> float:
    """Compute harmonic mean under zero edge case."""
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


class ArgIF1Metric(Metric):
    """Argument Identification F1: match by argument text only."""

    def name(self) -> str:
        """Return metric key."""
        return "arg_i_f1"

    def compute(self, rows: list[EvaluationRow]) -> dict[str, float]:
        """Compute micro Precision/Recall/F1 for text-only matching."""
        tp = 0
        pred_total = 0
        gold_total = 0

        for row in rows:
            pred_args = (
                row.parsed_output.arguments if row.parsed_output is not None else []
            )
            gold_args = row.gold.arguments

            pred_texts = Counter(arg.text.strip() for arg in pred_args)
            gold_texts = Counter(arg.text.strip() for arg in gold_args)

            pred_total += sum(pred_texts.values())
            gold_total += sum(gold_texts.values())
            tp += sum((pred_texts & gold_texts).values())

        precision = _safe_divide(tp, pred_total)
        recall = _safe_divide(tp, gold_total)
        return {
            "arg_i_precision": precision,
            "arg_i_recall": recall,
            "arg_i_f1": _f1(precision, recall),
        }


class ArgCF1Metric(Metric):
    """Argument Classification F1: match by role and argument text."""

    def name(self) -> str:
        """Return metric key."""
        return "arg_c_f1"

    def compute(self, rows: list[EvaluationRow]) -> dict[str, float]:
        """Compute micro Precision/Recall/F1 for (role, text) matching."""
        tp = 0
        pred_total = 0
        gold_total = 0

        for row in rows:
            pred_args = (
                row.parsed_output.arguments if row.parsed_output is not None else []
            )
            gold_args = row.gold.arguments

            pred_items = Counter((arg.role, arg.text.strip()) for arg in pred_args)
            gold_items = Counter((arg.role, arg.text.strip()) for arg in gold_args)

            pred_total += sum(pred_items.values())
            gold_total += sum(gold_items.values())
            tp += sum((pred_items & gold_items).values())

        precision = _safe_divide(tp, pred_total)
        recall = _safe_divide(tp, gold_total)
        return {
            "arg_c_precision": precision,
            "arg_c_recall": recall,
            "arg_c_f1": _f1(precision, recall),
        }


class ContentSimilaritySBERTMetric(Metric):
    """Soft content similarity using SBERT cosine similarity."""

    def __init__(self) -> None:
        """Initialize lazy SBERT resources."""
        self._model_name = os.getenv("SBERT_MODEL_NAME", "all-MiniLM-L6-v2")
        self._model: _SBERTEncoder | None = None
        self._util: _SBERTUtil | None = None
        self._embedding_cache: dict[str, object] = {}

    def name(self) -> str:
        """Return metric key."""
        return "content_similarity_sbert"

    def compute(self, rows: list[EvaluationRow]) -> dict[str, float]:
        """Compute soft precision/recall/F1 from role-keyed argument texts."""
        pred_score_sum = 0.0
        pred_key_count = 0
        gold_score_sum = 0.0
        gold_key_count = 0

        for row in rows:
            pred_map = _flatten_argument_text_slots(row.parsed_output)
            gold_map = _flatten_argument_text_slots(row.gold)

            pred_score_sum += self._average_directional_similarity(pred_map, gold_map)
            pred_key_count += 1
            gold_score_sum += self._average_directional_similarity(gold_map, pred_map)
            gold_key_count += 1

        soft_precision = _safe_divide_float(pred_score_sum, pred_key_count)
        soft_recall = _safe_divide_float(gold_score_sum, gold_key_count)
        return {
            "content_soft_precision": soft_precision,
            "content_soft_recall": soft_recall,
            "content_similarity": _f1(soft_precision, soft_recall),
        }

    def _average_directional_similarity(
        self,
        source: dict[str, str],
        target: dict[str, str],
    ) -> float:
        """Average directional similarity with missing-key zero contribution."""
        if not source:
            return 0.0
        scores = []
        for key, source_value in source.items():
            target_value = target.get(key)
            if target_value is None:
                scores.append(0.0)
                continue
            scores.append(self._semantic_similarity(source_value, target_value))
        return sum(scores) / len(scores)

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
                "_SBERTFactory", sentence_transformers.SentenceTransformer
            )
            util_module = cast("_SBERTUtil", sentence_transformers.util)
        except ImportError as exc:
            msg = (
                "sentence-transformers is required for content_similarity_sbert. "
                "Install dependencies and rerun."
            )
            raise RuntimeError(msg) from exc
        self._model = sentence_transformer_cls(self._model_name)
        self._util = util_module


def _flatten_argument_text_slots(output: EventOutput | None) -> dict[str, str]:
    """Flatten EventOutput into role-keyed text slots for semantic comparison."""
    if output is None:
        return {}

    by_role: dict[str, list[str]] = {}
    for argument in output.arguments:
        role = argument.role.strip()
        text = argument.text.strip()
        if not role:
            role = "unknown_role"
        by_role.setdefault(role, []).append(text)

    slot_map: dict[str, str] = {}
    for role in sorted(by_role):
        texts = sorted(by_role[role])
        for idx, text in enumerate(texts):
            slot_map[f"arguments.{role}.{idx}"] = str(text)
    return slot_map


def _safe_divide_float(numerator: float, denominator: int) -> float:
    """Return safe float division result under zero-denominator."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def register() -> None:
    """Register built-in strict rates metric."""
    METRIC_REGISTRY.register("strict_rates", StrictRatesMetric)
    METRIC_REGISTRY.register("arg_i_f1", ArgIF1Metric)
    METRIC_REGISTRY.register("arg_c_f1", ArgCF1Metric)
    METRIC_REGISTRY.register("content_similarity_sbert", ContentSimilaritySBERTMetric)
