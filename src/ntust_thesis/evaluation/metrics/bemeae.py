"""BEMEAE metric implementation."""

from __future__ import annotations

import os
from importlib import import_module
from typing import TYPE_CHECKING, cast

from ntust_thesis.core.interfaces import Metric
from ntust_thesis.evaluation.metrics.common import (
    SBERTEncoder,
    SBERTFactory,
    SBERTUtil,
    SpacyLanguage,
    SpacyLoader,
    SpacyToken,
    f1,
    group_argument_texts_by_role,
    hungarian_match_sum,
    safe_divide_float,
)

if TYPE_CHECKING:
    from ntust_thesis.core.schemas import EvaluationRow


class BEMEAEMetric(Metric):
    """BEMEAE metric with spaCy preprocessing and SBERT cosine similarity."""

    _MODIFIER_DEPS = frozenset(
        {
            "amod",
            "appos",
            "nmod",
            "nounmod",
            "nummod",
            "poss",
            "possessive",
            "compound",
        }
    )
    _LIST_CONNECTORS = frozenset({"and", "&"})
    _SAXON_GENITIVE = frozenset({"'s"})

    def __init__(self) -> None:
        """Initialize lazy spaCy/SBERT resources."""
        self._sbert_model_name = os.getenv("SBERT_MODEL_NAME", "all-mpnet-base-v2")
        self._spacy_model_name = os.getenv("SPACY_MODEL_NAME", "en_core_web_sm")
        self._sbert_model: SBERTEncoder | None = None
        self._sbert_util: SBERTUtil | None = None
        self._spacy_nlp: SpacyLanguage | None = None
        self._embedding_cache: dict[str, object] = {}

    def name(self) -> str:
        """Return metric key."""
        return "bemeae"

    def compute(self, rows: list[EvaluationRow]) -> dict[str, float]:
        """Compute BEMEAE soft precision/recall/F1 over preprocessed arguments."""
        matched_similarity_sum = 0.0
        pred_total = 0
        gold_total = 0

        for row in rows:
            pred_grouped = group_argument_texts_by_role(
                row.parsed_output,
                preprocess=self._preprocess_text,
            )
            gold_grouped = group_argument_texts_by_role(
                row.gold,
                preprocess=self._preprocess_text,
            )

            matched, pred_count, gold_count = self._match_row_hungarian(
                pred_grouped,
                gold_grouped,
            )
            matched_similarity_sum += matched
            pred_total += pred_count
            gold_total += gold_count

        precision = safe_divide_float(matched_similarity_sum, pred_total)
        recall = safe_divide_float(matched_similarity_sum, gold_total)
        return {
            "bemeae_soft_precision": precision,
            "bemeae_soft_recall": recall,
            "bemeae": f1(precision, recall),
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
        """Compute SBERT cosine similarity."""
        self._ensure_sbert_loaded()
        if self._sbert_model is None or self._sbert_util is None:
            msg = "SBERT model is not loaded."
            raise RuntimeError(msg)
        emb_left = self._embed(left)
        emb_right = self._embed(right)
        return float(self._sbert_util.cos_sim(emb_left, emb_right).item())

    def _embed(self, text: str) -> object:
        """Encode one text with cache."""
        cached = self._embedding_cache.get(text)
        if cached is not None:
            return cached
        if self._sbert_model is None:
            msg = "SBERT model is not loaded."
            raise RuntimeError(msg)
        embedding = self._sbert_model.encode(
            text,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        self._embedding_cache[text] = embedding
        return embedding

    def _preprocess_text(self, text: str) -> str:
        """Apply BEMEAE preprocessing on one argument text."""
        self._ensure_spacy_loaded()
        if self._spacy_nlp is None:
            msg = "spaCy model is not loaded."
            raise RuntimeError(msg)

        normalized_tokens = []
        doc = self._spacy_nlp(text)
        for token_obj in doc:
            token = cast("SpacyToken", token_obj)
            if self._should_drop_token(token):
                continue

            normalized = token.lemma_.lower().strip()
            if normalized in {"", "-pron-"}:
                normalized = token.lower_.strip()
            if normalized:
                normalized_tokens.append(normalized)
        normalized_text = " ".join(normalized_tokens).strip()
        if not normalized_text:
            return text.strip()  # fallback if token normalization failed
        return normalized_text

    def _should_drop_token(self, token: SpacyToken) -> bool:
        """Return whether token should be removed by BEMEAE rules."""
        if token.is_punct:
            return True
        if token.pos_ == "DET":
            return True
        if token.lower_ in self._SAXON_GENITIVE:
            return True
        if token.dep_.lower() in self._MODIFIER_DEPS:
            return True
        return token.lower_ in self._LIST_CONNECTORS and token.dep_.lower() == "cc"

    def _ensure_sbert_loaded(self) -> None:
        """Lazily import and initialize sentence-transformers model."""
        if self._sbert_model is not None and self._sbert_util is not None:
            return
        try:
            sentence_transformers = import_module("sentence_transformers")
            sentence_transformer_cls = cast(
                "SBERTFactory", sentence_transformers.SentenceTransformer
            )
            util_module = cast("SBERTUtil", sentence_transformers.util)
        except ImportError as exc:
            msg = (
                "sentence-transformers is required for bemeae. "
                "Install dependencies and rerun."
            )
            raise RuntimeError(msg) from exc
        self._sbert_model = sentence_transformer_cls(self._sbert_model_name)
        self._sbert_util = util_module

    def _ensure_spacy_loaded(self) -> None:
        """Lazily import and initialize spaCy language model."""
        if self._spacy_nlp is not None:
            return
        try:
            spacy_module = import_module("spacy")
            spacy_load = cast("SpacyLoader", spacy_module.load)
        except ImportError as exc:
            msg = "spaCy is required for bemeae. Install dependencies and rerun."
            raise RuntimeError(msg) from exc

        try:
            self._spacy_nlp = spacy_load(self._spacy_model_name)
        except OSError as exc:
            msg = "spaCy model is missing for bemeae."
            raise RuntimeError(msg) from exc
