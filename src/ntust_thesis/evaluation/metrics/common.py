"""Shared helpers and typing protocols for evaluation metrics."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from importlib import import_module
from typing import TYPE_CHECKING, Protocol

from ntust_thesis.core.role_path import role_to_path

if TYPE_CHECKING:
    from ntust_thesis.core.schemas import EventOutput


class CosineScore(Protocol):
    """Minimal cosine result protocol."""

    def item(self) -> float:
        """Return scalar value."""
        ...


class SBERTUtil(Protocol):
    """Minimal sentence-transformers util protocol."""

    def cos_sim(self, left: object, right: object) -> CosineScore:
        """Return cosine similarity scalar wrapper."""
        ...


class SBERTEncoder(Protocol):
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


SBERTFactory = Callable[[str], SBERTEncoder]


class SpacyToken(Protocol):
    """Minimal spaCy token protocol."""

    text: str
    lower_: str
    lemma_: str
    dep_: str
    pos_: str
    is_punct: bool


class SpacyDoc(Protocol):
    """Minimal spaCy doc protocol."""

    def __iter__(self) -> Iterator[SpacyToken]:
        """Iterate over spaCy tokens."""
        ...


class SpacyLanguage(Protocol):
    """Minimal spaCy language pipeline protocol."""

    def __call__(self, text: str) -> SpacyDoc:
        """Parse text into spaCy doc."""
        ...


SpacyLoader = Callable[[str], SpacyLanguage]


def safe_divide(numerator: int, denominator: int) -> float:
    """Return safe division result under zero-denominator."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def safe_divide_float(numerator: float, denominator: int) -> float:
    """Return safe float division result under zero-denominator."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def f1(precision: float, recall: float) -> float:
    """Compute harmonic mean under zero edge case."""
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def group_argument_spans_by_role(
    output: EventOutput | None,
    *,
    preprocess: Callable[[str], str] | None = None,
) -> dict[str, list[str]]:
    """Group argument spans by role for role-constrained matching."""
    if output is None:
        return {}

    grouped: dict[str, list[str]] = {}
    for argument in output.arguments:
        role = role_to_path(argument.role) or "unknown_role"
        span = argument.span.strip()
        if preprocess is not None:
            span = preprocess(span)
        grouped.setdefault(role, []).append(span)

    return grouped


def hungarian_match_sum(
    pred_spans: list[str],
    gold_spans: list[str],
    score_fn: Callable[[str, str], float],
) -> float:
    """Return max total score under one-to-one Hungarian matching."""
    scipy_optimize = import_module("scipy.optimize")
    linear_sum_assignment = scipy_optimize.linear_sum_assignment

    similarity_matrix = [
        [score_fn(pred, gold) for gold in gold_spans] for pred in pred_spans
    ]
    cost_matrix = [[1.0 - score for score in row] for row in similarity_matrix]
    row_ids, col_ids = linear_sum_assignment(cost_matrix)

    total = 0.0
    for row_idx, col_idx in zip(row_ids.tolist(), col_ids.tolist(), strict=True):
        total += similarity_matrix[row_idx][col_idx]
    return total
