"""Quick BEMEAE similarity checker for spans or grouped role spans."""

from __future__ import annotations

import json
import os
import sys

from ntust_thesis.evaluation.metrics.bemeae import BEMEAEMetric
from ntust_thesis.evaluation.metrics.common import f1, safe_divide_float

# Edit these values directly before running this script.
CHECK_MODE = "grouped"  # one of: span, grouped
SENTENCE_A = r"the agreement"
SENTENCE_B = r"agreement"
GROUP_A = {
    "participant": ["Hillary", "Alinsky"],
}
GROUP_B = {
    "participant": ["Hillary Clinton", "Saul Alinsky"],
}
SBERT_MODEL_NAME = os.getenv("SBERT_MODEL_NAME", "all-mpnet-base-v2")
SPACY_MODEL_NAME = os.getenv("SPACY_MODEL_NAME", "en_core_web_sm")
PRETTY_PRINT = True


def _normalize_grouped_spans(
    grouped_spans: dict[str, list[str]],
    metric: BEMEAEMetric,
) -> dict[str, list[str]]:
    """Apply BEMEAE span preprocessing to a grouped role-span mapping."""
    normalized: dict[str, list[str]] = {}
    for role, spans in grouped_spans.items():
        normalized[role] = [metric._preprocess_text(span) for span in spans]
    return normalized


def _run_span_check(metric: BEMEAEMetric) -> dict[str, object]:
    """Run BEMEAE normalization and similarity comparison for two spans."""
    normalized_a = metric._preprocess_text(SENTENCE_A)
    normalized_b = metric._preprocess_text(SENTENCE_B)
    similarity = metric._semantic_similarity(
        normalized_a,
        normalized_b,
    )

    return {
        "mode": "span",
        "metric": metric.name(),
        "sbert_model": SBERT_MODEL_NAME,
        "spacy_model": SPACY_MODEL_NAME,
        "sentence_a": SENTENCE_A,
        "sentence_b": SENTENCE_B,
        "normalized_sentence_a": normalized_a,
        "normalized_sentence_b": normalized_b,
        "bemeae_cosine_similarity": similarity,
    }


def _run_grouped_check(metric: BEMEAEMetric) -> dict[str, object]:
    """Run BEMEAE Hungarian matching for grouped role spans."""
    normalized_a = _normalize_grouped_spans(GROUP_A, metric)
    normalized_b = _normalize_grouped_spans(GROUP_B, metric)
    matched_similarity_sum, pred_total, gold_total = metric._match_row_hungarian(
        normalized_a,
        normalized_b,
    )
    precision = safe_divide_float(matched_similarity_sum, pred_total)
    recall = safe_divide_float(matched_similarity_sum, gold_total)

    return {
        "mode": "grouped",
        "metric": metric.name(),
        "sbert_model": SBERT_MODEL_NAME,
        "spacy_model": SPACY_MODEL_NAME,
        "group_a": GROUP_A,
        "group_b": GROUP_B,
        "normalized_group_a": normalized_a,
        "normalized_group_b": normalized_b,
        "matched_similarity_sum": matched_similarity_sum,
        "pred_total": pred_total,
        "gold_total": gold_total,
        "bemeae_soft_precision": precision,
        "bemeae_soft_recall": recall,
        "bemeae": f1(precision, recall),
    }


def main() -> None:
    """Run configured BEMEAE comparison."""
    metric = BEMEAEMetric()
    if CHECK_MODE == "span":
        result = _run_span_check(metric)
    elif CHECK_MODE == "grouped":
        result = _run_grouped_check(metric)
    else:
        msg = f"Unsupported CHECK_MODE: {CHECK_MODE}"
        raise ValueError(msg)

    if PRETTY_PRINT:
        sys.stdout.write(f"{json.dumps(result, ensure_ascii=False, indent=2)}\n")
    else:
        sys.stdout.write(f"{json.dumps(result, ensure_ascii=False)}\n")


if __name__ == "__main__":
    main()
