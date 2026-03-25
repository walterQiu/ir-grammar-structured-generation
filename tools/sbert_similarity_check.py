"""Quick SBERT similarity checker for two input sentences."""

from __future__ import annotations

import argparse
import json
import os
import sys
from importlib import import_module


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Encode two sentences with Sentence-BERT and compare similarity.",
    )
    parser.add_argument("sentence_a", type=str, help="First sentence")
    parser.add_argument("sentence_b", type=str, help="Second sentence")
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("SBERT_MODEL_NAME", "all-mpnet-base-v2"),
        help="Sentence-BERT model name",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output",
    )
    return parser


def main() -> None:
    """Run SBERT encoding and similarity comparison."""
    args = build_parser().parse_args()

    try:
        sentence_transformers = import_module("sentence_transformers")
    except ImportError as exc:
        msg = (
            "sentence-transformers is not installed. "
            "Install dependencies first, then rerun."
        )
        raise RuntimeError(msg) from exc

    sentence_transformer_cls = sentence_transformers.SentenceTransformer
    util_module = sentence_transformers.util
    model = sentence_transformer_cls(args.model)

    embedding_a = model.encode(
        args.sentence_a,
        convert_to_tensor=False,
        normalize_embeddings=True,
    )
    embedding_b = model.encode(
        args.sentence_b,
        convert_to_tensor=False,
        normalize_embeddings=True,
    )

    # util.cos_sim accepts array-like and returns shape (1, 1) for single pair.
    similarity = float(util_module.cos_sim([embedding_a], [embedding_b]).item())

    result = {
        "model": args.model,
        "sentence_a": args.sentence_a,
        "sentence_b": args.sentence_b,
        "cosine_similarity": similarity,
    }

    if args.pretty:
        sys.stdout.write(f"{json.dumps(result, ensure_ascii=False, indent=2)}\\n")
    else:
        sys.stdout.write(f"{json.dumps(result, ensure_ascii=False)}\\n")


if __name__ == "__main__":
    main()
