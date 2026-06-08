"""Quick SBERT similarity checker for two input sentences."""

from __future__ import annotations

import json
import os
import sys
from importlib import import_module

# Edit these values directly before running this script.
SENTENCE_A = r"\"Mrs. Clinton\""
SENTENCE_B = r"Mrs. Clinton"
MODEL_NAME = os.getenv("SBERT_MODEL_NAME", "all-mpnet-base-v2")
PRETTY_PRINT = True


def main() -> None:
    """Run SBERT encoding and similarity comparison."""
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
    model = sentence_transformer_cls(MODEL_NAME)

    embedding_a = model.encode(
        SENTENCE_A,
        convert_to_tensor=False,
        normalize_embeddings=True,
    )
    embedding_b = model.encode(
        SENTENCE_B,
        convert_to_tensor=False,
        normalize_embeddings=True,
    )

    # util.cos_sim accepts array-like and returns shape (1, 1) for single pair.
    similarity = float(util_module.cos_sim([embedding_a], [embedding_b]).item())

    result = {
        "model": MODEL_NAME,
        "sentence_a": SENTENCE_A,
        "sentence_b": SENTENCE_B,
        "cosine_similarity": similarity,
    }

    if PRETTY_PRINT:
        sys.stdout.write(f"{json.dumps(result, ensure_ascii=False, indent=2)}\\n")
    else:
        sys.stdout.write(f"{json.dumps(result, ensure_ascii=False)}\\n")


if __name__ == "__main__":
    main()
