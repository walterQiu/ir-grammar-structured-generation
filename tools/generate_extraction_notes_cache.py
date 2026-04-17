"""Generate test-split extraction-notes cache for two-stage experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ntust_thesis.datasets.rams import RAMSDataset
from ntust_thesis.datasets.rams_models import RAMSDatasetConfig
from ntust_thesis.models.llm.gemini_client import GeminiClient
from ntust_thesis.prompts import build_two_stage_extraction_prompt
from ntust_thesis.utils.env import (
    DEFAULT_DOTENV_PATH,
    get_env_float,
    get_env_int,
    get_env_int_list,
    get_required_env,
)

_DEFAULT_MODEL_NAME = "gemini-3.1-pro-preview"
_DEFAULT_OUTPUT_PATH = Path(
    "outputs/cache/extraction_notes/rams_test_gemini-3.1-pro-preview.jsonl"
)


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate RAMS test extraction-notes cache with Gemini 3.1 pro.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT_PATH,
        help="Output JSONL cache path.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=_DEFAULT_MODEL_NAME,
        help="Gemini model name for extraction generation.",
    )
    parser.add_argument(
        "--api-key-env",
        type=str,
        default="GEMINI_API_KEY",
        help="Env var name for Gemini API key.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it exists.",
    )
    parser.add_argument(
        "--failed-output",
        type=Path,
        default=None,
        help="Output JSON path for failed samples. Default: <output>.failed.json",
    )
    return parser


def main() -> int:
    """Run extraction-note cache generation."""
    args = build_parser().parse_args()

    output_path: Path = args.output
    failed_output_path: Path = (
        args.failed_output
        if args.failed_output is not None
        else output_path.with_suffix(".failed.json")
    )
    if output_path.exists() and not args.overwrite:
        print(  # noqa: T201
            f"[error] output already exists: {output_path}. Use --overwrite to replace."
        )
        return 1
    if failed_output_path.exists() and not args.overwrite:
        print(  # noqa: T201
            f"[error] failed-output already exists: {failed_output_path}. "
            "Use --overwrite to replace."
        )
        return 1

    dotenv_paths = [DEFAULT_DOTENV_PATH]
    api_key = get_required_env(args.api_key_env, fallback_paths=dotenv_paths)
    temperature = get_env_float(
        "llm_temperature", default=0.0, fallback_paths=dotenv_paths
    )
    timeout = get_env_int(
        "llm_timeout_seconds", default=120, fallback_paths=dotenv_paths
    )
    sleep_seconds = get_env_float(
        "llm_sleep_seconds",
        default=1.0,
        fallback_paths=dotenv_paths,
    )
    max_retries = get_env_int("llm_max_retries", default=5, fallback_paths=dotenv_paths)
    backoff_initial_seconds = get_env_float(
        "llm_backoff_initial_seconds",
        default=2.0,
        fallback_paths=dotenv_paths,
    )
    backoff_multiplier = get_env_float(
        "llm_backoff_multiplier",
        default=2.0,
        fallback_paths=dotenv_paths,
    )
    backoff_max_seconds = get_env_float(
        "llm_backoff_max_seconds",
        default=32.0,
        fallback_paths=dotenv_paths,
    )
    retry_http_statuses = tuple(
        get_env_int_list(
            "llm_retry_http_statuses",
            default=[429, 500, 502, 503, 504],
            fallback_paths=dotenv_paths,
        )
    )

    llm = GeminiClient(
        api_key=api_key,
        model_name=args.model_name,
        timeout_seconds=timeout,
        enable_sleep=True,
        sleep_seconds=sleep_seconds,
        enable_retry=True,
        max_retries=max_retries,
        backoff_initial_seconds=backoff_initial_seconds,
        backoff_multiplier=backoff_multiplier,
        backoff_max_seconds=backoff_max_seconds,
        retry_http_statuses=retry_http_statuses,
    )

    dataset = RAMSDataset(
        RAMSDatasetConfig(
            data_dir="datasets/RAMS/data",
            ontology_path="datasets/RAMS/scorer/event_role_multiplicities.txt",
            split="test",
            max_samples=None,
        )
    )
    samples = dataset.load()
    total = len(samples)
    print(f"[info] loaded {total} test samples")  # noqa: T201

    output_path.parent.mkdir(parents=True, exist_ok=True)
    failed_output_path.parent.mkdir(parents=True, exist_ok=True)
    failed_samples: list[dict[str, object]] = []
    success_count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for idx, sample in enumerate(samples, start=1):
            try:
                system_prompt, user_prompt = build_two_stage_extraction_prompt(
                    sentence=sample.raw_sentence,
                    event_type=sample.metadata.event_type,
                    role_multiplicities=sample.metadata.role_multiplicities,
                )
                extraction_text = llm.generate(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=temperature,
                )
                row = {
                    "sample_id": sample.sample_id,
                    "event_type": sample.metadata.event_type,
                    "extraction_text": extraction_text,
                    "model_name": args.model_name,
                }
                f.write(f"{json.dumps(row, ensure_ascii=False)}\n")
                success_count += 1
            except Exception as exc:
                failed_samples.append(
                    {
                        "sample_index": idx,
                        "sample_id": sample.sample_id,
                        "event_type": sample.metadata.event_type,
                        "error": str(exc),
                    }
                )

            if idx % 20 == 0 or idx == total:
                print(f"[progress] generated {idx}/{total}")  # noqa: T201

    failed_payload = {
        "model_name": args.model_name,
        "split": "test",
        "total_samples": total,
        "success_count": success_count,
        "failed_count": len(failed_samples),
        "failed_samples": failed_samples,
    }
    failed_output_path.write_text(
        f"{json.dumps(failed_payload, ensure_ascii=False, indent=2)}\n",
        encoding="utf-8",
    )

    print(f"[done] wrote cache to {output_path}")  # noqa: T201
    print(f"[done] wrote failed samples to {failed_output_path}")  # noqa: T201
    print(f"[summary] success={success_count}, failed={len(failed_samples)}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
