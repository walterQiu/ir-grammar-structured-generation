"""Recompute strict ECAR from saved experiment predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

BATCH_RUN_LIST_PATH = Path("experimental_configs/batch_run_list.yaml")
RUNS_DIR = Path("outputs/runs")
OUTPUT_PATH = Path("outputs/analysis/recomputed_ecar.json")
METRIC_DECIMAL_PLACES = 5


def _read_active_config_paths(batch_path: Path) -> list[Path]:
    """Read active experiment config paths from batch_run_list.yaml."""
    payload = yaml.safe_load(batch_path.read_text(encoding="utf-8"))
    experiments = payload.get("experiments") if isinstance(payload, dict) else None
    if not isinstance(experiments, list):
        msg = f"Invalid batch run list: {batch_path}"
        raise TypeError(msg)

    config_paths: list[Path] = []
    for item in experiments:
        if not isinstance(item, str):
            msg = f"Invalid experiment config entry in {batch_path}: {item!r}"
            raise TypeError(msg)
        config_paths.append(Path(item))
    return config_paths


def _read_experiment_name(config_path: Path) -> str:
    """Read experiment_name from one experiment config."""
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment_name = (
        payload.get("experiment_name") if isinstance(payload, dict) else None
    )
    if not isinstance(experiment_name, str) or not experiment_name.strip():
        msg = f"Missing experiment_name in config: {config_path}"
        raise ValueError(msg)
    return experiment_name


def _find_latest_run_dir(experiment_name: str) -> Path:
    """Find the latest run directory for an experiment name."""
    candidates = sorted(
        path
        for path in RUNS_DIR.iterdir()
        if path.is_dir() and path.name.endswith(f"_{experiment_name}")
    )
    if not candidates:
        msg = f"Run directory not found for experiment: {experiment_name}"
        raise FileNotFoundError(msg)
    return candidates[-1]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into dictionaries."""
    rows: list[dict[str, Any]] = []
    for line_no, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            msg = f"Invalid JSON at {path}:{line_no}: {exc}"
            raise ValueError(msg) from exc
        if not isinstance(row, dict):
            msg = f"Expected JSON object at {path}:{line_no}"
            raise TypeError(msg)
        rows.append(row)
    return rows


def _read_failed_sample_count(run_dir: Path) -> int:
    """Return failed sample count from failed_samples.json when available."""
    path = run_dir / "failed_samples.json"
    if not path.exists():
        return 0
    payload = json.loads(path.read_text(encoding="utf-8"))
    failed_samples = (
        payload.get("failed_samples") if isinstance(payload, dict) else None
    )
    return len(failed_samples) if isinstance(failed_samples, list) else 0


def _read_original_ecar(run_dir: Path) -> float | None:
    """Best-effort lookup of the original ECAR value in metrics.json."""
    path = run_dir / "metrics.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = _find_key_recursive(payload, "ecar")
    return value if isinstance(value, float | int) else None


def _find_key_recursive(value: Any, target_key: str) -> Any:  # noqa: ANN401
    """Return first matching key from a nested JSON-like value."""
    if isinstance(value, dict):
        if target_key in value:
            return value[target_key]
        for child in value.values():
            found = _find_key_recursive(child, target_key)
            if found is not None:
                return found
    if isinstance(value, list):
        for child in value:
            found = _find_key_recursive(child, target_key)
            if found is not None:
                return found
    return None


def _recompute_strict_ecar(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute ECAR only when gold/pred are empty and IR compiled successfully."""
    gold_empty_count = 0
    correct_abstention_count = 0
    fallback_empty_count = 0

    for row in rows:
        if not _has_empty_arguments(row.get("gold")):
            continue
        gold_empty_count += 1

        parsed_is_empty = _has_empty_arguments(row.get("parsed_output"))
        compile_error = _compile_error(row)
        compiled_successfully = compile_error is None

        if parsed_is_empty and compiled_successfully:
            correct_abstention_count += 1
        elif parsed_is_empty and not compiled_successfully:
            fallback_empty_count += 1

    ecar = 0.0
    if gold_empty_count > 0:
        ecar = correct_abstention_count / gold_empty_count

    return {
        "ecar": round(ecar, METRIC_DECIMAL_PLACES),
        "correct_abstention_count": correct_abstention_count,
        "gold_empty_count": gold_empty_count,
        "parse_error_fallback_empty_count": fallback_empty_count,
    }


def _has_empty_arguments(output: object) -> bool:
    """Return whether an EventOutput-like dict has an empty arguments list."""
    if not isinstance(output, dict):
        return False
    arguments = output.get("arguments")
    return isinstance(arguments, list) and len(arguments) == 0


def _compile_error(row: dict[str, Any]) -> object:
    """Return prediction_metadata.compile_error."""
    metadata = row.get("prediction_metadata")
    if not isinstance(metadata, dict):
        return "missing prediction_metadata"
    return metadata.get("compile_error")


def _build_report() -> dict[str, Any]:
    """Build strict ECAR report for all active batch experiments."""
    results: list[dict[str, Any]] = []
    config_paths = _read_active_config_paths(BATCH_RUN_LIST_PATH)

    for config_path in config_paths:
        experiment_name = _read_experiment_name(config_path)
        run_dir = _find_latest_run_dir(experiment_name)
        predictions_path = run_dir / "predictions.jsonl"
        rows = _read_jsonl(predictions_path)
        recomputed = _recompute_strict_ecar(rows)
        original_ecar = _read_original_ecar(run_dir)

        result = {
            "experiment_name": experiment_name,
            "config_yaml": str(config_path),
            "run_dir": str(run_dir),
            "predictions_jsonl": str(predictions_path),
            "prediction_row_count": len(rows),
            "failed_samples_count": _read_failed_sample_count(run_dir),
            **recomputed,
        }
        if original_ecar is not None:
            result["original_ecar"] = round(float(original_ecar), METRIC_DECIMAL_PLACES)
        results.append(result)

    return {
        "batch_run_list": str(BATCH_RUN_LIST_PATH),
        "runs_dir": str(RUNS_DIR),
        "experiment_count": len(results),
        "definition": (
            "strict_ecar = count(gold.arguments == [] and parsed_output.arguments == [] "
            "and prediction_metadata.compile_error is null) / "
            "count(gold.arguments == [])"
        ),
        "experiments": results,
    }


def main() -> int:
    """Recompute strict ECAR and write one JSON report."""
    report = _build_report()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        f"{json.dumps(report, ensure_ascii=False, indent=2)}\n",
        encoding="utf-8",
    )
    print(f"[done] wrote recomputed ECAR report to {OUTPUT_PATH}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
