"""Compare parsed outputs between two experiment runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Edit these paths directly before running this script.
EXPERIMENT_DIR_A = Path("outputs/runs/20260417_142855_M1_json_gemini3")
EXPERIMENT_DIR_B = Path("outputs/runs/20260418_081505_M1_code4struct_gemini3")
OUTPUT_PATH = Path("outputs/analysis/diff_predictions.json")
COMPARE_MODE = "role_only"  # one of: full, role_only


def _read_predictions(predictions_path: Path) -> dict[str, dict[str, Any]]:
    """Read predictions.jsonl and index by sample_id."""
    if not predictions_path.exists():
        msg = f"predictions.jsonl not found: {predictions_path}"
        raise FileNotFoundError(msg)

    rows: dict[str, dict[str, Any]] = {}
    for line_no, raw_line in enumerate(
        predictions_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            msg = f"Invalid JSON at {predictions_path}:{line_no}: {exc}"
            raise ValueError(msg) from exc

        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id.strip():
            msg = f"Missing/invalid sample_id at {predictions_path}:{line_no}"
            raise ValueError(msg)
        rows[sample_id] = row
    return rows


def _build_differences(
    rows_a: dict[str, dict[str, Any]],
    rows_b: dict[str, dict[str, Any]],
    *,
    compare_mode: str,
) -> list[dict[str, Any]]:
    """Return samples whose parsed_output differs between two runs."""
    common_ids = sorted(set(rows_a) & set(rows_b))
    differences: list[dict[str, Any]] = []

    for sample_id in common_ids:
        row_a = rows_a[sample_id]
        row_b = rows_b[sample_id]
        parsed_a = row_a.get("parsed_output")
        parsed_b = row_b.get("parsed_output")
        if _is_same_prediction(parsed_a, parsed_b, compare_mode=compare_mode):
            continue

        gold_a = row_a.get("gold")
        gold_b = row_b.get("gold")
        gold = gold_a if gold_a == gold_b else {"exp_a": gold_a, "exp_b": gold_b}

        differences.append(
            {
                "sample_id": sample_id,
                "parsed_output_exp_a": parsed_a,
                "parsed_output_exp_b": parsed_b,
                "gold": gold,
            }
        )

    return differences


def _is_same_prediction(
    parsed_a: object,
    parsed_b: object,
    *,
    compare_mode: str,
) -> bool:
    """Compare parsed outputs according to selected mode."""
    if compare_mode == "full":
        return parsed_a == parsed_b
    if compare_mode == "role_only":
        return _role_signature(parsed_a) == _role_signature(parsed_b)
    msg = f"Unsupported COMPARE_MODE: {compare_mode}"
    raise ValueError(msg)


def _role_signature(parsed_output: object) -> list[str]:
    """Return sorted role multiset signature from parsed_output."""
    if not isinstance(parsed_output, dict):
        return []
    arguments = parsed_output.get("arguments")
    if not isinstance(arguments, list):
        return []

    roles: list[str] = []
    for item in arguments:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        if not isinstance(role, str):
            continue
        normalized = role.strip()
        if normalized:
            roles.append(normalized)
    return sorted(roles)


def main() -> int:
    """Compare two runs and write differing samples into one JSON file."""
    pred_path_a = EXPERIMENT_DIR_A / "predictions.jsonl"
    pred_path_b = EXPERIMENT_DIR_B / "predictions.jsonl"
    rows_a = _read_predictions(pred_path_a)
    rows_b = _read_predictions(pred_path_b)

    ids_a = set(rows_a)
    ids_b = set(rows_b)
    only_a = sorted(ids_a - ids_b)
    only_b = sorted(ids_b - ids_a)
    differences = _build_differences(rows_a, rows_b, compare_mode=COMPARE_MODE)

    payload = {
        "experiment_dir_a": str(EXPERIMENT_DIR_A),
        "experiment_dir_b": str(EXPERIMENT_DIR_B),
        "predictions_file_a": str(pred_path_a),
        "predictions_file_b": str(pred_path_b),
        "compare_mode": COMPARE_MODE,
        "sample_count_a": len(rows_a),
        "sample_count_b": len(rows_b),
        "common_sample_count": len(ids_a & ids_b),
        "different_sample_count": len(differences),
        "sample_ids_only_in_a": only_a,
        "sample_ids_only_in_b": only_b,
        "different_samples": differences,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n",
        encoding="utf-8",
    )
    print(f"[done] wrote diff report to {OUTPUT_PATH}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
