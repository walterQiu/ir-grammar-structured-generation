#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIST_YAML="${ROOT_DIR}/experimental_configs/batch_run_list.yaml"

if [[ ! -f "${LIST_YAML}" ]]; then
  echo "[error] list file not found: ${LIST_YAML}" >&2
  exit 1
fi

mapfile -t EXPERIMENT_FILES < <(
  PYTHONPATH="${ROOT_DIR}/src" .venv/bin/python - <<'PY' "${LIST_YAML}"
import sys
from pathlib import Path

import yaml

list_path = Path(sys.argv[1])
data = yaml.safe_load(list_path.read_text(encoding="utf-8")) or {}
experiments = data.get("experiments", [])
if not isinstance(experiments, list):
    raise SystemExit("`experiments` must be a list in batch_run_list.yaml")

for item in experiments:
    if isinstance(item, str) and item.strip():
        print(item.strip())
PY
)

if [[ ${#EXPERIMENT_FILES[@]} -eq 0 ]]; then
  echo "[error] no experiments found in ${LIST_YAML}" >&2
  exit 1
fi

echo "[info] loaded ${#EXPERIMENT_FILES[@]} experiments from ${LIST_YAML}"

for idx in "${!EXPERIMENT_FILES[@]}"; do
  cfg_rel="${EXPERIMENT_FILES[$idx]}"
  cfg_path="${ROOT_DIR}/${cfg_rel}"
  run_no=$((idx + 1))

  if [[ ! -f "${cfg_path}" ]]; then
    echo "[error] config not found (${run_no}/${#EXPERIMENT_FILES[@]}): ${cfg_rel}" >&2
    exit 1
  fi

  echo "[run ${run_no}/${#EXPERIMENT_FILES[@]}] ${cfg_rel}"
  (
    cd "${ROOT_DIR}"
    UV_CACHE_DIR=/tmp/.uv-cache PYTHONPATH=src uv run python run.py --config "${cfg_rel}"
  )
done

echo "[done] all selected experiments finished."
