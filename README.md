install uv: curl -Ls https://astral.sh/uv/install.sh | sh

install ruff、precommit: uv add --dev ruff pre-commit

uv sync

install pre-commit hook: uv run pre-commit install

test the whole flow using mock data
 - UV_CACHE_DIR=/tmp/.uv-cache PYTHONPATH=src uv run run.py --config configs/experiments/eae_baseline.yaml
 - UV_CACHE_DIR=/tmp/.uv-cache PYTHONPATH=src uv run run.py --config configs/experiments/eae_ir.yaml
 - UV_CACHE_DIR=/tmp/.uv-cache PYTHONPATH=src uv run python run.py --config configs/experiments/eae_ir_gemini.yaml