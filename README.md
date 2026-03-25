install uv: curl -Ls https://astral.sh/uv/install.sh | sh

install ruff、precommit: uv add --dev ruff pre-commit

uv sync

install pre-commit hook: uv run pre-commit install

 
UV_CACHE_DIR=/tmp/.uv-cache PYTHONPATH=src uv run python run.py --config configs/experiments/rams_baseline_gemini_2_5_flash_lite.yaml