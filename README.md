install uv: curl -Ls https://astral.sh/uv/install.sh | sh

install ruff、precommit: uv add --dev ruff pre-commit

install pre-commit hook: uv run pre-commit install

uv sync

add datasets/

cp dotenv/.env.example dotenv.env & fill up environment variables in .env

UV_CACHE_DIR=/tmp/.uv-cache PYTHONPATH=src uv run python run.py --config configs/experiments/development_used.yaml

./tools/run_selected_experiments.sh