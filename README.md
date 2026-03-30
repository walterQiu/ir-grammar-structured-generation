install uv: curl -Ls https://astral.sh/uv/install.sh | sh

install ruff、precommit: uv add --dev ruff pre-commit

install pre-commit hook: uv run pre-commit install

uv sync

add datasets/

cp dotenv/.env.example dotenv.env & fill up environment variables in .env

export HF_TOKEN=<HF_TOKEN>  # optional

UV_CACHE_DIR=/tmp/.uv-cache PYTHONPATH=src uv run python run.py --config experimental_configs/development_used.yaml



# tools usage
- ./tools/run_selected_experiments.sh

- PYTHONPATH=src .venv/bin/python tools/sbert_similarity_check.py