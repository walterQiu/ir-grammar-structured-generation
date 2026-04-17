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

- PYTHONPATH=src python tools/generate_extraction_notes_cache.py --overwrite


# ICL examples
RAMS train split

每個 role 最多一個 span
nw_RC10bdc622f21c5c3aeb343898b9ed2d31fefd2e8e5db923badcfdc9fb

部分 role 有多個 span，且 gold 也有多個 span
nw_RC8bfda5484f6491dad438110794bc990f02525d6d9a4751a203f32acb

部分 role 有多個 span，但 gold 全部 role 都是一個 span
nw_RC4d768901bfdeb4296e42dda04d3d8ddc0872462814c789188ae8e591

沒有合法 role
nw_RC2adbdc1cf8523629763aacf509f3ba13ed44eda47ccf1404c710691e
