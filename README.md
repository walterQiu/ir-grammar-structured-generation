install uv: curl -Ls https://astral.sh/uv/install.sh | sh

install ruff、precommit: uv add --dev ruff pre-commit

uv sync

install pre-commit hook: uv run pre-commit install
