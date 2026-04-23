"""Difficulty slicing helpers for evaluation rows."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ntust_thesis.core.schemas import EvaluationRow

DIFFICULTY_LEVELS = ("easy", "medium", "hard")


def schema_difficulty(row: EvaluationRow) -> str:
    """Return schema difficulty bucket from gold argument count."""
    n_roles = len(row.gold.arguments)
    if n_roles <= 2:  # noqa: PLR2004
        return "easy"
    if n_roles == 3:  # noqa: PLR2004
        return "medium"
    return "hard"


def group_rows_by_schema_difficulty(
    rows: list[EvaluationRow],
) -> dict[str, list[EvaluationRow]]:
    """Group rows by schema difficulty level."""
    grouped: dict[str, list[EvaluationRow]] = {level: [] for level in DIFFICULTY_LEVELS}
    for row in rows:
        grouped[schema_difficulty(row)].append(row)
    return grouped
