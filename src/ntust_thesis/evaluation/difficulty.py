"""Difficulty slicing helpers for evaluation rows."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ntust_thesis.core.schemas import EvaluationRow


def _role_multiplicity_total(row: EvaluationRow) -> int:
    """Return sum of allowed role multiplicities for the row event schema."""
    return sum(row.sample_metadata.role_multiplicities.values())


def _gold_role_count(row: EvaluationRow) -> int:
    """Return number of gold arguments in the sample."""
    return len(row.gold.arguments)


def group_rows_by_role_multiplicity_and_gold_role(
    rows: list[EvaluationRow],
) -> dict[int, dict[int, list[EvaluationRow]]]:
    """Group rows by role_multiplicity_total -> gold_role_count."""
    grouped: dict[int, dict[int, list[EvaluationRow]]] = {}
    for row in rows:
        multiplicity_total = _role_multiplicity_total(row)
        role_count = _gold_role_count(row)
        grouped.setdefault(multiplicity_total, {})
        grouped[multiplicity_total].setdefault(role_count, [])
        grouped[multiplicity_total][role_count].append(row)
    return grouped
