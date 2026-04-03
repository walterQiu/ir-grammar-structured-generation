"""Analyze RAMS number-of-roles distribution by split.

Definition used in this script:
- number_of_roles (per sample): number of distinct normalized roles
  appearing in that sample's event links.
"""

from __future__ import annotations

import json
from collections import Counter

from common import DATA_DIR, SPLITS, normalize_role_name


def sample_number_of_roles(row: dict) -> int:
    """Return number of distinct roles in one sample."""
    roles: set[str] = set()
    for link in row.get("gold_evt_links", []):
        if not isinstance(link, list) or len(link) < 3:
            continue
        raw_role = link[2]
        if not isinstance(raw_role, str):
            continue
        normalized = normalize_role_name(raw_role)
        if not normalized:
            continue
        roles.add(normalized)
    return len(roles)


def analyze_split(split: str) -> Counter[int]:
    """Compute number_of_roles -> sample count for one split."""
    path = DATA_DIR / f"{split}.jsonlines"
    counter: Counter[int] = Counter()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            n_roles = sample_number_of_roles(row)
            counter[n_roles] += 1
    return counter


def merged_counter(split_counters: dict[str, Counter[int]]) -> Counter[int]:
    """Merge all split counters into one counter."""
    merged: Counter[int] = Counter()
    for counter in split_counters.values():
        merged.update(counter)
    return merged


def quantile_role_count(counter: Counter[int], quantile: float) -> int:
    """Return smallest role-count bucket whose CDF reaches quantile."""
    total = sum(counter.values())
    if total == 0:
        return 0
    threshold = quantile * total
    cumulative = 0
    for n_roles in sorted(counter):
        cumulative += counter[n_roles]
        if cumulative >= threshold:
            return n_roles
    return max(counter)


def main() -> None:
    """Run role-count analysis for train/dev/test and merged view."""
    split_counters = {split: analyze_split(split) for split in SPLITS}
    all_counts = sorted({k for c in split_counters.values() for k in c})

    header = ["number_of_roles", *SPLITS]
    print("\t".join(header))
    for n_roles in all_counts:
        row = [str(n_roles)]
        for split in SPLITS:
            row.append(str(split_counters[split].get(n_roles, 0)))
        print("\t".join(row))

    merged = merged_counter(split_counters)
    print()
    print("merged_number_of_roles\tsample_count")
    for n_roles in sorted(merged):
        print(f"{n_roles}\t{merged[n_roles]}")

    q1 = quantile_role_count(merged, 1 / 3)
    q2 = quantile_role_count(merged, 2 / 3)
    print()
    print("quantile\tnumber_of_roles")
    print(f"1/3\t{q1}")
    print(f"2/3\t{q2}")


if __name__ == "__main__":
    main()
