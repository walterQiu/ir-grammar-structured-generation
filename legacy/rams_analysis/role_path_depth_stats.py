"""Analyze RAMS role-path depth distribution by split.

Definition used in this script:
- role path depth: number of segments in normalized role path, split by '.'
  e.g., 'role1.role2.role3' -> depth 3
- sample depth: maximum role depth within that sample
  (so each sample contributes to exactly one depth bucket)
"""

from __future__ import annotations

import json
from collections import Counter

from legacy.rams_analysis.common import DATA_DIR, SPLITS, normalize_role_name


def role_depth(role_path: str) -> int:
    """Return depth of role path using dot segments."""
    parts = [part.strip() for part in role_path.split(".") if part.strip()]
    return len(parts)


def sample_max_role_depth(row: dict) -> int:
    """Return max role depth in one RAMS sample."""
    max_depth = 0
    for link in row.get("gold_evt_links", []):
        if not isinstance(link, list) or len(link) < 3:
            continue
        raw_role = link[2]
        if not isinstance(raw_role, str):
            continue
        normalized = normalize_role_name(raw_role)
        if not normalized:
            continue
        max_depth = max(max_depth, role_depth(normalized))
    return max_depth


def analyze_split(split: str) -> Counter[int]:
    """Compute depth -> sample count for one split."""
    path = DATA_DIR / f"{split}.jsonlines"
    counter: Counter[int] = Counter()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            depth = sample_max_role_depth(row)
            counter[depth] += 1
    return counter


def main() -> None:
    """Run depth analysis for train/dev/test and print table."""
    split_counters = {split: analyze_split(split) for split in SPLITS}
    all_depths = sorted({d for c in split_counters.values() for d in c})

    header = ["depth", *SPLITS]
    print("\t".join(header))
    for depth in all_depths:
        row = [str(depth)]
        for split in SPLITS:
            row.append(str(split_counters[split].get(depth, 0)))
        print("\t".join(row))


if __name__ == "__main__":
    main()
