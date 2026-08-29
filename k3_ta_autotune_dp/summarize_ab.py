#!/usr/bin/env python3
"""Compare two K3 TA reproducer result directories without hiding missing ranks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path, help="TA 3.2.1 phase directory")
    parser.add_argument("right", type=Path, help="TA 3.2.2 phase directory")
    parser.add_argument("--expected-world-size", type=int, default=64)
    return parser.parse_args()


def load_case(root: Path, expected_world_size: int) -> dict[str, Any]:
    rank_files = sorted(root.glob("node*/rank*.json"))
    if not rank_files:
        rank_files = sorted(root.glob("rank*.json"))
    ranks: dict[int, dict[str, Any]] = {}
    for path in rank_files:
        value = json.loads(path.read_text())
        ranks[int(value["rank"])] = value

    exit_codes: dict[str, int] = {}
    for path in sorted(root.glob("node*/exit_code.txt")):
        exit_codes[path.parent.name] = int(path.read_text().strip())
    if (root / "exit_code.txt").exists():
        exit_codes[root.name] = int((root / "exit_code.txt").read_text().strip())

    missing = sorted(set(range(expected_world_size)) - set(ranks))
    first_times = [float(value["first_call_total_seconds"]) for value in ranks.values()]
    collective_times = [
        float(value["dp_all_gather"]["seconds"]) for value in ranks.values()
    ]
    source_hash_sets = {
        tuple(sorted(value["environment"]["source_sha256"].items()))
        for value in ranks.values()
    }
    return {
        "root": str(root.resolve()),
        "case": next(iter(ranks.values()))["case"] if ranks else root.name,
        "rank_count": len(ranks),
        "missing_ranks": missing,
        "exit_codes": exit_codes,
        "all_correct": bool(ranks)
        and all(bool(value["correct"]) for value in ranks.values()),
        "min_first_call_seconds": min(first_times) if first_times else None,
        "max_first_call_seconds": max(first_times) if first_times else None,
        "first_call_skew_seconds": (
            max(first_times) - min(first_times) if first_times else None
        ),
        "max_dp_all_gather_seconds": max(collective_times)
        if collective_times
        else None,
        "source_hash_sets": [dict(items) for items in source_hash_sets],
    }


def main() -> None:
    args = parse_args()
    left = load_case(args.left, args.expected_world_size)
    right = load_case(args.right, args.expected_world_size)
    sources_aligned = (
        len(left["source_hash_sets"]) == 1
        and len(right["source_hash_sets"]) == 1
        and left["source_hash_sets"] == right["source_hash_sets"]
    )
    result = {"left": left, "right": right, "sources_aligned": sources_aligned}
    print(json.dumps(result, indent=2, sort_keys=True))

    if left["missing_ranks"] or right["missing_ranks"]:
        raise SystemExit(3)
    if not sources_aligned:
        raise SystemExit(2)
    if not left["all_correct"] or not right["all_correct"]:
        raise SystemExit(4)


if __name__ == "__main__":
    main()
