#!/usr/bin/env python3
"""Summarize one or more collected alloc_extend 64-rank result trees."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "result_roots",
        nargs="+",
        help="Case/phase roots containing node*/rank*.{result,state}.json",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rank_from_path(path: Path) -> int:
    return int(path.name.split(".", 1)[0].removeprefix("rank"))


def summarize(root: Path) -> dict[str, Any]:
    state_paths = list(root.rglob("rank*.state.json"))
    result_paths = list(root.rglob("rank*.result.json"))
    states = {rank_from_path(path): load_json(path) for path in state_paths}
    results = {rank_from_path(path): load_json(path) for path in result_paths}
    expected_world_sizes = {
        int(state["world_size"])
        for state in states.values()
        if "world_size" in state
    }
    expected_world_size = (
        next(iter(expected_world_sizes)) if len(expected_world_sizes) == 1 else None
    )
    expected_ranks = (
        set(range(expected_world_size)) if expected_world_size is not None else set()
    )
    compile_seconds = [
        float(result["compile_seconds"]) for result in results.values()
    ]
    source_hashes = {
        result.get("allocator_sha256") for result in results.values()
    } - {None}
    variants = {result.get("variant") for result in results.values()} - {None}
    shapes = {
        json.dumps(result.get("shape"), sort_keys=True)
        for result in results.values()
    }
    versions = {
        json.dumps(result.get("versions"), sort_keys=True)
        for result in results.values()
    }
    return {
        "root": str(root.resolve()),
        "expected_world_size": expected_world_size,
        "seen_state_ranks": sorted(states),
        "completed_result_ranks": sorted(results),
        "missing_state_ranks": sorted(expected_ranks - states.keys()),
        "incomplete_ranks": sorted(expected_ranks - results.keys()),
        "last_event_counts": dict(
            sorted(Counter(state.get("event", "unknown") for state in states.values()).items())
        ),
        "all_correct": bool(results)
        and all(bool(result.get("correct")) for result in results.values()),
        "allocator_sha256": sorted(source_hashes),
        "variants": sorted(variants),
        "shapes": sorted(shapes),
        "versions": sorted(versions),
        "compile_seconds": {
            "count": len(compile_seconds),
            "min": min(compile_seconds) if compile_seconds else None,
            "median": statistics.median(compile_seconds) if compile_seconds else None,
            "max": max(compile_seconds) if compile_seconds else None,
        },
    }


def main() -> int:
    args = parse_args()
    summaries = [summarize(Path(value)) for value in args.result_roots]
    print(json.dumps(summaries, indent=2, sort_keys=True))
    complete = all(
        summary["expected_world_size"] is not None
        and not summary["incomplete_ranks"]
        and summary["all_correct"]
        for summary in summaries
    )
    comparable = len(
        {
            tuple(summary["allocator_sha256"])
            for summary in summaries
        }
    ) == 1 and len({tuple(summary["shapes"]) for summary in summaries}) == 1
    if not comparable:
        return 2
    return 0 if complete else 3


if __name__ == "__main__":
    raise SystemExit(main())
