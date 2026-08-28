#!/usr/bin/env python3
"""Summarize block-level results from run_container_matrix.sh."""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Run:
    round_number: int
    label: str
    update_p50_us: float
    overlap_p50_us: float
    eager_submit_p50_us: float
    eager_sync_p50_us: float
    block_update_p50_us: list[float]
    stack: dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument(
        "--slow-block-us",
        type=float,
        default=6000.0,
        help="count update blocks above this p50 threshold (default: 6000)",
    )
    return parser.parse_args()


def load_runs(result_dir: Path) -> list[Run]:
    runs: list[Run] = []
    for path in sorted(result_dir.glob("round_*_*.json")):
        parts = path.stem.split("_")
        if len(parts) != 3 or parts[0] != "round":
            continue
        round_number = int(parts[1])
        label = parts[2]
        if label not in {"old", "new"}:
            continue
        data = json.loads(path.read_text())
        blocks = data.get("blocks")
        if not blocks:
            raise RuntimeError(
                f"{path} has no block data; rerun with --measurement-blocks"
            )
        runs.append(
            Run(
                round_number=round_number,
                label=label,
                update_p50_us=float(data["update"]["p50_us"]),
                overlap_p50_us=float(data["overlap_total"]["p50_us"]),
                eager_submit_p50_us=float(data["eager_submit"]["p50_us"]),
                eager_sync_p50_us=float(data["eager_sync_total"]["p50_us"]),
                block_update_p50_us=[
                    float(block["update"]["p50_us"]) for block in blocks
                ],
                stack={key: str(value) for key, value in data["stack"].items()},
            )
        )
    if not runs:
        raise RuntimeError(f"no round_NN_old/new.json files found in {result_dir}")
    labels = {run.label for run in runs}
    if labels != {"old", "new"}:
        raise RuntimeError(f"both old and new results are required; found {sorted(labels)}")
    return runs


def fmt_stack(stack: dict[str, str]) -> str:
    return (
        f"Python {stack.get('python', 'unknown')}, "
        f"torch {stack.get('torch', 'unknown')}, "
        f"torch-npu {stack.get('torch_npu', 'unknown')}"
    )


def main() -> int:
    args = parse_args()
    runs = load_runs(args.result_dir)

    first_by_label = {
        label: next(run for run in runs if run.label == label)
        for label in ("old", "new")
    }
    print("# IFA NPU Graph container comparison")
    print()
    print(f"- old: {fmt_stack(first_by_label['old'].stack)}")
    print(f"- new: {fmt_stack(first_by_label['new'].stack)}")
    print(f"- slow-block threshold: {args.slow_block_us:.3f} us")
    print()
    print("## Per-process results")
    print()
    print(
        "| Round | Stack | Update p50 (us) | Overlap p50 (us) | "
        "Eager submit p50 (us) | Eager sync p50 (us) | Slow blocks |"
    )
    print("| ---: | --- | ---: | ---: | ---: | ---: | ---: |")
    for run in sorted(
        runs,
        key=lambda item: (item.round_number, 0 if item.label == "old" else 1),
    ):
        slow_blocks = sum(
            value > args.slow_block_us for value in run.block_update_p50_us
        )
        print(
            f"| {run.round_number} | {run.label} | {run.update_p50_us:.3f} | "
            f"{run.overlap_p50_us:.3f} | {run.eager_submit_p50_us:.3f} | "
            f"{run.eager_sync_p50_us:.3f} | "
            f"{slow_blocks}/{len(run.block_update_p50_us)} |"
        )

    print()
    print("## Combined block distribution")
    print()
    print(
        "| Stack | Blocks | Median (us) | Mean (us) | Stddev (us) | "
        "CV | Min (us) | Max (us) | Slow blocks |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

    combined: dict[str, list[float]] = {}
    for label in ("old", "new"):
        values = [
            value
            for run in runs
            if run.label == label
            for value in run.block_update_p50_us
        ]
        combined[label] = values
        mean = statistics.fmean(values)
        stddev = statistics.pstdev(values)
        slow_blocks = sum(value > args.slow_block_us for value in values)
        print(
            f"| {label} | {len(values)} | {statistics.median(values):.3f} | "
            f"{mean:.3f} | {stddev:.3f} | {stddev / mean * 100:.2f}% | "
            f"{min(values):.3f} | {max(values):.3f} | "
            f"{slow_blocks}/{len(values)} |"
        )

    old_mean = statistics.fmean(combined["old"])
    new_mean = statistics.fmean(combined["new"])
    old_median = statistics.median(combined["old"])
    new_median = statistics.median(combined["new"])
    print()
    print(
        f"New-vs-old block mean: {new_mean - old_mean:+.3f} us "
        f"({(new_mean / old_mean - 1) * 100:+.2f}%)."
    )
    print(
        f"New-vs-old block median: {new_median - old_median:+.3f} us "
        f"({(new_median / old_median - 1) * 100:+.2f}%)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
