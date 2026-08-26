#!/usr/bin/env python3
"""Reproduce the CANN 9.1 IFA NPU Graph update latency regression.

The Qwen3-30B-A3B EAGLE3 target-verify graph contains 48 independent
FusedInferAttentionScore records.  This reproducer captures the same number
of IFA records, changes ``actual_seq_lengths_kv`` through ``NPUGraph.update``,
and overlaps the update with ``NPUGraph.replay`` in the same shape as the
production graph runner.

No SGLang checkout or model weights are required.  Run the production-shaped
case with:

    NPU_DEVICE_INDEX=0 python3 ifa_npugraph_update_repro.py

Use ``--max-update-p50-us`` to turn the measurement into a regression gate.
The default is report-only because absolute host latency varies by machine.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import torch
import torch_npu


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * quantile))
    return ordered[index]


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "mean_us": round(statistics.fmean(values), 3),
        "p50_us": round(statistics.median(values), 3),
        "p95_us": round(_percentile(values, 0.95), 3),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        type=int,
        default=int(os.environ.get("NPU_DEVICE_INDEX", "0")),
        help="physical NPU device index (default: NPU_DEVICE_INDEX or 0)",
    )
    parser.add_argument(
        "--records",
        type=int,
        default=48,
        help="number of captured IFA records (production target verify: 48)",
    )
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--seq-len", type=int, default=3500)
    parser.add_argument(
        "--unique-tensors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="give each captured record unique tensor addresses (default: true)",
    )
    parser.add_argument("--eager-iters", type=int, default=100)
    parser.add_argument(
        "--eager-only",
        action="store_true",
        help="measure the eager control without capturing an NPU graph",
    )
    parser.add_argument(
        "--profile-dir",
        type=Path,
        help="optionally collect a torch-npu CPU+NPU profile of the timed loop",
    )
    parser.add_argument(
        "--max-update-p50-us",
        type=float,
        help="exit nonzero when graph.update p50 exceeds this machine-specific limit",
    )
    parser.add_argument(
        "--max-overlap-p50-us",
        type=float,
        help="exit nonzero when update/replay overlap p50 exceeds this limit",
    )
    args = parser.parse_args()
    if args.records < 1:
        parser.error("--records must be at least 1")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.iters < 1:
        parser.error("--iters must be at least 1")
    if args.eager_iters < 1:
        parser.error("--eager-iters must be at least 1")
    if args.seq_len < 1:
        parser.error("--seq-len must be at least 1")
    return args


def _make_profiler(profile_dir: Path):
    profile_dir.mkdir(parents=True, exist_ok=True)
    return torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        record_shapes=False,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(profile_dir)),
        experimental_config=torch_npu.profiler._ExperimentalConfig(
            export_type=torch_npu.profiler.ExportType.Text,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            msprof_tx=False,
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            l2_cache=False,
            op_attr=False,
            data_simplification=False,
            record_op_args=False,
        ),
    )


def main() -> int:
    args = _parse_args()
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        print("ERROR: this reproducer requires an Ascend NPU", file=sys.stderr)
        return 2

    torch.npu.set_device(args.device)
    torch.manual_seed(42)
    device = f"npu:{args.device}"
    dtype = torch.bfloat16

    # Qwen3-30B-A3B target-verify attention shape.
    num_tokens = 5
    num_heads = 16
    num_kv_heads = 2
    head_dim = 128
    page_size = 128
    num_blocks = math.ceil((args.seq_len + num_tokens) / page_size)

    query = torch.randn(
        (num_tokens, 1, num_heads * head_dim), device=device, dtype=dtype
    )
    key = torch.randn(
        (num_blocks, page_size, num_kv_heads * head_dim),
        device=device,
        dtype=dtype,
    )
    value = torch.randn_like(key)
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32)[
        None, :
    ].repeat(num_tokens, 1)
    common = {
        "block_size": page_size,
        "num_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "input_layout": "BSH",
        "scale": head_dim**-0.5,
        "actual_seq_lengths_kv": [args.seq_len] * num_tokens,
        "sparse_mode": 0,
    }

    # The calls are serial, so sharing the workspace is safe.  Unique tensor
    # addresses still force 48 independent graph records, as in the model.
    workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
        query, key, value, **common, block_table=block_table
    )
    output = torch.empty_like(query)
    lse = torch.empty(1, device=device, dtype=dtype)
    workspaces = [workspace] * args.records
    if args.unique_tensors:
        queries = [query] + [torch.empty_like(query) for _ in range(args.records - 1)]
        keys = [key] + [torch.empty_like(key) for _ in range(args.records - 1)]
        values = [value] + [torch.empty_like(value) for _ in range(args.records - 1)]
        block_tables = [block_table] + [
            block_table.clone() for _ in range(args.records - 1)
        ]
        outputs = [[output, lse]] + [
            [torch.empty_like(output), torch.empty_like(lse)]
            for _ in range(args.records - 1)
        ]
    else:
        queries = [query] * args.records
        keys = [key] * args.records
        values = [value] * args.records
        block_tables = [block_table] * args.records
        outputs = [[output, lse]] * args.records

    def issue_ops() -> None:
        for index in range(args.records):
            torch_npu.npu_fused_infer_attention_score.out(
                queries[index],
                keys[index],
                values[index],
                **common,
                block_table=block_tables[index],
                workspace=workspaces[index],
                out=outputs[index],
            )

    def issue_one() -> None:
        torch_npu.npu_fused_infer_attention_score.out(
            queries[0],
            keys[0],
            values[0],
            **common,
            block_table=block_tables[0],
            workspace=workspaces[0],
            out=outputs[0],
        )

    for _ in range(3):
        issue_ops()
    torch.npu.synchronize()

    eager_submit_us: list[float] = []
    eager_sync_total_us: list[float] = []
    for _ in range(args.eager_iters):
        started = time.perf_counter_ns()
        issue_one()
        submitted = time.perf_counter_ns()
        torch.npu.synchronize()
        synchronized = time.perf_counter_ns()
        eager_submit_us.append((submitted - started) / 1_000)
        eager_sync_total_us.append((synchronized - started) / 1_000)

    result = {
        "stack": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_npu": getattr(torch_npu, "__version__", "unknown"),
        },
        "device": args.device,
        "records": args.records,
        "seq_len": args.seq_len,
        "unique_tensors": args.unique_tensors,
        "workspace_bytes": workspace.numel() * workspace.element_size(),
        "eager_submit": _stats(eager_submit_us),
        "eager_sync_total": _stats(eager_sync_total_us),
    }
    if args.eager_only:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    graph = torch.npu.NPUGraph()
    capture_stream = torch.npu.Stream(device=args.device)
    with torch.npu.graph(
        graph,
        stream=capture_stream,
        auto_dispatch_capture=True,
    ):
        issue_ops()
    torch.npu.synchronize()

    updates = [
        {"actual_seq_lengths_kv": [args.seq_len + 1] * num_tokens}
        for _ in range(args.records)
    ]
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    update_us: list[float] = []
    replay_us: list[float] = []
    overlap_total_us: list[float] = []

    def timed_update() -> float:
        started = time.perf_counter_ns()
        graph.update(updates)
        return (time.perf_counter_ns() - started) / 1_000

    profiler = _make_profiler(args.profile_dir) if args.profile_dir else None
    if profiler is not None:
        profiler.start()

    try:
        for iteration in range(args.warmup + args.iters):
            started = time.perf_counter_ns()
            future = executor.submit(timed_update)
            replay_started = time.perf_counter_ns()
            graph.replay()
            replay_finished = time.perf_counter_ns()
            current_update_us = future.result()
            finished = time.perf_counter_ns()
            if iteration >= args.warmup:
                update_us.append(current_update_us)
                replay_us.append((replay_finished - replay_started) / 1_000)
                overlap_total_us.append((finished - started) / 1_000)
    finally:
        if profiler is not None:
            profiler.stop()
        torch.npu.synchronize()
        executor.shutdown()

    result.update(
        {
            "update": _stats(update_us),
            "replay_submit": _stats(replay_us),
            "overlap_total": _stats(overlap_total_us),
        }
    )
    print(json.dumps(result, indent=2, sort_keys=True))

    violations = []
    if (
        args.max_update_p50_us is not None
        and result["update"]["p50_us"] > args.max_update_p50_us
    ):
        violations.append(
            f"update p50 {result['update']['p50_us']:.3f} us > "
            f"{args.max_update_p50_us:.3f} us"
        )
    if (
        args.max_overlap_p50_us is not None
        and result["overlap_total"]["p50_us"] > args.max_overlap_p50_us
    ):
        violations.append(
            f"overlap total p50 {result['overlap_total']['p50_us']:.3f} us > "
            f"{args.max_overlap_p50_us:.3f} us"
        )
    if violations:
        print("REGRESSION: " + "; ".join(violations), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
