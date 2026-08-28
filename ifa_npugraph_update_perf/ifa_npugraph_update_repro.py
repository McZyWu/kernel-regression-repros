#!/usr/bin/env python3
"""Measure IFA NPU Graph update latency and scheduling stability.

The Qwen3-30B-A3B EAGLE3 target-verify graph contains 48 independent
FusedInferAttentionScore records.  At the production graph bucket of 162
requests, each target-verify forward contains four query tokens per request
and updates a 162-element ``actual_seq_lengths_kv`` list.  This reproducer
captures the same number and shapes of IFA records and changes that list
through ``NPUGraph.update``.  The default isolates the host update path:
standalone replay of this extracted TND subgraph can leave a pending device
task because it omits the surrounding whole-model graph dependencies.

No SGLang checkout or model weights are required.  Run the production-shaped
case with:

    NPU_DEVICE_INDEX=0 python3 ifa_npugraph_update_repro.py

Use multiple measurement blocks and repeated processes when comparing stacks.
The default is report-only because absolute latency and slow-tail frequency
depend on host scheduling.  ``--max-update-p50-us`` is available only after a
machine-specific, CPU-affinity-controlled baseline has been established.
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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=162,
        help="requests in the target-verify graph bucket (production: 162)",
    )
    parser.add_argument(
        "--tokens-per-request",
        type=int,
        default=4,
        help="target-verify query tokens per request (production: 4)",
    )
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument(
        "--measurement-blocks",
        type=int,
        default=1,
        help=(
            "number of independently reported timing blocks after one graph "
            "capture (default: 1)"
        ),
    )
    parser.add_argument("--seq-len", type=int, default=3500)
    parser.add_argument(
        "--unique-tensors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="give each captured record unique tensor addresses (default: true)",
    )
    parser.add_argument(
        "--update-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "measure only the production-shaped host graph.update path "
            "(default: true); --no-update-only also submits standalone graph "
            "replay and is experimental for the extracted TND subgraph"
        ),
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
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.tokens_per_request < 1:
        parser.error("--tokens-per-request must be at least 1")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.iters < 1:
        parser.error("--iters must be at least 1")
    if args.measurement_blocks < 1:
        parser.error("--measurement-blocks must be at least 1")
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
    num_tokens = args.batch_size * args.tokens_per_request
    num_heads = 16
    num_kv_heads = 2
    head_dim = 128
    page_size = 128
    blocks_per_request = math.ceil(
        (args.seq_len + args.tokens_per_request) / page_size
    )
    num_blocks = args.batch_size * blocks_per_request

    query = torch.randn(
        (num_tokens, num_heads, head_dim), device=device, dtype=dtype
    )
    key = torch.randn(
        (num_blocks, page_size, num_kv_heads * head_dim),
        device=device,
        dtype=dtype,
    )
    value = torch.randn_like(key)
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32).view(
        args.batch_size, blocks_per_request
    )
    actual_seq_lengths = list(
        range(args.tokens_per_request, num_tokens + 1, args.tokens_per_request)
    )
    # AscendAttnMaskBuilder.generate_mask_flag(2048), as used by forward_mtp.
    atten_mask = ~torch.ones(
        (2048, 2048), device=device, dtype=torch.bool
    ).tril_()
    common = {
        "block_size": page_size,
        "num_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "input_layout": "TND",
        "scale": head_dim**-0.5,
        "atten_mask": atten_mask,
        "actual_seq_lengths": actual_seq_lengths,
        "actual_seq_lengths_kv": [args.seq_len] * args.batch_size,
        "sparse_mode": 3,
    }

    # Qwen3-30B-A3B is on the non-MLA forward_mtp branch.  That production
    # path uses the functional torch.ops API; the explicit-workspace `.out`
    # API belongs to the MLA branch and does not replay equivalently here.
    if args.unique_tensors:
        queries = [query] + [torch.empty_like(query) for _ in range(args.records - 1)]
        keys = [key] + [torch.empty_like(key) for _ in range(args.records - 1)]
        values = [value] + [torch.empty_like(value) for _ in range(args.records - 1)]
        block_tables = [block_table] + [
            block_table.clone() for _ in range(args.records - 1)
        ]
    else:
        queries = [query] * args.records
        keys = [key] * args.records
        values = [value] * args.records
        block_tables = [block_table] * args.records

    def issue_ops() -> list[object]:
        outputs = []
        for index in range(args.records):
            outputs.append(
                torch.ops.npu.npu_fused_infer_attention_score(
                    queries[index],
                    keys[index],
                    values[index],
                    **common,
                    block_table=block_tables[index],
                )
            )
        return outputs

    def issue_one() -> object:
        return torch.ops.npu.npu_fused_infer_attention_score(
            queries[0],
            keys[0],
            values[0],
            **common,
            block_table=block_tables[0],
        )

    for _ in range(3):
        warmup_outputs = issue_ops()
    torch.npu.synchronize()

    eager_submit_us: list[float] = []
    eager_sync_total_us: list[float] = []
    for _ in range(args.eager_iters):
        started = time.perf_counter_ns()
        eager_output = issue_one()
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
        "batch_size": args.batch_size,
        "tokens_per_request": args.tokens_per_request,
        "query_tokens": num_tokens,
        "seq_len": args.seq_len,
        "blocks_per_request": blocks_per_request,
        "kv_blocks": num_blocks,
        "unique_tensors": args.unique_tensors,
        "operator_api": "torch.ops.npu.npu_fused_infer_attention_score",
        "mode": "update_only" if args.update_only else "update_replay_overlap",
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
        # Retain every functional output so capture does not recycle output
        # addresses between the 48 layer-like records.
        captured_outputs = issue_ops()
    torch.npu.synchronize()

    # Match the target graph runner exactly: submit one CPU update input and
    # let torch-npu broadcast it to all captured IFA dispatch records.  Passing
    # one dictionary per record produces the same values, but bypasses the
    # singleton expansion path whose host cost is part of this regression.
    updates = [
        {"actual_seq_lengths_kv": [args.seq_len + 1] * args.batch_size}
    ]
    # NPU device selection is thread-local.  Without an initializer, a worker
    # created while the main thread is on (for example) npu:14 silently opens
    # npu:0 and graph.update/final synchronization can cross devices or hang.
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1,
        initializer=torch.npu.set_device,
        initargs=(args.device,),
    )
    update_us: list[float] = []
    replay_us: list[float] = []
    overlap_total_us: list[float] = []
    block_results: list[dict[str, object]] = []

    def timed_update() -> float:
        started = time.perf_counter_ns()
        graph.update(cpu_update_input=updates)
        return (time.perf_counter_ns() - started) / 1_000

    profiler = _make_profiler(args.profile_dir) if args.profile_dir else None
    if profiler is not None:
        profiler.start()

    try:
        for _ in range(args.warmup):
            future = executor.submit(timed_update)
            if not args.update_only:
                graph.replay()
            future.result()

        for block_index in range(args.measurement_blocks):
            block_update_us: list[float] = []
            block_replay_us: list[float] = []
            block_overlap_total_us: list[float] = []
            for _ in range(args.iters):
                started = time.perf_counter_ns()
                future = executor.submit(timed_update)
                replay_started = time.perf_counter_ns()
                if not args.update_only:
                    graph.replay()
                replay_finished = time.perf_counter_ns()
                current_update_us = future.result()
                finished = time.perf_counter_ns()
                block_update_us.append(current_update_us)
                block_replay_us.append(
                    0.0
                    if args.update_only
                    else (replay_finished - replay_started) / 1_000
                )
                block_overlap_total_us.append((finished - started) / 1_000)

            update_us.extend(block_update_us)
            replay_us.extend(block_replay_us)
            overlap_total_us.extend(block_overlap_total_us)
            block_update_stats = _stats(block_update_us)
            block_result = {
                "block": block_index + 1,
                "update": block_update_stats,
                "replay_submit": _stats(block_replay_us),
                "overlap_total": _stats(block_overlap_total_us),
            }
            block_results.append(block_result)
            print(
                f"measurement block {block_index + 1}/{args.measurement_blocks}: "
                f"update p50={block_update_stats['p50_us']:.3f} us",
                file=sys.stderr,
                flush=True,
            )
    finally:
        if profiler is not None:
            profiler.stop()
        torch.npu.synchronize()
        executor.shutdown()

    result.update(
        {
            "measurement_blocks": args.measurement_blocks,
            "iterations_per_block": args.iters,
            "update": _stats(update_us),
            "replay_submit": _stats(replay_us),
            "overlap_total": _stats(overlap_total_us),
            "blocks": block_results,
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
