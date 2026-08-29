#!/usr/bin/env python3
"""64-rank Kimi-K3 reproducer for the Triton ``alloc_extend`` JIT path.

The test deliberately separates Triton compilation, first launch, device
synchronization, and the following DP all-gather.  It supports the exact SGLang
dynamic-loop kernel and a control variant that restores the constexpr loop
bound used before SGLang PR #19898.

Cache variables are configured before importing Torch, Triton, or SGLang.
Use ``run_alloc_extend_network.sh`` for cold-cache refusal and environment
capture; do not invoke this file directly for an A/B result.
"""

from __future__ import annotations

import argparse
import faulthandler
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import sys
import time
from datetime import timedelta
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--phase", choices=("cold", "warm"), required=True)
    parser.add_argument(
        "--variant", choices=("exact-dynamic", "static-bound"), required=True
    )
    parser.add_argument("--cache-root", required=True)
    parser.add_argument(
        "--cache-layout", choices=("per-node", "per-rank"), default="per-node"
    )
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--page-size", type=int, default=128)
    parser.add_argument("--prefix-tokens", type=int, default=0)
    parser.add_argument("--extend-tokens", type=int, default=70)
    parser.add_argument("--hot-repeats", type=int, default=2)
    parser.add_argument("--dist-timeout-seconds", type=int, default=900)
    parser.add_argument("--dump-stack-after-seconds", type=int, default=90)
    return parser.parse_args()


def coordinates() -> tuple[int, int, int, int, int]:
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    local_world_size = int(os.getenv("LOCAL_WORLD_SIZE", "1"))
    node_rank = rank // local_world_size
    return rank, world_size, local_rank, local_world_size, node_rank


ARGS = parse_args()
RANK, WORLD_SIZE, LOCAL_RANK, LOCAL_WORLD_SIZE, NODE_RANK = coordinates()


def configure_cache() -> dict[str, Any]:
    node_root = Path(ARGS.cache_root).resolve() / f"node{NODE_RANK}"
    process_root = (
        node_root
        if ARGS.cache_layout == "per-node"
        else node_root / f"rank{RANK}"
    )
    paths = {
        "TRITON_CACHE_DIR": process_root / "triton",
        "TORCHINDUCTOR_CACHE_DIR": process_root / "inductor",
        "SGLANG_CACHE_DIR": process_root / "sglang",
    }
    preexisting = {name: path.exists() for name, path in paths.items()}
    for name, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    return {
        "layout": ARGS.cache_layout,
        "paths": {name: str(path) for name, path in paths.items()},
        "preexisting_at_process_start": preexisting,
    }


CACHE_INFO = configure_cache()
RESULT_DIR = Path(ARGS.result_dir).resolve()
RESULT_DIR.mkdir(parents=True, exist_ok=True)
STATE_PATH = RESULT_DIR / f"rank{RANK}.state.json"
EVENTS_PATH = RESULT_DIR / f"rank{RANK}.events.jsonl"
RESULT_PATH = RESULT_DIR / f"rank{RANK}.result.json"

if ARGS.dump_stack_after_seconds > 0:
    faulthandler.dump_traceback_later(
        ARGS.dump_stack_after_seconds, repeat=True, file=sys.stderr
    )

# The cache paths above must exist in the environment before these imports.
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch_npu  # noqa: F401,E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

import sglang.kernels.ops.memory.allocator as allocator_module  # noqa: E402


def emit(event: str, **fields: Any) -> None:
    record = {
        "event": event,
        "wall_time": time.time(),
        "monotonic": time.monotonic(),
        "rank": RANK,
        "world_size": WORLD_SIZE,
        "local_rank": LOCAL_RANK,
        "node_rank": NODE_RANK,
        **fields,
    }
    print("ALLOC_EXTEND_EVENT=" + json.dumps(record, sort_keys=True), flush=True)
    with EVENTS_PATH.open("a", encoding="utf-8") as history:
        history.write(json.dumps(record, sort_keys=True) + "\n")
    temporary = STATE_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    temporary.replace(STATE_PATH)


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def distribution_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for name in ("torch", "torch-npu", "triton", "triton-ascend", "sglang"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "NOT_INSTALLED"
    return result


def cache_summary() -> dict[str, int]:
    root = Path(os.environ["TRITON_CACHE_DIR"])
    files: list[Path] = []
    total_bytes = 0
    for path in root.rglob("*"):
        try:
            if path.is_file():
                files.append(path)
                total_bytes += path.stat().st_size
        except FileNotFoundError:
            # Another local rank may atomically replace a shared-cache file
            # while this diagnostic snapshot is walking the directory.
            continue
    return {
        "file_count": len(files),
        "total_bytes": total_bytes,
    }


def module_artifacts() -> dict[str, dict[str, str] | None]:
    result: dict[str, dict[str, str] | None] = {}
    for module_name in (
        "triton.compiler.compiler",
        "triton.runtime.jit",
        "triton.runtime.cache",
        "triton.backends.ascend.compiler",
        "triton.backends.ascend.driver",
    ):
        try:
            module = importlib.import_module(module_name)
            path = Path(module.__file__).resolve()
            result[module_name] = {
                "path": str(path),
                "sha256": sha256_file(path),
            }
        except (AttributeError, ImportError, TypeError):
            result[module_name] = None
    return result


# Control kernel: this is the production algorithm with only the Part 2 loop
# bound restored to a constexpr. It mirrors the implementation immediately
# before SGLang PR #19898 and is not an autotuned kernel.
@triton.jit(do_not_specialize=["free_page_ptr"])
def alloc_extend_kernel_static_bound(
    pre_lens_ptr,
    seq_lens_ptr,
    last_loc_ptr,
    free_page_ptr,
    out_indices,
    bs_upper: tl.constexpr,
    page_size: tl.constexpr,
    max_num_extend_tokens: tl.constexpr,
):
    pid = tl.program_id(0)

    load_offset = tl.arange(0, bs_upper)
    seq_lens = tl.load(seq_lens_ptr + load_offset, mask=load_offset <= pid)
    pre_lens = tl.load(pre_lens_ptr + load_offset, mask=load_offset <= pid)
    extend_lens = seq_lens - pre_lens

    seq_len = tl.load(seq_lens_ptr + pid)
    pre_len = tl.load(pre_lens_ptr + pid)
    extend_len = seq_len - pre_len

    sum_extend_lens = tl.sum(extend_lens)
    output_start_loc = sum_extend_lens - extend_len

    num_pages_after = (seq_lens + page_size - 1) // page_size
    num_pages_before = (pre_lens + page_size - 1) // page_size
    num_new_pages = num_pages_after - num_pages_before

    num_page_start_loc_self = (seq_len + page_size - 1) // page_size - (
        pre_len + page_size - 1
    ) // page_size
    sum_num_new_pages = tl.sum(num_new_pages)
    new_page_start_loc = sum_num_new_pages - num_page_start_loc_self

    last_loc = tl.load(last_loc_ptr + pid)
    num_part1 = (
        min(seq_len, (pre_len + page_size - 1) // page_size * page_size) - pre_len
    )
    offset_one_page = tl.arange(0, page_size)
    tl.store(
        out_indices + output_start_loc + offset_one_page,
        last_loc + 1 + offset_one_page,
        mask=offset_one_page < num_part1,
    )
    if pre_len + num_part1 == seq_len:
        return

    num_part2 = (
        seq_len // page_size * page_size
        - (pre_len + page_size - 1) // page_size * page_size
    )
    BLOCK_EXTEND: tl.constexpr = 4096
    num_blocks: tl.constexpr = (
        max_num_extend_tokens + BLOCK_EXTEND - 1
    ) // BLOCK_EXTEND
    for block_id in range(num_blocks):
        offset_in_block = tl.arange(0, BLOCK_EXTEND)
        offset = block_id * BLOCK_EXTEND + offset_in_block
        mask = offset < num_part2
        page_start = tl.load(
            free_page_ptr + new_page_start_loc + offset // page_size,
            mask=mask,
        )
        tl.store(
            out_indices + output_start_loc + num_part1 + offset,
            page_start * page_size + offset % page_size,
            mask=mask,
        )
    if pre_len + num_part1 + num_part2 == seq_len:
        return

    num_part3 = seq_len - seq_len // page_size * page_size
    start_loc = tl.load(
        free_page_ptr + new_page_start_loc + num_page_start_loc_self - 1
    )
    tl.store(
        out_indices + output_start_loc + num_part1 + num_part2 + offset_one_page,
        start_loc * page_size + offset_one_page,
        mask=offset_one_page < num_part3,
    )


def make_inputs(device: torch.device) -> tuple[torch.Tensor, ...]:
    if ARGS.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if ARGS.page_size < 1 or ARGS.page_size & (ARGS.page_size - 1):
        raise ValueError("page-size must be a positive power of two")
    if ARGS.prefix_tokens < 0 or ARGS.extend_tokens < 1:
        raise ValueError("prefix-tokens must be nonnegative and extend-tokens positive")

    prefix = torch.full(
        (ARGS.batch_size,), ARGS.prefix_tokens, dtype=torch.int64
    )
    sequence = prefix + ARGS.extend_tokens
    last_locations = []
    for index, prefix_length in enumerate(prefix.tolist()):
        if prefix_length == 0:
            last_locations.append(-1)
        else:
            page = 50_000 + index
            last_locations.append(
                page * ARGS.page_size + (prefix_length - 1) % ARGS.page_size
            )
    last_location = torch.tensor(last_locations, dtype=torch.int64)
    pages_needed = int(
        (
            (sequence + ARGS.page_size - 1) // ARGS.page_size
            - (prefix + ARGS.page_size - 1) // ARGS.page_size
        )
        .sum()
        .item()
    )
    free_pages = torch.arange(
        1_000,
        1_000 + max(pages_needed, 1) + 16,
        dtype=torch.int64,
    )
    output = torch.empty(
        (ARGS.batch_size * ARGS.extend_tokens,), dtype=torch.int64
    )
    return tuple(
        tensor.to(device)
        for tensor in (prefix, sequence, last_location, free_pages, output)
    )


def reference(
    prefix: torch.Tensor,
    sequence: torch.Tensor,
    last_location: torch.Tensor,
    free_pages: torch.Tensor,
) -> torch.Tensor:
    prefix_values = prefix.cpu().tolist()
    sequence_values = sequence.cpu().tolist()
    last_values = last_location.cpu().tolist()
    free_values = free_pages.cpu().tolist()
    output: list[int] = []
    page_cursor = 0
    for pre_len, seq_len, last_loc in zip(
        prefix_values, sequence_values, last_values, strict=True
    ):
        page_boundary = ((pre_len + ARGS.page_size - 1) // ARGS.page_size) * (
            ARGS.page_size
        )
        num_part1 = min(seq_len, page_boundary) - pre_len
        output.extend(last_loc + 1 + offset for offset in range(num_part1))

        full_start = page_boundary
        full_end = (seq_len // ARGS.page_size) * ARGS.page_size
        for token_position in range(full_start, full_end):
            relative = token_position - full_start
            page_id = free_values[page_cursor + relative // ARGS.page_size]
            output.append(page_id * ARGS.page_size + relative % ARGS.page_size)

        full_pages = max(0, full_end - full_start) // ARGS.page_size
        page_cursor += full_pages
        remainder = seq_len - full_end
        if remainder:
            page_id = free_values[page_cursor]
            output.extend(
                page_id * ARGS.page_size + offset for offset in range(remainder)
            )
            page_cursor += 1
    return torch.tensor(output, dtype=torch.int64)


def selected_kernel() -> tuple[Any, tuple[int, ...]]:
    if ARGS.variant == "exact-dynamic":
        return allocator_module.alloc_extend_kernel, ()
    max_tokens = triton.next_power_of_2(ARGS.batch_size * ARGS.extend_tokens)
    return alloc_extend_kernel_static_bound, (max_tokens,)


def create_dp_group() -> tuple[Any | None, list[int]]:
    if WORLD_SIZE == 1:
        return None, [0]
    if WORLD_SIZE % LOCAL_WORLD_SIZE != 0:
        raise RuntimeError(
            f"WORLD_SIZE={WORLD_SIZE} must be divisible by "
            f"LOCAL_WORLD_SIZE={LOCAL_WORLD_SIZE}"
        )
    selected = None
    selected_ranks: list[int] = []
    for slot in range(LOCAL_WORLD_SIZE):
        ranks = [
            node * LOCAL_WORLD_SIZE + slot
            for node in range(WORLD_SIZE // LOCAL_WORLD_SIZE)
        ]
        group = dist.new_group(ranks=ranks, backend="hccl")
        if slot == LOCAL_RANK:
            selected = group
            selected_ranks = ranks
    assert selected is not None
    return selected, selected_ranks


def run() -> dict[str, Any]:
    emit("worker_start", cache=CACHE_INFO)
    torch.npu.set_device(LOCAL_RANK)
    device = torch.device(f"npu:{LOCAL_RANK}")
    emit("before_dist_init")
    dist.init_process_group(
        backend="hccl", timeout=timedelta(seconds=ARGS.dist_timeout_seconds)
    )
    emit("after_dist_init")
    dp_group, dp_ranks = create_dp_group()
    emit("before_global_barrier")
    dist.barrier()
    emit("after_global_barrier")

    prefix, sequence, last_location, free_pages, output = make_inputs(device)
    expected = reference(prefix, sequence, last_location, free_pages)
    kernel, extra_constexpr = selected_kernel()
    bs_upper = triton.next_power_of_2(ARGS.batch_size)
    kernel_args = (
        prefix,
        sequence,
        last_location,
        free_pages,
        output,
        bs_upper,
        ARGS.page_size,
        *extra_constexpr,
    )
    metadata = {
        "case": ARGS.case,
        "phase": ARGS.phase,
        "variant": ARGS.variant,
        "hostname": socket.gethostname(),
        "cache": CACHE_INFO,
        "versions": distribution_versions(),
        "triton_artifacts": module_artifacts(),
        "triton_module": str(Path(triton.__file__).resolve()),
        "allocator_source": str(Path(allocator_module.__file__).resolve()),
        "allocator_sha256": sha256_file(allocator_module.__file__),
        "shape": {
            "batch_size": ARGS.batch_size,
            "bs_upper": bs_upper,
            "page_size": ARGS.page_size,
            "prefix_tokens": ARGS.prefix_tokens,
            "extend_tokens": ARGS.extend_tokens,
        },
        "dp_ranks": dp_ranks,
    }

    torch.npu.synchronize()
    emit("before_compile", metadata=metadata, cache=cache_summary())
    compile_started = time.perf_counter()
    if not hasattr(kernel, "warmup"):
        raise RuntimeError(
            "installed Triton JITFunction has no warmup(); cannot separate "
            "compilation from execution"
        )
    kernel.warmup(*kernel_args, grid=(ARGS.batch_size,))
    compile_seconds = time.perf_counter() - compile_started
    emit(
        "after_compile",
        elapsed_seconds=compile_seconds,
        cache=cache_summary(),
    )

    emit("before_launch")
    launch_started = time.perf_counter()
    kernel[(ARGS.batch_size,)](*kernel_args)
    launch_seconds = time.perf_counter() - launch_started
    emit("after_launch", elapsed_seconds=launch_seconds)

    emit("before_sync")
    sync_started = time.perf_counter()
    torch.npu.synchronize()
    sync_seconds = time.perf_counter() - sync_started
    emit("after_sync", elapsed_seconds=sync_seconds)

    actual = output.cpu()
    correct = torch.equal(actual, expected)
    mismatch_count = int((actual != expected).sum().item())
    emit("after_correctness", correct=correct, mismatch_count=mismatch_count)

    hot_seconds: list[float] = []
    for repeat in range(ARGS.hot_repeats):
        torch.npu.synchronize()
        started = time.perf_counter()
        kernel[(ARGS.batch_size,)](*kernel_args)
        torch.npu.synchronize()
        elapsed = time.perf_counter() - started
        hot_seconds.append(elapsed)
        emit("after_hot_launch", repeat=repeat, elapsed_seconds=elapsed)

    if WORLD_SIZE > 1:
        gather_input = torch.tensor(
            [float(RANK), compile_seconds], dtype=torch.float32, device=device
        )
        gathered = [torch.empty_like(gather_input) for _ in dp_ranks]
        emit("before_dp_all_gather", dp_ranks=dp_ranks)
        gather_started = time.perf_counter()
        dist.all_gather(gathered, gather_input, group=dp_group)
        gather_seconds = time.perf_counter() - gather_started
        gather_values = [tensor.cpu().tolist() for tensor in gathered]
        emit(
            "after_dp_all_gather",
            elapsed_seconds=gather_seconds,
            values=gather_values,
        )
    else:
        gather_seconds = 0.0
        gather_values = [[0.0, compile_seconds]]

    return {
        **metadata,
        "compile_seconds": compile_seconds,
        "launch_seconds": launch_seconds,
        "sync_seconds": sync_seconds,
        "hot_seconds": hot_seconds,
        "dp_all_gather_seconds": gather_seconds,
        "dp_all_gather_values": gather_values,
        "correct": correct,
        "mismatch_count": mismatch_count,
        "final_cache": cache_summary(),
    }


def main() -> int:
    try:
        result = run()
        RESULT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        emit("complete", result_path=str(RESULT_PATH), correct=result["correct"])
        return 0 if result["correct"] else 4
    except BaseException as error:
        emit("failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
