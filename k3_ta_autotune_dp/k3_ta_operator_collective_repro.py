#!/usr/bin/env python3
"""Reproduce Kimi-K3 Triton-Ascend autotune and DP-wait behavior.

This program has two intentionally separate launch paths:

* ``forced-autotune`` launches the decorated Triton kernels directly.  It
  benchmarks all configured candidates even when the SGLang production wrapper
  contains an NPU static-config bypass.  Use this path to compare TA versions.
* ``production`` calls the public SGLang wrappers.  Use this path to compare the
  unpatched and static-config implementations seen by Kimi-K3.

With 4 nodes and 16 processes per node, ranks are split into 16 DP groups of
four ranks: [0, 16, 32, 48], [1, 17, 33, 49], ... .  This mirrors the K3
TP64/DP4 deployment closely enough to show the diagnostic pattern in which one
rank remains in JIT/autotune while its peers appear blocked in all_gather.
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
import subprocess
import sys
import time
from datetime import timedelta
from typing import Any, Callable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, help="Result label, e.g. cann90-ta321")
    parser.add_argument("--phase", choices=("cold", "warm"), required=True)
    parser.add_argument(
        "--launch-path",
        choices=("forced-autotune", "production"),
        default="forced-autotune",
    )
    parser.add_argument("--op", choices=("cumsum", "kda", "both"), default="both")
    parser.add_argument(
        "--layout", choices=("fixed", "varlen", "both"), default="varlen"
    )
    parser.add_argument("--tokens", type=int, default=70)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument(
        "--steady-repeats",
        type=int,
        default=2,
        help="Calls after the first call; these hit the in-process autotune cache.",
    )
    parser.add_argument(
        "--cache-root",
        required=True,
        help="Case-level cache root. The script adds node/rank subdirectories.",
    )
    parser.add_argument(
        "--cache-layout",
        choices=("per-node", "per-rank"),
        default="per-node",
        help="per-node mirrors 16 local SGLang workers sharing a Triton cache.",
    )
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--dist-timeout-seconds", type=int, default=900)
    parser.add_argument(
        "--dump-stack-after-seconds",
        type=int,
        default=180,
        help="Dump Python stacks while a Triton benchmark or collective is stuck; 0 disables.",
    )
    parser.add_argument(
        "--inject-delay-global-rank",
        type=int,
        default=-1,
        help="Diagnostic control: sleep this global rank before DP all_gather.",
    )
    parser.add_argument(
        "--inject-delay-node-rank",
        type=int,
        default=-1,
        help="Diagnostic control: sleep all ranks on this node before DP all_gather.",
    )
    parser.add_argument("--inject-delay-seconds", type=float, default=0.0)
    parser.add_argument("--cumsum-atol", type=float, default=1e-5)
    parser.add_argument("--kda-atol", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def distributed_coordinates() -> tuple[int, int, int, int]:
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    local_world_size = int(os.getenv("LOCAL_WORLD_SIZE", "1"))
    return rank, world_size, local_rank, local_world_size


def configure_cache(args: argparse.Namespace) -> dict[str, Any]:
    """Set cache variables before importing torch, Triton, or SGLang."""

    rank, _, local_rank, local_world_size = distributed_coordinates()
    node_rank = rank // local_world_size
    node_root = Path(args.cache_root).resolve() / f"node{node_rank}"
    if args.cache_layout == "per-node":
        process_root = node_root
    else:
        process_root = node_root / f"rank{rank}"

    paths = {
        "SGLANG_CACHE_DIR": process_root / "sglang",
        "TRITON_CACHE_DIR": process_root / "triton",
        "TORCHINDUCTOR_CACHE_DIR": process_root / "inductor",
    }
    preexisting = {name: path.exists() for name, path in paths.items()}
    for name, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)

    return {
        "rank": rank,
        "local_rank": local_rank,
        "node_rank": node_rank,
        "cache_layout": args.cache_layout,
        "paths": {name: str(path) for name, path in paths.items()},
        "preexisting_at_process_start": preexisting,
    }


ARGS = parse_args()
CACHE_INFO = configure_cache(ARGS)
RANK, WORLD_SIZE, LOCAL_RANK, LOCAL_WORLD_SIZE = distributed_coordinates()
NODE_RANK = CACHE_INFO["node_rank"]

if ARGS.dump_stack_after_seconds > 0:
    faulthandler.dump_traceback_later(
        ARGS.dump_stack_after_seconds, repeat=True, file=sys.stderr
    )

# Cache variables must be configured before these imports.
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch_npu  # noqa: E402
import triton  # noqa: E402

import sglang.kernels.ops.attention.fla.cumsum as cumsum_module  # noqa: E402
import sglang.kernels.ops.attention.fla.kda as kda_module  # noqa: E402
from sglang.kernels.ops.attention.fla.index import (  # noqa: E402
    prepare_chunk_indices,
)


def log(event: str, **fields: Any) -> None:
    record = {
        "event": event,
        "monotonic": time.monotonic(),
        "rank": RANK,
        "local_rank": LOCAL_RANK,
        "node_rank": NODE_RANK,
        **fields,
    }
    print("K3_TA_EVENT=" + json.dumps(record, sort_keys=True), flush=True)


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def distribution_versions() -> list[tuple[str, str]]:
    names = {
        "torch",
        "torch-npu",
        "triton",
        "triton-ascend",
        "sglang",
        "sgl-kernel-npu",
    }
    found: list[tuple[str, str]] = []
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name") or ""
        if name.lower() in names:
            found.append((name, distribution.version))
    return sorted(found)


def module_artifact(module_name: str) -> dict[str, str] | None:
    try:
        module = importlib.import_module(module_name)
        path = Path(module.__file__).resolve()
    except (AttributeError, ImportError, TypeError):
        return None
    return {"path": str(path), "sha256": sha256_file(path)}


def candidate_count(kernel: Any) -> int | None:
    configs = getattr(kernel, "configs", None)
    return len(configs) if configs is not None else None


def autotune_cache_size(kernel: Any) -> int | None:
    cache = getattr(kernel, "cache", None)
    return len(cache) if cache is not None else None


def make_sequence_metadata(
    tokens: int, chunk_size: int, varlen: bool, device: torch.device
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not varlen:
        return None, None
    cu_seqlens = torch.tensor([0, tokens], dtype=torch.int64, device=device)
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    return cu_seqlens, chunk_indices


def synchronize() -> None:
    torch.npu.synchronize()


def timed_call(name: str, call: Callable[[], Any]) -> tuple[Any, float]:
    synchronize()
    log("phase_start", phase=name)
    started = time.perf_counter()
    output = call()
    synchronize()
    elapsed = time.perf_counter() - started
    log("phase_end", phase=name, elapsed_seconds=elapsed)
    return output, elapsed


def cumsum_reference(
    source: torch.Tensor, chunk_size: int, scale: float
) -> torch.Tensor:
    output = torch.empty_like(source, dtype=torch.float32)
    for start in range(0, source.shape[1], chunk_size):
        end = min(start + chunk_size, source.shape[1])
        output[:, start:end] = source[:, start:end].float().cumsum(dim=1) * scale
    return output


def launch_cumsum(
    source: torch.Tensor,
    *,
    chunk_size: int,
    scale: float,
    cu_seqlens: torch.Tensor | None,
    chunk_indices: torch.Tensor | None,
) -> torch.Tensor:
    if ARGS.launch_path == "production":
        return cumsum_module.chunk_local_cumsum_vector(
            source,
            chunk_size=chunk_size,
            scale=scale,
            cu_seqlens=cu_seqlens,
            output_dtype=torch.float32,
            chunk_indices=chunk_indices,
        )

    batch, tokens, heads, width = source.shape
    num_chunks = (
        triton.cdiv(tokens, chunk_size)
        if cu_seqlens is None
        else len(chunk_indices)
    )
    output = torch.empty_like(source, dtype=torch.float32)

    def grid(meta: dict[str, Any]) -> tuple[int, int, int]:
        return (
            triton.cdiv(width, meta["BS"]),
            num_chunks,
            batch * heads,
        )

    # Launch the Autotuner object directly. This intentionally bypasses the
    # production static-config branch added for NPU.
    cumsum_module.chunk_local_cumsum_vector_kernel[grid](
        s=source,
        o=output,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=tokens,
        B=batch,
        H=heads,
        S=width,
        BT=chunk_size,
        HEAD_FIRST=False,
        REVERSE=False,
        HAS_SCALE=True,
        IS_VARLEN=cu_seqlens is not None,
    )
    return output


def kda_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, tokens, heads, _ = q.shape
    triangular = torch.zeros(
        batch,
        tokens,
        heads,
        chunk_size,
        device=q.device,
        dtype=torch.float32,
    )
    query_key = torch.zeros_like(triangular)
    for start in range(0, tokens, chunk_size):
        end = min(start + chunk_size, tokens)
        q_chunk = q[:, start:end].float()
        k_chunk = k[:, start:end].float()
        g_chunk = g[:, start:end].float()
        qg = q_chunk * torch.exp2(g_chunk)
        kg = k_chunk * torch.exp2(g_chunk)
        k_inv_g = k_chunk * torch.exp2(-g_chunk)
        qk = torch.einsum("bihk,bjhk->bhij", qg, k_inv_g) * scale
        kk = torch.einsum("bihk,bjhk->bhij", kg, k_inv_g)
        length = end - start
        lower_inclusive = torch.tril(
            torch.ones(length, length, dtype=torch.bool, device=q.device)
        )
        lower_strict = torch.tril(
            torch.ones(length, length, dtype=torch.bool, device=q.device),
            diagonal=-1,
        )
        qk = qk.masked_fill(~lower_inclusive, 0).permute(0, 2, 1, 3)
        kk = kk.masked_fill(~lower_strict, 0).permute(0, 2, 1, 3)
        kk *= beta[:, start:end, :, None].float()
        query_key[:, start:end, :, :length] = qk
        triangular[:, start:end, :, :length] = kk
    return triangular, query_key


def launch_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float,
    chunk_size: int,
    cu_seqlens: torch.Tensor | None,
    chunk_indices: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if ARGS.launch_path == "production":
        return kda_module.chunk_kda_scaled_dot_kkt_fwd(
            q,
            k,
            gk=g,
            beta=beta,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            output_dtype=torch.float32,
        )

    batch, tokens, heads, width = k.shape
    block_time = chunk_size
    num_chunks = (
        triton.cdiv(tokens, block_time)
        if cu_seqlens is None
        else len(chunk_indices)
    )
    block_inner = min(16, block_time)
    inner_blocks = triton.cdiv(block_time, block_inner)
    block_width = max(triton.next_power_of_2(width), 16)
    triangular = torch.zeros(
        batch,
        tokens,
        heads,
        block_time,
        device=k.device,
        dtype=torch.float32,
    )
    query_key = torch.zeros_like(triangular)

    # These are the two Autotuner objects bypassed by the NPU static-config
    # production patch. Calling them directly is the TA-version A/B probe.
    kda_module.chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter[
        (num_chunks, inner_blocks * inner_blocks, batch * heads)
    ](
        q=q,
        k=k,
        g=g,
        beta=beta,
        A=triangular,
        Aqk=query_key,
        scale=scale,
        gk_scale=1.0,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=tokens,
        H=heads,
        K=width,
        BT=block_time,
        BC=block_inner,
        NC=inner_blocks,
        IS_VARLEN=cu_seqlens is not None,
    )
    kda_module.chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra[
        (num_chunks, inner_blocks, batch * heads)
    ](
        q=q,
        k=k,
        g=g,
        beta=beta,
        A=triangular,
        Aqk=query_key,
        scale=scale,
        gk_scale=1.0,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=tokens,
        H=heads,
        K=width,
        BT=block_time,
        BC=block_inner,
        BK=block_width,
        IS_VARLEN=cu_seqlens is not None,
    )
    return triangular, query_key


def run_cumsum_case(device: torch.device, layout: str) -> dict[str, Any]:
    varlen = layout == "varlen"
    shape = (1, ARGS.tokens, ARGS.heads, ARGS.head_dim)
    source = torch.randn(shape, dtype=torch.bfloat16, device=device) * 0.05
    scale = 1.4426950408889634
    cu_seqlens, chunk_indices = make_sequence_metadata(
        ARGS.tokens, ARGS.chunk_size, varlen, device
    )
    kernel = cumsum_module.chunk_local_cumsum_vector_kernel
    cache_before = autotune_cache_size(kernel)

    def call() -> torch.Tensor:
        return launch_cumsum(
            source,
            chunk_size=ARGS.chunk_size,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )

    actual, first_seconds = timed_call(f"cumsum_{layout}_first", call)
    expected = cumsum_reference(source, ARGS.chunk_size, scale)
    synchronize()
    max_abs_error = (actual - expected).abs().max().item()
    steady_seconds: list[float] = []
    for index in range(ARGS.steady_repeats):
        _, elapsed = timed_call(f"cumsum_{layout}_steady_{index}", call)
        steady_seconds.append(elapsed)
    return {
        "layout": layout,
        "first_seconds": first_seconds,
        "steady_seconds": steady_seconds,
        "max_abs_error": max_abs_error,
        "correct": max_abs_error <= ARGS.cumsum_atol,
        "candidate_count": candidate_count(kernel),
        "autotune_cache_size_before": cache_before,
        "autotune_cache_size_after": autotune_cache_size(kernel),
    }


def run_kda_case(device: torch.device, layout: str) -> dict[str, Any]:
    varlen = layout == "varlen"
    shape = (1, ARGS.tokens, ARGS.heads, ARGS.head_dim)
    q = torch.randn(shape, dtype=torch.bfloat16, device=device) * 0.05
    k = torch.randn(shape, dtype=torch.bfloat16, device=device) * 0.05
    g = torch.randn(shape, dtype=torch.float32, device=device) * 0.01
    beta = torch.sigmoid(
        torch.randn(shape[:-1], dtype=torch.float32, device=device)
    )
    scale = 1.0 / (ARGS.head_dim**0.5)
    cu_seqlens, chunk_indices = make_sequence_metadata(
        ARGS.tokens, ARGS.chunk_size, varlen, device
    )
    inter_kernel = kda_module.chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter
    intra_kernel = kda_module.chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra
    cache_before = {
        "inter": autotune_cache_size(inter_kernel),
        "intra": autotune_cache_size(intra_kernel),
    }

    def call() -> tuple[torch.Tensor, torch.Tensor]:
        return launch_kda(
            q,
            k,
            g,
            beta,
            scale=scale,
            chunk_size=ARGS.chunk_size,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )

    (triangular, query_key), first_seconds = timed_call(
        f"kda_{layout}_first", call
    )
    triangular_ref, query_key_ref = kda_reference(
        q, k, g, beta, scale, ARGS.chunk_size
    )
    synchronize()
    triangular_error = (triangular - triangular_ref).abs().max().item()
    query_key_error = (query_key - query_key_ref).abs().max().item()
    steady_seconds: list[float] = []
    for index in range(ARGS.steady_repeats):
        _, elapsed = timed_call(f"kda_{layout}_steady_{index}", call)
        steady_seconds.append(elapsed)
    return {
        "layout": layout,
        "first_seconds": first_seconds,
        "steady_seconds": steady_seconds,
        "triangular_max_abs_error": triangular_error,
        "query_key_max_abs_error": query_key_error,
        "correct": max(triangular_error, query_key_error) <= ARGS.kda_atol,
        "candidate_count": {
            "inter": candidate_count(inter_kernel),
            "intra": candidate_count(intra_kernel),
        },
        "autotune_cache_size_before": cache_before,
        "autotune_cache_size_after": {
            "inter": autotune_cache_size(inter_kernel),
            "intra": autotune_cache_size(intra_kernel),
        },
    }


def create_dp_group() -> tuple[Any | None, list[int]]:
    if WORLD_SIZE == 1:
        return None, [0]
    if WORLD_SIZE % LOCAL_WORLD_SIZE != 0:
        raise RuntimeError(
            f"WORLD_SIZE={WORLD_SIZE} is not divisible by "
            f"LOCAL_WORLD_SIZE={LOCAL_WORLD_SIZE}"
        )
    num_nodes = WORLD_SIZE // LOCAL_WORLD_SIZE
    selected_group = None
    selected_ranks: list[int] = []
    # All ranks create all groups in identical order. A local slot therefore
    # communicates with the matching slot on each of the four nodes.
    for slot in range(LOCAL_WORLD_SIZE):
        ranks = [node * LOCAL_WORLD_SIZE + slot for node in range(num_nodes)]
        group = dist.new_group(ranks=ranks, backend="hccl")
        if slot == LOCAL_RANK:
            selected_group = group
            selected_ranks = ranks
    assert selected_group is not None
    return selected_group, selected_ranks


def run_dp_all_gather(
    device: torch.device, group: Any | None, group_ranks: list[int]
) -> dict[str, Any]:
    if WORLD_SIZE == 1:
        return {"seconds": 0.0, "group_ranks": group_ranks, "values": [0.0]}

    should_delay = (
        RANK == ARGS.inject_delay_global_rank
        or NODE_RANK == ARGS.inject_delay_node_rank
    )
    if should_delay and ARGS.inject_delay_seconds > 0:
        log(
            "injected_delay_start",
            seconds=ARGS.inject_delay_seconds,
            group_ranks=group_ranks,
        )
        time.sleep(ARGS.inject_delay_seconds)
        log("injected_delay_end", group_ranks=group_ranks)

    value = torch.tensor([float(RANK)], dtype=torch.float32, device=device)
    outputs = [torch.empty_like(value) for _ in group_ranks]
    synchronize()
    log("phase_start", phase="dp_all_gather", group_ranks=group_ranks)
    started = time.perf_counter()
    dist.all_gather(outputs, value, group=group)
    synchronize()
    elapsed = time.perf_counter() - started
    values = [tensor.item() for tensor in outputs]
    log(
        "phase_end",
        phase="dp_all_gather",
        elapsed_seconds=elapsed,
        group_ranks=group_ranks,
    )
    return {"seconds": elapsed, "group_ranks": group_ranks, "values": values}


def first_call_total(result: dict[str, Any]) -> float:
    total = 0.0
    for operator in ("cumsum", "kda"):
        for item in result.get(operator, []):
            total += float(item["first_seconds"])
    return total


def correctness(result: dict[str, Any]) -> bool:
    return all(
        item["correct"]
        for operator in ("cumsum", "kda")
        for item in result.get(operator, [])
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    torch.npu.set_device(LOCAL_RANK)
    device = torch.device(f"npu:{LOCAL_RANK}")
    torch.manual_seed(ARGS.seed)

    if WORLD_SIZE > 1:
        dist.init_process_group(
            backend="hccl", timeout=timedelta(seconds=ARGS.dist_timeout_seconds)
        )
    group, group_ranks = create_dp_group()

    source_files = {
        "cumsum": str(Path(cumsum_module.__file__).resolve()),
        "kda": str(Path(kda_module.__file__).resolve()),
    }
    environment = {
        "hostname": socket.gethostname(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "triton_import_version": triton.__version__,
        "triton_import_file": str(Path(triton.__file__).resolve()),
        "triton_artifacts": {
            module_name: module_artifact(module_name)
            for module_name in (
                "triton.backends.ascend.compiler",
                "triton.backends.ascend.driver",
                "triton.runtime.autotuner",
            )
        },
        "distributions": distribution_versions(),
        "source_files": source_files,
        "source_sha256": {
            name: sha256_file(path) for name, path in source_files.items()
        },
        "git_head": git_value("rev-parse", "HEAD"),
        "git_status": git_value("status", "--porcelain"),
        "cache": CACHE_INFO,
        "selected_env": {
            name: os.getenv(name)
            for name in sorted(os.environ)
            if name.startswith(
                (
                    "ASCEND_",
                    "HCCL_",
                    "SGLANG_",
                    "TORCH",
                    "TRITON_",
                )
            )
        },
    }
    log("environment", environment=environment)

    if WORLD_SIZE > 1:
        log("world_barrier_start")
        dist.barrier()
        synchronize()
        log("world_barrier_end")

    layouts = [ARGS.layout] if ARGS.layout != "both" else ["fixed", "varlen"]
    result: dict[str, Any] = {
        "case": ARGS.case,
        "phase": ARGS.phase,
        "launch_path": ARGS.launch_path,
        "operator": ARGS.op,
        "layouts": layouts,
        "rank": RANK,
        "world_size": WORLD_SIZE,
        "local_rank": LOCAL_RANK,
        "local_world_size": LOCAL_WORLD_SIZE,
        "node_rank": NODE_RANK,
        "environment": environment,
        "cumsum": [],
        "kda": [],
    }

    suite_started = time.perf_counter()
    if ARGS.op in ("cumsum", "both"):
        for layout in layouts:
            result["cumsum"].append(run_cumsum_case(device, layout))
    if ARGS.op in ("kda", "both"):
        for layout in layouts:
            result["kda"].append(run_kda_case(device, layout))
    result["operator_suite_seconds"] = time.perf_counter() - suite_started
    result["first_call_total_seconds"] = first_call_total(result)
    result["correct"] = correctness(result)
    result["dp_all_gather"] = run_dp_all_gather(device, group, group_ranks)

    result_dir = Path(ARGS.result_dir).resolve()
    write_json(result_dir / f"rank{RANK}.json", result)
    print("K3_TA_RESULT=" + json.dumps(result, sort_keys=True), flush=True)

    if WORLD_SIZE > 1:
        metrics = torch.tensor(
            [
                result["first_call_total_seconds"],
                result["operator_suite_seconds"],
                result["dp_all_gather"]["seconds"],
                1.0 if result["correct"] else 0.0,
            ],
            dtype=torch.float32,
            device=device,
        )
        gathered = [torch.empty_like(metrics) for _ in range(WORLD_SIZE)]
        dist.all_gather(gathered, metrics)
        synchronize()
        if RANK == 0:
            rank_metrics = [tensor.cpu().tolist() for tensor in gathered]
            summary = {
                "case": ARGS.case,
                "phase": ARGS.phase,
                "launch_path": ARGS.launch_path,
                "world_size": WORLD_SIZE,
                "columns": [
                    "first_call_total_seconds",
                    "operator_suite_seconds",
                    "dp_all_gather_seconds",
                    "correct",
                ],
                "rank_metrics": rank_metrics,
                "max_first_call_seconds": max(row[0] for row in rank_metrics),
                "min_first_call_seconds": min(row[0] for row in rank_metrics),
                "first_call_skew_seconds": max(row[0] for row in rank_metrics)
                - min(row[0] for row in rank_metrics),
                "max_dp_all_gather_seconds": max(row[2] for row in rank_metrics),
                "all_correct": all(row[3] == 1.0 for row in rank_metrics),
            }
            write_json(result_dir / "summary.json", summary)
            print("K3_TA_SUMMARY=" + json.dumps(summary, sort_keys=True), flush=True)
        dist.barrier()
        dist.destroy_process_group()

    return 0 if result["correct"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
