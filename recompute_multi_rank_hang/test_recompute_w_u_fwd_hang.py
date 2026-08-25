"""Regression reproducer for the Qwen3-Next recompute_w_u_fwd NPU hang.

The default fixtures were captured from the blocking, ZBAL-free TP4/DP1 run on
813.  Each worker loads the input from its own TP rank and all four workers
launch the operator together.  The test is opt-in when collected by pytest; it
can also be invoked directly with no arguments.
"""

from __future__ import annotations

import argparse
import faulthandler
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from importlib import metadata
from pathlib import Path


DEFAULT_CAPTURE_PATTERN = str(
    Path(__file__).resolve().parent
    / "fixtures"
    / "gdn_recompute_call160_npu{rank}_20260824.pt"
)
TENSOR_ARGUMENTS = ("k", "v", "beta", "g_cumsum", "A", "cu_seqlens")


def _resolve_input_path(input_spec: str, rank: int) -> Path:
    if "{rank}" in input_spec:
        return Path(input_spec.format(rank=rank)).resolve()
    path = Path(input_spec).resolve()
    if path.is_dir():
        return path / f"gdn_recompute_hung_npu{rank}_tp4_dp1.pt"
    return path


def _worker(args: argparse.Namespace) -> int:
    os.environ.setdefault("ASCEND_LAUNCH_BLOCKING", "1")
    os.environ.setdefault("TASK_QUEUE_ENABLE", "1")
    os.environ.setdefault("TRITON_F32_DEFAULT", "ieee")
    os.environ.setdefault("TORCH_SHOW_CPP_STACKTRACES", "1")
    faulthandler.enable(all_threads=True)
    faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)

    import torch
    import torch_npu  # noqa: F401

    from sgl_kernel_npu.fla.wy_fast import recompute_w_u_fwd_npu

    device = torch.device(f"npu:{args.rank}")
    torch.npu.set_device(device)
    input_path = _resolve_input_path(args.input, args.rank)
    try:
        captured = torch.load(input_path, map_location="cpu", weights_only=True)
    except TypeError:
        captured = torch.load(input_path, map_location="cpu")
    original_seq_len = captured["k"].shape[1]
    if args.slice_seq_len > 0:
        first_sequence_len = int(captured["cu_seqlens"][1])
        replay_seq_len = min(args.slice_seq_len, first_sequence_len)
        captured = dict(captured)
        for name in ("k", "v", "beta", "g_cumsum", "A"):
            captured[name] = captured[name][:, :replay_seq_len].contiguous()
        captured["cu_seqlens"] = torch.tensor([0, replay_seq_len], dtype=torch.int32)
    inputs = {
        name: None if captured[name] is None else captured[name].to(device)
        for name in TENSOR_ARGUMENTS
    }

    print(
        "WORKER",
        f"rank={args.rank}",
        f"pid={os.getpid()}",
        f"input={input_path}",
        f"captured_call={captured.get('trace_call', 'unknown')}",
        f"captured_device={captured.get('trace_device', 'unknown')}",
        f"original_seq_len={original_seq_len}",
        f"replay_seq_len={inputs['k'].shape[1]}",
        f"triton_ascend={metadata.version('triton-ascend')}",
        f"k={tuple(inputs['k'].shape)}",
        f"v={tuple(inputs['v'].shape)}",
        f"A={tuple(inputs['A'].shape)}",
        flush=True,
    )

    def launch() -> None:
        recompute_w_u_fwd_npu(
            inputs["k"],
            inputs["v"],
            inputs["beta"],
            inputs["g_cumsum"],
            inputs["A"],
            inputs["cu_seqlens"],
        )

    # Do not execute an unsynchronised warmup: the captured production failure
    # occurred while all TP ranks entered the same model-layer call together.
    barrier = Path(args.barrier_dir)
    (barrier / f"ready-{args.rank}").touch()
    while not (barrier / "go").exists():
        time.sleep(0.01)

    started = time.monotonic()
    for iteration in range(1, args.repeats + 1):
        trace = args.trace_every > 0 and (
            iteration == 1 or iteration % args.trace_every == 0
        )
        if trace:
            print(f"CALL_BEGIN rank={args.rank} iteration={iteration}", flush=True)
        launch()
        if trace:
            print(f"LAUNCH_RETURN rank={args.rank} iteration={iteration}", flush=True)
        torch.npu.synchronize()
        if trace:
            print(f"SYNC_END rank={args.rank} iteration={iteration}", flush=True)
        if iteration == 1 or iteration % args.progress_every == 0:
            print(
                "PROGRESS",
                f"rank={args.rank}",
                f"iteration={iteration}",
                f"elapsed={time.monotonic() - started:.3f}s",
                flush=True,
            )

    print(
        "PASS",
        f"rank={args.rank}",
        f"iterations={args.repeats}",
        f"elapsed={time.monotonic() - started:.3f}s",
        flush=True,
    )
    return 0


def _capture_command(command: list[str], output_path: Path) -> None:
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
            check=False,
        )
        output = (
            f"COMMAND: {' '.join(command)}\nEXIT: {result.returncode}\n\n"
            f"{result.stdout}"
        )
    except Exception as error:
        output = f"COMMAND: {' '.join(command)}\nERROR: {error!r}\n"
    output_path.write_text(output, encoding="utf-8", errors="replace")


def _capture_timeout_diagnostics(
    processes: list[subprocess.Popen[str]], log_dir: Path, phase: str
) -> None:
    diagnostic_dir = log_dir / "timeout_diagnostics"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    live = [
        (rank, process)
        for rank, process in enumerate(processes)
        if process.poll() is None
    ]
    summary = [f"phase={phase}", f"time={time.strftime('%Y-%m-%d %H:%M:%S')}"]
    summary.extend(f"rank={rank} pid={process.pid}" for rank, process in live)
    (diagnostic_dir / "summary.log").write_text(
        "\n".join(summary) + "\n", encoding="utf-8"
    )

    py_spy = shutil.which("py-spy")
    if py_spy:
        for rank, process in live:
            _capture_command(
                [
                    py_spy,
                    "dump",
                    "--pid",
                    str(process.pid),
                    "--native",
                    "--full-filenames",
                ],
                diagnostic_dir / f"py-spy-rank-{rank}.log",
            )
    _capture_command(["ps", "-elf"], diagnostic_dir / "ps-elf.log")
    if shutil.which("npu-smi"):
        _capture_command(["npu-smi", "info"], diagnostic_dir / "npu-smi-info.log")

    # faulthandler appends the in-process Python stack to each worker log.
    for _, process in live:
        try:
            os.kill(process.pid, signal.SIGUSR1)
        except ProcessLookupError:
            pass
    time.sleep(3)


def _terminate(processes: list[subprocess.Popen[str]]) -> None:
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and any(p.poll() is None for p in processes):
        time.sleep(0.1)
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    for process in processes:
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass


def _coordinator(args: argparse.Namespace) -> int:
    barrier_dir = Path(tempfile.mkdtemp(prefix="wy-fast-barrier-"))
    log_dir = Path(args.log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    processes: list[subprocess.Popen[str]] = []
    log_files = []
    try:
        for rank in range(args.workers):
            log_file = (log_dir / f"worker-{rank}.log").open("w", encoding="utf-8")
            log_files.append(log_file)
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--rank",
                str(rank),
                "--input",
                str(Path(args.input).resolve()),
                "--barrier-dir",
                str(barrier_dir),
                "--repeats",
                str(args.repeats),
                "--slice-seq-len",
                str(args.slice_seq_len),
                "--progress-every",
                str(args.progress_every),
                "--trace-every",
                str(args.trace_every),
            ]
            processes.append(
                subprocess.Popen(
                    command,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
            )

        ready_deadline = time.monotonic() + args.ready_timeout
        while time.monotonic() < ready_deadline:
            if any(p.poll() not in (None, 0) for p in processes):
                print(f"ERROR phase=input-load-barrier logs={log_dir}", flush=True)
                return 2
            if all(
                (barrier_dir / f"ready-{rank}").exists() for rank in range(args.workers)
            ):
                break
            time.sleep(0.1)
        else:
            print(f"TIMEOUT phase=input-load-barrier logs={log_dir}", flush=True)
            _capture_timeout_diagnostics(processes, log_dir, "input-load-barrier")
            return 124

        (barrier_dir / "go").touch()
        started = time.monotonic()
        print(
            "START",
            f"workers={args.workers}",
            f"slice_seq_len={args.slice_seq_len}",
            f"repeats={args.repeats}",
            f"timeout={args.timeout}",
            f"logs={log_dir}",
            flush=True,
        )
        while time.monotonic() - started < args.timeout:
            if all(process.poll() is not None for process in processes):
                codes = [process.returncode for process in processes]
                if all(code == 0 for code in codes):
                    print(f"PASS exit_codes={codes} logs={log_dir}", flush=True)
                    return 0
                print(f"ERROR exit_codes={codes} logs={log_dir}", flush=True)
                return 2
            time.sleep(0.2)

        live = [
            rank for rank, process in enumerate(processes) if process.poll() is None
        ]
        print(f"TIMEOUT live_workers={live} logs={log_dir}", flush=True)
        _capture_timeout_diagnostics(processes, log_dir, "stress-loop")
        return 124
    finally:
        _terminate(processes)
        for log_file in log_files:
            log_file.close()
        shutil.rmtree(barrier_dir, ignore_errors=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--rank", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--barrier-dir", default="", help=argparse.SUPPRESS)
    parser.add_argument(
        "--input",
        default=DEFAULT_CAPTURE_PATTERN,
        help=(
            "Captured fixture, a directory containing per-rank fixtures, or a "
            "path pattern containing {rank}"
        ),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--slice-seq-len",
        type=int,
        default=0,
        help=(
            "Replay this many tokens from the first captured sequence; "
            "0 (the default) replays the exact packed production input"
        ),
    )
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--ready-timeout", type=float, default=90)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--trace-every", type=int, default=1)
    parser.add_argument("--log-dir", default="/tmp/recompute-w-u-fwd-hang")
    return parser.parse_args()


def test_recompute_w_u_fwd_kernel_does_not_hang() -> None:
    """Opt-in four-NPU regression; a watchdog timeout fails the test."""
    import pytest

    if os.getenv("RUN_RECOMPUTE_W_U_FWD_HANG_TEST") != "1":
        pytest.skip("set RUN_RECOMPUTE_W_U_FWD_HANG_TEST=1 to enable")
    fixture = os.environ.get("RECOMPUTE_W_U_FWD_CAPTURE", DEFAULT_CAPTURE_PATTERN)
    args = argparse.Namespace(
        input=fixture,
        workers=int(os.getenv("RECOMPUTE_W_U_FWD_WORKERS", "4")),
        repeats=int(os.getenv("RECOMPUTE_W_U_FWD_REPEATS", "100")),
        slice_seq_len=int(os.getenv("RECOMPUTE_W_U_FWD_SLICE_SEQ_LEN", "0")),
        timeout=float(os.getenv("RECOMPUTE_W_U_FWD_TIMEOUT", "60")),
        ready_timeout=float(os.getenv("RECOMPUTE_W_U_FWD_READY_TIMEOUT", "90")),
        progress_every=int(os.getenv("RECOMPUTE_W_U_FWD_PROGRESS_EVERY", "100")),
        trace_every=int(os.getenv("RECOMPUTE_W_U_FWD_TRACE_EVERY", "1")),
        log_dir=os.getenv("RECOMPUTE_W_U_FWD_LOG_DIR", "/tmp/recompute-w-u-fwd-hang"),
    )
    result = _coordinator(args)
    assert result == 0, (
        f"operator stress failed with exit code {result}; see {args.log_dir}"
    )


if __name__ == "__main__":
    parsed = _parse_args()
    raise SystemExit(_worker(parsed) if parsed.worker else _coordinator(parsed))
