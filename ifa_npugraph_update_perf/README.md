# IFA NPU Graph update performance diagnostic

This microbenchmark investigates the NPU Graph update path implicated by the
Qwen3-30B-A3B W8A8 EAGLE3 performance regression observed after upgrading the
native Ascend stack from CANN 9.0 to CANN 9.1.  It is designed to distinguish
stack differences from host scheduling and process-startup noise; it does not
by itself attribute the whole-model regression to CANN or IFA.

It requires only PyTorch, `torch-npu`, and one Ascend NPU.  It does not import
SGLang or load model weights.

## What it measures

Each target-verification decode step in the original model graph updates 48
`npu_fused_infer_attention_score.out` records.  The reproducer defaults to the
same full `bs=162` graph bucket and core shape:

```text
records=48
batch size=162
query tokens per request=4
total query tokens=648
query heads=16
KV heads=2
head dimension=128
query layout=TND
KV sequence length=3500 -> 3501
actual_seq_lengths_kv elements=162
actual_seq_lengths=[4, 8, ..., 648]
KV page size=128
sparse mode=3 with the production 2048x2048 boolean MTP mask
dtype=bfloat16
```

Every record receives distinct query, key, value, block-table, and output
addresses.  The workspace can be shared because the calls are issued
serially.  As in the production graph runner, `NPUGraph.update` runs on a CPU
worker while the main thread submits `NPUGraph.replay`.  The worker explicitly
selects the requested NPU because torch-npu device selection is thread-local;
without that initialization a run targeting a nonzero device can silently
open device 0 and hang during final synchronization.

The eager IFA path is measured as a control.  Before `--batch-size` was added,
the diagnostic used five independent single-query entries rather than the
production 162-request target-verify shape.  Those legacy measurements showed
flat eager latency and a slower 48-record update:

| Stack | 48-record update p50 | Eager submit p50 | Eager sync-total p50 |
| --- | ---: | ---: | ---: |
| 0723 / CANN 9.0 | 4,158.525 us | 44.880 us | 112.777 us |
| 0813 / CANN 9.1 | 4,630.838 us | 44.726 us | 113.612 us |
| Difference | **+472.313 us (+11.36%)** | -0.154 us | +0.835 us |

That legacy-shape snapshot motivated this reproducer, but repeated validation showed that
the result is sensitive to CPU placement.  Unpinned runs produced a bimodal
slow tail primarily in the 0813 container; applying the same CPU affinity to
both containers removed the slow tail and the 0813 stack was not slower.  The
full-model regression therefore remains real, but this operator-level result
must be treated as a scheduling-sensitive diagnostic rather than a standalone
CANN attribution.

## Run

The default command runs the production-shaped 48-record, `bs=162` case and
prints JSON:

```bash
NPU_DEVICE_INDEX=0 \
python3 ifa_npugraph_update_perf/ifa_npugraph_update_repro.py
```

For a lightweight control that does not reproduce the full graph bucket, pass
`--batch-size 1`.  Keep this result separate from the `bs=162` comparison.

The important fields are:

- `update`: time spent in `NPUGraph.update` on the CPU worker;
- `replay_submit`: main-thread `NPUGraph.replay` submission time;
- `overlap_total`: wall time from starting the update worker until both update
  and replay submission have returned;
- `eager_submit` and `eager_sync_total`: same-stack controls for a single IFA
  call.

Absolute host latency depends on the server, CPU placement, and load.  The
default is therefore report-only.  Do not reuse the historical 4,400 us
threshold: the repeated machine-204 measurements did not support it as a
stable separator.  If a CI gate is needed, establish it from repeated runs of
the healthy stack with the same CPU affinity, then pass
`--max-update-p50-us` and/or `--max-overlap-p50-us`.

For stability analysis, collect several timing blocks from one graph capture
and Python process.  The top-level statistics cover all iterations, while the
`blocks` array preserves each block's p50/p95 values.  This avoids treating
Python startup, graph capture, and first-use initialization as run-to-run
latency variation:

```bash
NPU_DEVICE_INDEX=14 \
python3 ifa_npugraph_update_perf/ifa_npugraph_update_repro.py \
  --warmup 20 \
  --iters 150 \
  --measurement-blocks 10 \
  --batch-size 162
```

Run only the eager control:

```bash
NPU_DEVICE_INDEX=0 \
python3 ifa_npugraph_update_perf/ifa_npugraph_update_repro.py \
  --eager-only
```

Capture a CPU+NPU profile with stack collection disabled by the script's
profiler configuration:

```bash
NPU_DEVICE_INDEX=0 \
python3 ifa_npugraph_update_perf/ifa_npugraph_update_repro.py \
  --warmup 5 \
  --iters 30 \
  --profile-dir /tmp/ifa-npugraph-profile
```

The profile should be analyzed using the generated
`ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_*.db`, `kernel_details.csv`,
and `trace_view.json`.  CANN API durations are nested; for example, a CANN 9.1
IFA tiling range can be contained inside `GetWorkspaceSize`.  Use interval
union or exclusive/self time rather than adding every displayed API duration
as wall time.

## Compare two containers repeatedly

`run_container_matrix.sh` runs the same checked-in Python source in two
already-running containers.  It alternates container order on each round so
that startup order and machine drift do not always favor the same stack.  It
also refuses to begin a measurement while `npu-smi` reports another NPU
process.

On machine 204, the validated containers are:

```text
old: sglwmc-723
     sglang:cann9.0.0-a3-20260723
new: sglwmc-813-ifa-repro
     sglang:cann9.1.0-a3-20260813
```

Make sure an unrelated full-model job or NPU-process guard is not running,
then execute:

```bash
docker start sglwmc-723 sglwmc-813-ifa-repro

result_dir=/data/wzy/ifa_npugraph_matrix_$(date +%Y%m%d_%H%M%S)
bash ifa_npugraph_update_perf/run_container_matrix.sh \
  --old-container sglwmc-723 \
  --new-container sglwmc-813-ifa-repro \
  --device 14 \
  --rounds 3 \
  --measurement-blocks 10 \
  --warmup 20 \
  --iters 150 \
  --eager-iters 100 \
  --batch-size 162 \
  --output-dir "$result_dir"
```

The command runs three independent Python processes per stack.  Each process
captures one 48-record graph and reports ten blocks of 150 iterations.  It
writes the environment, image IDs, script SHA-256, per-run JSON, stderr,
pre-run `npu-smi` snapshots, and a Markdown summary under `result_dir`.
The output directory must be new or empty so stale rounds cannot be mixed into
the summary.

To remove ordinary CPU migration as a variable, repeat the matrix with the
same CPU set for both containers:

```bash
result_dir=/data/wzy/ifa_npugraph_matrix_pinned_$(date +%Y%m%d_%H%M%S)
bash ifa_npugraph_update_perf/run_container_matrix.sh \
  --old-container sglwmc-723 \
  --new-container sglwmc-813-ifa-repro \
  --device 14 \
  --rounds 3 \
  --measurement-blocks 10 \
  --warmup 20 \
  --iters 150 \
  --eager-iters 100 \
  --batch-size 162 \
  --cpu-set 242-259 \
  --output-dir "$result_dir"
```

Regenerate a summary without rerunning the NPU workload:

```bash
python3 ifa_npugraph_update_perf/summarize_results.py "$result_dir"
```

The repeated machine-204 results below used the legacy five-entry diagnostic
shape.  They remain evidence about CPU scheduling sensitivity, but they are
not a substitute for the `bs=162` matrix:

| CPU placement | Stack | Block median | Block mean | CV | Blocks above 6 ms |
| --- | --- | ---: | ---: | ---: | ---: |
| Unpinned | 0723 / CANN 9.0 | 4,502.000 us | 4,582.277 us | 8.31% | 0/30 |
| Unpinned | 0813 / CANN 9.1 | 4,536.601 us | 5,348.924 us | 25.09% | 11/30 |
| CPU 242--259 | 0723 / CANN 9.0 | 4,371.570 us | 4,324.555 us | 5.51% | 0/30 |
| CPU 242--259 | 0813 / CANN 9.1 | 4,285.497 us | 4,251.966 us | 4.20% | 0/30 |

The unpinned new-stack distribution was bimodal: a roughly 4.0--4.7 ms fast
path and a roughly 6.9--7.7 ms slow path.  With identical affinity, the slow
path disappeared and the new-stack mean was 1.68% lower than the old stack.
Therefore, do not decide pass/fail from one process-level p50 or attribute an
unpinned slow tail directly to CANN.  Compare all block medians, coefficient
of variation, slow-block count, and the affinity-controlled matrix in
`summary.md`.

If the runner exits with status 75, another NPU process was present.  Treat
that attempt as contaminated and rerun after the machine is idle.  A separate
host scheduler can still start after the idle check, so also inspect
`harness.log`, the per-run exit status, and the saved `npu-smi` snapshots.
