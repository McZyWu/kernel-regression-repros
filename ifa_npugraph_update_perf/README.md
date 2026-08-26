# IFA NPU Graph update performance regression

This reproducer isolates the host-side NPU Graph update regression observed in
the Qwen3-30B-A3B W8A8 EAGLE3 performance case after upgrading the native
Ascend stack from CANN 9.0 to CANN 9.1.

It requires only PyTorch, `torch-npu`, and one Ascend NPU.  It does not import
SGLang or load model weights.

## What it reproduces

Each target-verification decode step in the original model graph updates 48
`npu_fused_infer_attention_score.out` records.  The reproducer uses the same
core shape:

```text
records=48
query tokens=5
query heads=16
KV heads=2
head dimension=128
KV sequence length=3500 -> 3501
KV page size=128
dtype=bfloat16
```

Every record receives distinct query, key, value, block-table, and output
addresses.  The workspace can be shared because the calls are issued
serially.  As in the production graph runner, `NPUGraph.update` runs on a CPU
worker while the main thread submits `NPUGraph.replay`.

The eager IFA path is measured as a control.  In the original comparison,
eager submit and synchronized latency stayed flat while the 48-record graph
update became slower:

| Stack | 48-record update p50 | Eager submit p50 | Eager sync-total p50 |
| --- | ---: | ---: | ---: |
| 0723 / CANN 9.0 | 4,158.525 us | 44.880 us | 112.777 us |
| 0813 / CANN 9.1 | 4,630.838 us | 44.726 us | 113.612 us |
| Difference | **+472.313 us (+11.36%)** | -0.154 us | +0.835 us |

This localizes the standalone regression to NPU Graph record update rather
than IFA device computation or the ordinary eager launch path.  Whole-model
profiling shows a larger amplification because it also includes workspace
query/tiling, graph task management, queueing, and 51 IFA records across the
target, draft, and draft-extend phases.

## Run

The default command runs the production-shaped 48-record case and prints JSON:

```bash
NPU_DEVICE_INDEX=0 \
python3 ifa_npugraph_update_perf/ifa_npugraph_update_repro.py
```

The important fields are:

- `update`: time spent in `NPUGraph.update` on the CPU worker;
- `replay_submit`: main-thread `NPUGraph.replay` submission time;
- `overlap_total`: wall time from starting the update worker until both update
  and replay submission have returned;
- `eager_submit` and `eager_sync_total`: same-stack controls for a single IFA
  call.

Absolute host latency depends on the server and its load.  The default is
therefore report-only.  On the original machine, this threshold separated the
two stacks:

```bash
NPU_DEVICE_INDEX=14 \
python3 ifa_npugraph_update_perf/ifa_npugraph_update_repro.py \
  --warmup 20 \
  --iters 150 \
  --max-update-p50-us 4400
```

Use a threshold established from repeated healthy-stack measurements when
running on another machine.

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
