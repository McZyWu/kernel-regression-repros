# IFA pooled K/V offset regression

This directory preserves the scripts and aggregate evidence from the
2026-09-10 Qwen3-30B-A3B W8A8 investigation. With 48 IFA records backed by
different slices of one K/V pool, CANN 9.1 performs 48 tilings per sequence
length update. The tested CANN 9.0 stack performs one. Rebasing each contiguous
K/V view to a zero-offset storage alias restores one tiling per update on 9.1,
without copying K/V or skipping any record's sequence-length update.

The earlier diagnostic in the parent directory allocates independent K/V
tensors and does not preserve this trigger: its view offsets are all zero.
This snapshot adds shared K/V storage and changing sequence lengths.

## Files

| File | Purpose |
| --- | --- |
| [`ifa_worker_probe.py`](ifa_worker_probe.py) | The validated standalone NPUGraph update probe, including independent, pooled, and rebased K/V modes |
| [`run_packed_probe.sh`](run_packed_probe.sh) | Four runs: old/new stack × pooled/independent K/V, with progressing sequence lengths |
| [`run_rebase_probe.sh`](run_rebase_probe.sh) | New-stack zero-copy rebasing control |
| [`trace_summary.json`](trace_summary.json) | Original aggregate counts, durations, and remote trace paths for all eleven runs, including the thread and ragged-length controls |
| [`summarize_traces.py`](summarize_traces.py) | Rebuild the aggregate from the microbenchmark's exported `trace_view.json` files |
| [`SOURCE_SHA256SUMS.txt`](SOURCE_SHA256SUMS.txt) | Checksums of the five original experiment files |

The five experiment files are preserved verbatim. The absolute paths in the
saved JSON identify the original artifacts; raw, large profiler traces are
not included in this repository. This snapshot is separate from the older
diagnostic so the recorded experiment remains reviewable without changing it.

## Recorded results

Each progressing-length run profiles four updates: one warmup update plus
three measured updates. Each update supplies one CPU dictionary that
torch-npu expands across all 48 captured records. KV lengths are 3501, 3502,
3503, and 3504. All runs use `bs=162`, four query tokens per request, BF16,
TND query shape `(648,16,128)`, page size 128, and a `(162,28)` block table.

| Label in `trace_summary.json` | K/V representation | Workspace calls | Tiling calls |
| --- | --- | ---: | ---: |
| `723-fia-packed0-progress` | Independent storage, zero offsets | 192 | 4 |
| `723-fia-packed1-progress` | Shared pool, different offsets | 192 | 4 |
| `813-packed0-progress` | Independent storage, zero offsets | 192 | 4 |
| `813-packed1-progress` | Shared pool, different offsets | 192 | **192** |
| `813-packed2-progress` | Same pool, zero-offset aliases | 192 | **4** |

In the 9.1 profile, mean Workspace duration was 72.402 us with pooled offsets,
versus 13.326 us after rebasing. Tiling is nested inside Workspace; do not add
their durations together. These are profiled API timings, not whole-model
TPOT or an absolute pass/fail threshold.

The six other entries are controls with fixed updated lengths: fresh versus
pooled Python workers on both stacks, and ragged KV lengths on both stacks.
Each records 192 Workspace calls and one tiling in the original experiment.

## Run the saved matrix

The saved launchers target the original Linux environment:

- Containers: `sglwmc-723-fia` and `sglwmc-813`.
- Physical device 14; `npu-smi` reports it under NPU 7 / chip 0.
- Identical CPU affinity: `560-577`.
- A host directory under `/home` that is mounted at the same path in both
  containers.
- PyTorch and torch-npu, with a compatible Ascend stack in each container.
  No model weights or SGLang import are required.

Use a new shared output directory on each invocation. The original launchers
have a fixed `root` and fixed labels and do not reject old result directories.
The following stages copies and adjusts only their output root:

```bash
# From the repository root, on the Linux NPU host.
src="$PWD/ifa_npugraph_update_perf/pooled_kv_offset"
run=$(mktemp -d /home/wzy/ifa_offset_matrix_XXXXXXXX)
cp "$src/ifa_worker_probe.py" "$src/run_packed_probe.sh" \
   "$src/run_rebase_probe.sh" "$src/summarize_traces.py" "$run/"
sed -i "s|^root=.*|root=$run|" \
  "$run/run_packed_probe.sh" "$run/run_rebase_probe.sh"

bash "$run/run_packed_probe.sh"
bash "$run/run_rebase_probe.sh"
python3 "$run/summarize_traces.py" "$run"
```

Run the two launchers sequentially. Each checks that NPU 7 is idle and exits
with status 75 if it is occupied. They do not stop external processes. If the
host differs, adapt the container names, CPU set, physical device, and matching
idle check in the staged copies before running them.

For one short profile inside a configured Ascend environment:

```bash
IFA_KV_PACKED=1 IFA_SEQ_PROGRESS=1 IFA_WORKER_MODE=pool \
python3 ifa_worker_probe.py \
  --device 14 --records 48 --batch-size 162 --tokens-per-request 4 \
  --seq-len 3500 --warmup 1 --iters 3 --measurement-blocks 1 \
  --eager-iters 1 --update-only --profile-dir /path/to/new/profile
```

Use `IFA_KV_PACKED=0` for the independent-storage control or `=2` for the
zero-copy rebasing control. Set other diagnostic environment variables
explicitly when adapting the run; the original launchers assume a clean shell.

| Environment variable | Values |
| --- | --- |
| `IFA_KV_PACKED` | `0` (default): independent storage; `1`: pooled views; `2`: pooled zero-offset aliases |
| `IFA_SEQ_PROGRESS` | `1`: increment KV length for each update; otherwise keep the updated length fixed |
| `IFA_WORKER_MODE` | `pool` (default): reuse one worker; `fresh`: start and join a thread for each update |
| `IFA_SEQ_PATTERN` | `ragged`: fixed lengths `seq_len + 1 + i % 3`; otherwise uniform. Progressing mode replaces this list with uniform, increasing lengths. |

## Interpretation and boundaries

- Use the short matrix parameters above. Increasing the update count in
  progressing mode may exceed the allocated KV/page-table capacity; allocation
  in this archived probe is based on `seq_len + tokens_per_request`, not on the
  number of updates.
- Keep `--update-only`. Standalone replay of the extracted functional TND
  subgraph was observed to leave a pending device task and is not a valid
  performance comparison.
- The `fresh` worker control joins its thread before `submit` returns. It is
  suitable for the archived update-only comparison, not replay-overlap timing.
- The probe contains uninitialized layer tensors in some controls. Use it to
  compare host tiling/cache behavior, not to validate model accuracy.
- `summarize_traces.py` counts complete (`ph="X"`) events in these standalone
  traces. Full-model CANN API accounting needs the profiler database and must
  not be inferred from this JSON alone.
- The rebased view keeps the effective data pointer and still updates all 48
  records. It is a diagnostic control, not a general instruction to remove
  offsets from every CANN cache key.
- This evidence does not assert that the original registered serving case
  passes its unprofiled `<10 ms` TPOT requirement. Whole-model token/logit and
  EAGLE acceptance validation remain separate acceptance work.

The measurements were collected on 2026-09-10. Importing this snapshot into
the repository does not represent a new NPU benchmark run.
