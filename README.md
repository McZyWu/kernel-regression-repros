# Ascend kernel regression reproducers

Small, operator-level reproducers for regressions observed while running
SGLang on Ascend NPUs.  These tests avoid model weights, HTTP serving, and
multi-hour profiling runs so a stack change can be evaluated quickly.

## Reproducers

| Directory | Target operator | Regression |
| --- | --- | --- |
| `csgmv_lora_shrink_perf` | `sglang.kernels.ops.gemm.chunked_sgmv_shrink._chunked_lora_shrink_kernel` | LoRA-A shrink became tens to hundreds of times slower on the 0813 stack |
| `recompute_multi_rank_hang` | `sgl_kernel_npu.fla.wy_fast.recompute_w_u_fwd_npu` | A Qwen3-Next TP4/DP1 process stopped inside the operator after prior model execution |

## csgmv LoRA-A shrink performance regression

The test uses the Llama-3.1-8B QKV LoRA shape:

```text
input_dim=4096
output_dim=6144
rank=256
num_slices=3
num_adapters=3
chunk_size=16
```

It checks numerical correctness, then measures decode (3 tokens) and prefill
(384 tokens).  `_chunked_lora_expand_kernel` is measured as a same-stack
control.  Triton compilation is excluded from the timed samples.

Representative measurements on the same Ascend NPU:

| Stack | Decode shrink | Prefill shrink | Result |
| --- | ---: | ---: | --- |
| 0723 / CANN 9.0 / Triton-Ascend 3.2.1.dev20260530 | 0.278 ms | 1.314 ms | PASS |
| 0813 / CANN 9.1 / Triton-Ascend 3.2.2 | 28.327 ms | 291.265 ms | FAIL |

The 0813 expand control remained healthy at 2.901 ms for decode and 65.501 ms
for prefill.  This localizes the regression to the csgmv LoRA-A shrink path,
not general device load or the LoRA-B expand path.

Run it in an Ascend SGLang environment:

```bash
NPU_DEVICE_INDEX=0 \
python3 csgmv_lora_shrink_perf/test_chunked_lora_shrink_perf_regression.py -v
```

The default regression limits are intentionally far above the healthy 0723
values and can be overridden for another device:

```bash
CSGMV_SHRINK_DECODE_MAX_MS=5 \
CSGMV_SHRINK_PREFILL_MAX_MS=30 \
NPU_DEVICE_INDEX=0 \
python3 csgmv_lora_shrink_perf/test_chunked_lora_shrink_perf_regression.py -v
```

## `recompute_w_u_fwd_npu` multi-rank hang replay

This opt-in test launches four worker processes, loads a captured tensor set
for each TP rank, synchronizes the first launch with a file barrier, and uses
a watchdog to capture Python stacks, process information, and `npu-smi`
output before terminating a stuck worker group.

Place the four local fixtures in
`recompute_multi_rank_hang/fixtures/`, or pass an absolute path pattern that
contains `{rank}`:

```bash
ASCEND_LAUNCH_BLOCKING=1 \
python3 recompute_multi_rank_hang/test_recompute_w_u_fwd_hang.py \
  --input '/path/to/fixtures/gdn_recompute_call160_npu{rank}_20260824.pt' \
  --workers 4 \
  --repeats 100 \
  --timeout 60 \
  --ready-timeout 120 \
  --trace-every 1 \
  --progress-every 1 \
  --log-dir /tmp/recompute-w-u-fwd-hang
```

Pytest collection is disabled by default because the replay requires four
NPUs and external fixtures.  Enable it explicitly:

```bash
RUN_RECOMPUTE_W_U_FWD_HANG_TEST=1 \
RECOMPUTE_W_U_FWD_CAPTURE='/path/to/fixtures/gdn_recompute_call160_npu{rank}_20260824.pt' \
python3 -m pytest -q -s \
  recompute_multi_rank_hang/test_recompute_w_u_fwd_hang.py
```

### Reproduction boundary

The original full-model stop is real and the blocked stack identifies
`recompute_w_u_fwd_npu`.  However, the four captured tensor sets alone did
not reproduce the hang in a clean process: a clean four-rank replay completed
100 calls per rank in about 0.45 seconds.  The missing trigger is prior
in-process state, such as runtime or compilation history, allocator/device
address state, or model-layer timing.  The replay preserves the exact inputs,
coordinated launch, and timeout diagnostics, but should not be described as a
deterministic standalone reproducer until that state dependency is isolated.

## Requirements

- Linux with Ascend NPUs and a matching CANN runtime
- PyTorch with `torch-npu`
- SGLang for the csgmv LoRA shrink test
- `sgl-kernel-npu` and Triton-Ascend for the recompute replay
- `pytest` only when invoking the tests through pytest

No model weights are required.  Captured `.pt` fixtures are intentionally not
tracked by this public repository.
