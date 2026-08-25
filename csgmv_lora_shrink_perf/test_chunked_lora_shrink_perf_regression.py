#!/usr/bin/env python3
"""Fast regression test for the SGLang csgmv LoRA-A shrink kernel.

The test uses the Llama-3.1-8B QKV LoRA shape that separates the 0723 and
0813 Ascend stacks without starting a model server.  On the measured NPU:

* 0723 / CANN 9.0: decode ~= 0.18 ms, prefill ~= 1.22 ms
* 0813 / CANN 9.1: decode ~= 24.95 ms, prefill ~= 374.92 ms

It can be run either as a normal unittest or through pytest:

    python3 test_chunked_lora_shrink_perf_regression.py -v
    python3 -m pytest -q -s test_chunked_lora_shrink_perf_regression.py

Environment variables:

    NPU_DEVICE_INDEX=8
    CSGMV_SHRINK_DECODE_MAX_MS=5
    CSGMV_SHRINK_PREFILL_MAX_MS=30
"""

from __future__ import annotations

import os
import statistics
import time
import unittest
from dataclasses import dataclass

import torch

try:
    from sglang.kernels.ops.gemm.chunked_sgmv_expand import (
        chunked_sgmv_lora_expand_forward,
    )
    from sglang.kernels.ops.gemm.chunked_sgmv_shrink import (
        chunked_sgmv_lora_shrink_forward,
    )
    from sglang.srt.lora.utils import LoRABatchInfo

    _IMPORT_ERROR = None
except Exception as error:  # pragma: no cover - used only for a clean skip
    _IMPORT_ERROR = error


INPUT_DIM = 4096
OUTPUT_DIM = 6144
RANK = 256
NUM_SLICES = 3
NUM_ADAPTERS = 3
CHUNK_SIZE = 16


def _sync() -> None:
    torch.npu.synchronize()


def _make_batch_info(device: torch.device, tokens: int) -> "LoRABatchInfo":
    """Lay tokens out adapter-major, split into csgmv 16-token segments."""
    base, remainder = divmod(tokens, NUM_ADAPTERS)
    per_adapter = [base + (index < remainder) for index in range(NUM_ADAPTERS)]

    segment_lengths: list[int] = []
    weight_indices: list[int] = []
    for adapter, count in enumerate(per_adapter):
        while count:
            length = min(CHUNK_SIZE, count)
            segment_lengths.append(length)
            weight_indices.append(adapter)
            count -= length

    indptr = [0]
    for length in segment_lengths:
        indptr.append(indptr[-1] + length)

    return LoRABatchInfo(
        use_cuda_graph=False,
        bs=tokens,
        num_segments=len(segment_lengths),
        seg_indptr=torch.tensor(indptr, dtype=torch.int32, device=device),
        weight_indices=torch.tensor(weight_indices, dtype=torch.int32, device=device),
        lora_ranks=torch.full((NUM_ADAPTERS,), RANK, dtype=torch.int32, device=device),
        scalings=torch.ones(NUM_ADAPTERS, dtype=torch.float32, device=device),
        max_len=CHUNK_SIZE,
        seg_lens=torch.tensor(segment_lengths, dtype=torch.int32, device=device),
        permutation=torch.arange(tokens, dtype=torch.int32, device=device),
    )


@dataclass
class _Case:
    label: str
    tokens: int
    x: torch.Tensor
    weights_a: torch.Tensor
    intermediate: torch.Tensor
    weights_b: torch.Tensor
    base_output: torch.Tensor
    slice_offsets: torch.Tensor
    batch_info: "LoRABatchInfo"

    def shrink(self) -> torch.Tensor:
        # This wrapper launches the operator under test:
        # sglang...chunked_sgmv_shrink._chunked_lora_shrink_kernel
        return chunked_sgmv_lora_shrink_forward(
            self.x, self.weights_a, self.batch_info, num_slices=NUM_SLICES
        )

    def expand(self) -> torch.Tensor:
        # Same-stack control operator. It did not regress on 0813.
        return chunked_sgmv_lora_expand_forward(
            self.intermediate,
            self.weights_b,
            self.batch_info,
            self.slice_offsets,
            OUTPUT_DIM // NUM_SLICES,
            self.base_output,
        )


def _make_case(device: torch.device, label: str, tokens: int) -> _Case:
    dtype = torch.bfloat16
    return _Case(
        label=label,
        tokens=tokens,
        x=torch.randn(tokens, INPUT_DIM, dtype=dtype, device=device),
        weights_a=torch.randn(
            NUM_ADAPTERS,
            NUM_SLICES * RANK,
            INPUT_DIM,
            dtype=dtype,
            device=device,
        ),
        intermediate=torch.randn(tokens, NUM_SLICES * RANK, dtype=dtype, device=device),
        weights_b=torch.randn(
            NUM_ADAPTERS, OUTPUT_DIM, RANK, dtype=dtype, device=device
        ),
        base_output=torch.zeros(tokens, OUTPUT_DIM, dtype=dtype, device=device),
        slice_offsets=torch.tensor(
            [0, 2048, 4096, 6144], dtype=torch.int32, device=device
        ),
        batch_info=_make_batch_info(device, tokens),
    )


def _median_ms(function, *, warmup: int = 2, iterations: int = 5) -> float:
    # Warmup contains compilation and is deliberately excluded from timing.
    for _ in range(warmup):
        function()
    _sync()

    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        function()
        _sync()
        samples.append((time.perf_counter() - started) * 1000)
    return statistics.median(samples)


@unittest.skipIf(
    _IMPORT_ERROR is not None,
    f"SGLang csgmv kernel is unavailable: {_IMPORT_ERROR}",
)
class TestChunkedLoraShrinkPerformance(unittest.TestCase):
    """Operator-level correctness and performance regression tests."""

    @classmethod
    def setUpClass(cls) -> None:
        if not hasattr(torch, "npu") or not torch.npu.is_available():
            raise unittest.SkipTest("this regression test requires an Ascend NPU")

        device_index = int(os.environ.get("NPU_DEVICE_INDEX", "8"))
        cls.device = torch.device(f"npu:{device_index}")
        torch.npu.set_device(cls.device)
        torch.manual_seed(42)

    def test_00_shrink_numerical_correctness(self) -> None:
        """Check that the timed kernel still computes LoRA-A X @ A.T."""
        case = _make_case(self.device, "decode_bs3_qkv", tokens=3)
        actual = case.shrink()
        _sync()

        references = []
        for adapter in range(NUM_ADAPTERS):
            references.append(
                torch.matmul(
                    case.x[adapter : adapter + 1].float(),
                    case.weights_a[adapter].float().transpose(0, 1),
                ).to(torch.bfloat16)
            )
        expected = torch.cat(references, dim=0)

        torch.testing.assert_close(actual, expected, rtol=0.05, atol=1.0)

    def test_10_shrink_has_no_0813_performance_regression(self) -> None:
        """Fail quickly on the 0813 shrink regression, with expand as control."""
        limits = {
            "decode_bs3_qkv": float(os.environ.get("CSGMV_SHRINK_DECODE_MAX_MS", "5")),
            "prefill_384_qkv": float(
                os.environ.get("CSGMV_SHRINK_PREFILL_MAX_MS", "30")
            ),
        }
        cases = (
            _make_case(self.device, "decode_bs3_qkv", tokens=3),
            _make_case(self.device, "prefill_384_qkv", tokens=384),
        )

        results = []
        violations = []
        for case in cases:
            shrink_ms = _median_ms(case.shrink)
            expand_ms = _median_ms(case.expand)
            limit_ms = limits[case.label]
            ratio = shrink_ms / expand_ms
            results.append(
                f"{case.label}: shrink={shrink_ms:.3f} ms, "
                f"expand(control)={expand_ms:.3f} ms, "
                f"ratio={ratio:.3f}, shrink_limit={limit_ms:.3f} ms"
            )

            # On 0723 shrink is far below both limits and below 0.5x expand.
            # On 0813 it violates both conditions by a wide margin.
            if shrink_ms > limit_ms:
                violations.append(
                    f"{case.label} shrink {shrink_ms:.3f} ms > {limit_ms:.3f} ms"
                )
            if ratio > 0.5:
                violations.append(
                    f"{case.label} shrink/expand ratio {ratio:.3f} > 0.500"
                )

        print("\n" + "\n".join(results), flush=True)
        self.assertFalse(
            violations,
            "0813 csgmv LoRA-A shrink performance regression detected in "
            "sglang.kernels.ops.gemm.chunked_sgmv_shrink."
            "_chunked_lora_shrink_kernel:\n"
            + "\n".join(violations)
            + "\nMeasurements:\n"
            + "\n".join(results),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
