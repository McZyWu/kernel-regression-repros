#!/usr/bin/env bash
set -euo pipefail
root=/home/wzy/qwen3_cann91_diagnosis_20260910
npu-smi info >"$root/npu_before_packed.txt"
grep -q 'No running processes found in NPU 7' "$root/npu_before_packed.txt" || exit 75
for stack in 813 723-fia; do
  for packed in 1 0; do
    label="$stack-packed$packed-progress"
    mkdir -p "$root/$label"
    docker exec -e IFA_KV_PACKED="$packed" -e IFA_SEQ_PROGRESS=1 -e IFA_WORKER_MODE=pool "sglwmc-$stack" bash -lc "timeout 180 taskset -c 560-577 python3 $root/ifa_worker_probe.py --device 14 --warmup 1 --iters 3 --measurement-blocks 1 --eager-iters 1 --profile-dir $root/$label/profile" >"$root/$label/stdout.log" 2>"$root/$label/stderr.log"
    echo "done $label"
  done
done
npu-smi info >"$root/npu_after_packed.txt"
