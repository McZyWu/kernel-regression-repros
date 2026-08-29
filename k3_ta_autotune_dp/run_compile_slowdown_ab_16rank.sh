#!/usr/bin/env bash
# Reproduce the Kimi-K3 cold-cache Triton Ascend compile slowdown on one
# 16-NPU host while keeping CANN, Python, torch-npu, source and inputs fixed.

set -euo pipefail

: "${SGLANG_SOURCE:?set SGLANG_SOURCE to the one checkout shared by both TA venvs}"
: "${TA321_VENV:?set TA321_VENV to a Python venv containing TA 3.2.1}"
: "${TA322_VENV:?set TA322_VENV to a Python venv containing the TA 3.2.2 build under test}"

RUNS="${RUNS:-1}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
CASE_PREFIX="${CASE_PREFIX:-k3-compile-slowdown-ab}"
RESULT_ROOT="${RESULT_ROOT:-/home/wzy/k3-ta-compile-ab-results}"
CACHE_ROOT="${CACHE_ROOT:-/tmp/k3-ta-compile-ab}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-30400}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-600}"
STACK_DUMP_SECONDS="${STACK_DUMP_SECONDS:-120}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN_CASE="${SCRIPT_DIR}/run_case.sh"
SUMMARIZER="${SCRIPT_DIR}/summarize_ab.py"

if ! [[ "${RUNS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "RUNS must be a positive integer, got ${RUNS}" >&2
    exit 2
fi
for path in \
    "${TA321_VENV}/bin/python3" \
    "${TA322_VENV}/bin/python3" \
    "${RUN_CASE}" \
    "${SUMMARIZER}" \
    "${SGLANG_SOURCE}/python/sglang/kernels/ops/attention/fla/cumsum.py" \
    "${SGLANG_SOURCE}/python/sglang/kernels/ops/attention/fla/kda.py"; do
    if [[ ! -e "${path}" ]]; then
        echo "required path does not exist: ${path}" >&2
        exit 2
    fi
done

mkdir -p "${RESULT_ROOT}/ab-summaries/${RUN_ID}"

print_version() {
    local label="$1"
    local venv="$2"
    echo "${label}:"
    "${venv}/bin/python3" - <<'PY'
import importlib.metadata as metadata
import sys

print("python=", sys.version.replace("\n", " "), sep="")
for name in ("torch", "torch-npu", "triton-ascend"):
    try:
        print(f"{name}={metadata.version(name)}")
    except metadata.PackageNotFoundError:
        print(f"{name}=NOT_INSTALLED")
PY
}

environment_fingerprint() {
    local venv="$1"
    "${venv}/bin/python3" - <<'PY'
import importlib.metadata as metadata
import sys

print(
    f"python={sys.version_info.major}.{sys.version_info.minor};"
    f"torch={metadata.version('torch')};"
    f"torch-npu={metadata.version('torch-npu')}"
)
PY
}

run_one() {
    local label="$1"
    local venv="$2"
    local round="$3"
    local port="$4"
    local case_name="${CASE_PREFIX}-${RUN_ID}-${label}-r${round}"
    local case_cache="${CACHE_ROOT}/${case_name}"

    if [[ -e "${case_cache}" ]]; then
        echo "refusing to reuse cold cache: ${case_cache}" >&2
        exit 3
    fi
    echo "Running ${case_name} on port ${port}"
    (
        source "${venv}/bin/activate"
        hash -r
        env \
            NODE_RANK=0 \
            NNODES=1 \
            NPROC_PER_NODE=16 \
            MASTER_ADDR=127.0.0.1 \
            MASTER_PORT="${port}" \
            CASE_NAME="${case_name}" \
            PHASE=cold \
            SGLANG_SOURCE="${SGLANG_SOURCE}" \
            CACHE_ROOT="${case_cache}" \
            RESULT_ROOT="${RESULT_ROOT}" \
            CACHE_LAYOUT=per-node \
            LAUNCH_PATH=forced-autotune \
            OP=both \
            LAYOUT=varlen \
            TIMEOUT_SECONDS="${TIMEOUT_SECONDS}" \
            STACK_DUMP_SECONDS="${STACK_DUMP_SECONDS}" \
            bash "${RUN_CASE}"
    )
}

summarize_round() {
    local round="$1"
    local left="${RESULT_ROOT}/${CASE_PREFIX}-${RUN_ID}-ta321-r${round}/cold"
    local right="${RESULT_ROOT}/${CASE_PREFIX}-${RUN_ID}-ta322-r${round}/cold"
    "${TA321_VENV}/bin/python3" "${SUMMARIZER}" \
        --expected-world-size 16 \
        "${left}" "${right}" \
        | tee "${RESULT_ROOT}/ab-summaries/${RUN_ID}/round${round}.json"
}

echo "run_id=${RUN_ID}"
echo "source_commit=$(git -C "${SGLANG_SOURCE}" rev-parse HEAD 2>/dev/null || true)"
sha256sum \
    "${SGLANG_SOURCE}/python/sglang/kernels/ops/attention/fla/cumsum.py" \
    "${SGLANG_SOURCE}/python/sglang/kernels/ops/attention/fla/kda.py"
print_version ta321 "${TA321_VENV}"
print_version ta322 "${TA322_VENV}"
ta321_fingerprint="$(environment_fingerprint "${TA321_VENV}")"
ta322_fingerprint="$(environment_fingerprint "${TA322_VENV}")"
if [[ "${ta321_fingerprint}" != "${ta322_fingerprint}" ]]; then
    echo "refusing a non-controlled A/B: Python/torch environments differ" >&2
    echo "ta321: ${ta321_fingerprint}" >&2
    echo "ta322: ${ta322_fingerprint}" >&2
    exit 2
fi
echo "controlled_environment=${ta321_fingerprint}"

for round in $(seq 1 "${RUNS}"); do
    first_port=$((MASTER_PORT_BASE + (round - 1) * 2))
    second_port=$((first_port + 1))
    # Alternate order to reduce a systematic first-run/thermal bias.
    if (( round % 2 == 1 )); then
        run_one ta321 "${TA321_VENV}" "${round}" "${first_port}"
        run_one ta322 "${TA322_VENV}" "${round}" "${second_port}"
    else
        run_one ta322 "${TA322_VENV}" "${round}" "${first_port}"
        run_one ta321 "${TA321_VENV}" "${round}" "${second_port}"
    fi
    summarize_round "${round}"
done

echo "Results: ${RESULT_ROOT}"
echo "A/B summaries: ${RESULT_ROOT}/ab-summaries/${RUN_ID}"
