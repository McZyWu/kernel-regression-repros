#!/usr/bin/env bash
# Run on every node. The four invocations must start at approximately the same
# time and use the same CASE_NAME, PHASE, source tree, and arguments.

set -euo pipefail

: "${NODE_RANK:?set NODE_RANK to 0, 1, 2, or 3}"
: "${CASE_NAME:?set CASE_NAME, e.g. cann90-ta321-main5f216}"
: "${PHASE:?set PHASE to cold or warm}"
: "${SGLANG_SOURCE:?set SGLANG_SOURCE to the identical source tree on every node}"

NNODES="${NNODES:-4}"
NPROC_PER_NODE="${NPROC_PER_NODE:-16}"
MASTER_ADDR="${MASTER_ADDR:-192.168.25.209}"
MASTER_PORT="${MASTER_PORT:-30221}"
CACHE_ROOT="${CACHE_ROOT:-/tmp/k3-ta-ab/${CASE_NAME}}"
RESULT_ROOT="${RESULT_ROOT:-/home/wzy/k3-ta-ab-results}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPRO_SCRIPT="${REPRO_SCRIPT:-${SCRIPT_DIR}/k3_ta_operator_collective_repro.py}"
LAUNCH_PATH="${LAUNCH_PATH:-forced-autotune}"
OP="${OP:-both}"
LAYOUT="${LAYOUT:-varlen}"
CACHE_LAYOUT="${CACHE_LAYOUT:-per-node}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-1200}"
STACK_DUMP_SECONDS="${STACK_DUMP_SECONDS:-180}"

if [[ "${PHASE}" != "cold" && "${PHASE}" != "warm" ]]; then
    echo "PHASE must be cold or warm, got ${PHASE}" >&2
    exit 2
fi
if [[ ! -f "${REPRO_SCRIPT}" ]]; then
    echo "reproducer not found: ${REPRO_SCRIPT}" >&2
    exit 2
fi

NODE_CACHE_ROOT="${CACHE_ROOT}/node${NODE_RANK}"
if [[ "${PHASE}" == "cold" ]]; then
    if [[ -e "${NODE_CACHE_ROOT}" ]]; then
        echo "refusing to reuse cold cache: ${NODE_CACHE_ROOT}" >&2
        exit 3
    fi
elif [[ ! -d "${NODE_CACHE_ROOT}" ]]; then
    echo "warm phase requires cold-phase cache: ${NODE_CACHE_ROOT}" >&2
    exit 3
fi

RUN_DIR="${RESULT_ROOT}/${CASE_NAME}/${PHASE}/node${NODE_RANK}"
mkdir -p "${RUN_DIR}"
cd "${SGLANG_SOURCE}"

# CANN 9.0's ATB environment script reads ZSH_VERSION without a default.
# Keep strict nounset for this runner, but not while sourcing vendor scripts.
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u

unset ASCEND_LAUNCH_BLOCKING
unset ASCEND_CUSTOM_OPP_PATH
unset ENABLE_PROFILING
unset SGLANG_K3_TRACE_STATE_FILE
unset SGLANG_K3_TRACE_HIDDEN_FILE
unset ASCEND_RT_VISIBLE_DEVICES
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

export PYTHONPATH="${SGLANG_SOURCE}/python"
export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=1
export HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-enp196s0f0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-enp196s0f0}"
export STREAMS_PER_DEVICE="${STREAMS_PER_DEVICE:-32}"
export TRITON_PRINT_AUTOTUNING="${TRITON_PRINT_AUTOTUNING:-1}"

{
    echo "case=${CASE_NAME}"
    echo "phase=${PHASE}"
    echo "node_rank=${NODE_RANK}"
    echo "hostname=$(hostname)"
    echo "source=${SGLANG_SOURCE}"
    echo "commit=$(git -C "${SGLANG_SOURCE}" rev-parse HEAD 2>/dev/null || true)"
    echo "git_status_begin"
    git -C "${SGLANG_SOURCE}" status --porcelain 2>/dev/null || true
    echo "git_status_end"
    sha256sum \
        "${SGLANG_SOURCE}/python/sglang/kernels/ops/attention/fla/cumsum.py" \
        "${SGLANG_SOURCE}/python/sglang/kernels/ops/attention/fla/kda.py"
    python3 --version
    for version_file in \
        /usr/local/Ascend/ascend-toolkit/latest/version.cfg \
        /usr/local/Ascend/ascend-toolkit/latest/*-version.info; do
        if [[ -f "${version_file}" ]]; then
            echo "ascend_version_file=${version_file}"
            cat "${version_file}"
        fi
    done
    python3 - <<'PY'
import importlib.metadata as metadata
import torch
import triton

print("torch_import_version=", torch.__version__, sep="")
print("triton_import_version=", triton.__version__, sep="")
print("triton_import_file=", triton.__file__, sep="")
for name in ("torch-npu", "triton-ascend", "sglang", "sgl-kernel-npu"):
    try:
        print(f"{name}={metadata.version(name)}")
    except metadata.PackageNotFoundError:
        print(f"{name}=NOT_INSTALLED")
PY
    env | sort | grep -E '^(ASCEND_|HCCL_|SGLANG_|TORCH|TRITON_|PYTHONPATH=|STREAMS_PER_DEVICE=)' || true
} > "${RUN_DIR}/environment.log" 2>&1

EXTRA_ARGS=()
if [[ -n "${INJECT_DELAY_GLOBAL_RANK:-}" ]]; then
    EXTRA_ARGS+=(--inject-delay-global-rank "${INJECT_DELAY_GLOBAL_RANK}")
fi
if [[ -n "${INJECT_DELAY_NODE_RANK:-}" ]]; then
    EXTRA_ARGS+=(--inject-delay-node-rank "${INJECT_DELAY_NODE_RANK}")
fi
if [[ -n "${INJECT_DELAY_SECONDS:-}" ]]; then
    EXTRA_ARGS+=(--inject-delay-seconds "${INJECT_DELAY_SECONDS}")
fi

set +e
timeout --signal=TERM --kill-after=30s "${TIMEOUT_SECONDS}s" \
    python3 -m torch.distributed.run \
    --nnodes "${NNODES}" \
    --nproc-per-node "${NPROC_PER_NODE}" \
    --node-rank "${NODE_RANK}" \
    --master-addr "${MASTER_ADDR}" \
    --master-port "${MASTER_PORT}" \
    "${REPRO_SCRIPT}" \
    --case "${CASE_NAME}" \
    --phase "${PHASE}" \
    --launch-path "${LAUNCH_PATH}" \
    --op "${OP}" \
    --layout "${LAYOUT}" \
    --cache-root "${CACHE_ROOT}" \
    --cache-layout "${CACHE_LAYOUT}" \
    --result-dir "${RUN_DIR}" \
    --dump-stack-after-seconds "${STACK_DUMP_SECONDS}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "${RUN_DIR}/torchrun.log"
launcher_rc="${PIPESTATUS[0]}"
set -e

echo "${launcher_rc}" > "${RUN_DIR}/exit_code.txt"
if [[ "${launcher_rc}" -eq 124 ]]; then
    echo "K3 TA reproducer timed out; inspect repeated faulthandler stacks in ${RUN_DIR}/torchrun.log" >&2
elif [[ "${launcher_rc}" -ne 0 ]]; then
    echo "K3 TA reproducer failed with exit code ${launcher_rc}" >&2
fi
exit "${launcher_rc}"
