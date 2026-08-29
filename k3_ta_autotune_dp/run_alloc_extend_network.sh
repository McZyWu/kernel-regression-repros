#!/usr/bin/env bash
# Launch the alloc_extend single-operator reproducer on one or more nodes.
# Cold runs refuse any pre-existing cache rather than deleting it.

set -euo pipefail

: "${NODE_RANK:?set NODE_RANK to a value in [0, NNODES)}"
: "${CASE_NAME:?set a unique CASE_NAME}"
: "${PHASE:?set PHASE to cold or warm}"
: "${VARIANT:?set VARIANT to exact-dynamic or static-bound}"
: "${SGLANG_SOURCE:?set SGLANG_SOURCE to the identical checkout on all nodes}"

NNODES="${NNODES:-4}"
NPROC_PER_NODE="${NPROC_PER_NODE:-16}"
MASTER_ADDR="${MASTER_ADDR:-192.168.25.209}"
MASTER_PORT="${MASTER_PORT:-30231}"
CACHE_ROOT="${CACHE_ROOT:-/tmp/alloc-extend-network/${CASE_NAME}}"
RESULT_ROOT="${RESULT_ROOT:-/home/wzy/alloc-extend-network-results}"
CACHE_LAYOUT="${CACHE_LAYOUT:-per-node}"
BATCH_SIZE="${BATCH_SIZE:-1}"
PAGE_SIZE="${PAGE_SIZE:-128}"
PREFIX_TOKENS="${PREFIX_TOKENS:-0}"
EXTEND_TOKENS="${EXTEND_TOKENS:-70}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-900}"
STACK_DUMP_SECONDS="${STACK_DUMP_SECONDS:-90}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPRO_SCRIPT="${REPRO_SCRIPT:-${SCRIPT_DIR}/alloc_extend_network_repro.py}"

if [[ "${NODE_RANK}" -lt 0 || "${NODE_RANK}" -ge "${NNODES}" ]]; then
    echo "NODE_RANK=${NODE_RANK} is outside [0, ${NNODES})" >&2
    exit 2
fi
if [[ "${PHASE}" != "cold" && "${PHASE}" != "warm" ]]; then
    echo "PHASE must be cold or warm" >&2
    exit 2
fi
if [[ "${VARIANT}" != "exact-dynamic" && \
      "${VARIANT}" != "npu-production" && \
      "${VARIANT}" != "static-bound" ]]; then
    echo "VARIANT must be exact-dynamic, npu-production, or static-bound" >&2
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
    echo "warm phase requires the matching cold cache: ${NODE_CACHE_ROOT}" >&2
    exit 3
fi

RUN_DIR="${RESULT_ROOT}/${CASE_NAME}/${PHASE}/node${NODE_RANK}"
if [[ "${PHASE}" == "cold" && -e "${RUN_DIR}" ]]; then
    echo "refusing to overwrite cold result directory: ${RUN_DIR}" >&2
    exit 3
fi
mkdir -p "${RUN_DIR}"

set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u

unset ASCEND_LAUNCH_BLOCKING
unset ASCEND_CUSTOM_OPP_PATH
unset ENABLE_PROFILING
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

export PYTHONPATH="${SGLANG_SOURCE}/python"
export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=1
export HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-enp196s0f0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-enp196s0f0}"
export STREAMS_PER_DEVICE="${STREAMS_PER_DEVICE:-32}"

{
    echo "case=${CASE_NAME}"
    echo "phase=${PHASE}"
    echo "variant=${VARIANT}"
    echo "cache_layout=${CACHE_LAYOUT}"
    echo "cache_root=${CACHE_ROOT}"
    echo "node_cache_preexisting=false"
    echo "node_rank=${NODE_RANK}"
    echo "hostname=$(hostname)"
    echo "source=${SGLANG_SOURCE}"
    echo "commit=$(git -C "${SGLANG_SOURCE}" rev-parse HEAD 2>/dev/null || true)"
    echo "git_status_begin"
    git -C "${SGLANG_SOURCE}" status --porcelain -- \
        python/sglang/kernels/ops/memory/allocator.py 2>/dev/null || true
    echo "git_status_end"
    sha256sum "${SGLANG_SOURCE}/python/sglang/kernels/ops/memory/allocator.py"
    python3 --version
    python3 - <<'PY'
import importlib.metadata as metadata
import torch
import triton

print("torch_import_version=", torch.__version__, sep="")
print("triton_import_version=", triton.__version__, sep="")
print("triton_import_file=", triton.__file__, sep="")
for name in ("torch-npu", "triton-ascend", "sglang"):
    try:
        print(f"{name}={metadata.version(name)}")
    except metadata.PackageNotFoundError:
        print(f"{name}=NOT_INSTALLED")
PY
    for version_file in \
        /usr/local/Ascend/ascend-toolkit/latest/version.cfg \
        /usr/local/Ascend/ascend-toolkit/latest/*-version.info; do
        if [[ -f "${version_file}" ]]; then
            echo "ascend_version_file=${version_file}"
            cat "${version_file}"
        fi
    done
    env | sort | grep -E '^(ASCEND_|HCCL_|SGLANG_|TORCH|TRITON_|PYTHONPATH=|STREAMS_PER_DEVICE=)' || true
} > "${RUN_DIR}/environment.log" 2>&1

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
    --variant "${VARIANT}" \
    --cache-root "${CACHE_ROOT}" \
    --cache-layout "${CACHE_LAYOUT}" \
    --result-dir "${RUN_DIR}" \
    --batch-size "${BATCH_SIZE}" \
    --page-size "${PAGE_SIZE}" \
    --prefix-tokens "${PREFIX_TOKENS}" \
    --extend-tokens "${EXTEND_TOKENS}" \
    --dump-stack-after-seconds "${STACK_DUMP_SECONDS}" \
    2>&1 | tee "${RUN_DIR}/torchrun.log"
launcher_rc="${PIPESTATUS[0]}"
set -e

echo "${launcher_rc}" > "${RUN_DIR}/exit_code.txt"
find "${NODE_CACHE_ROOT}" -type f -printf '%P\t%s\t%TY-%Tm-%TdT%TH:%TM:%TS%Tz\n' \
    2>/dev/null | sort > "${RUN_DIR}/cache_manifest.tsv" || true
exit "${launcher_rc}"
