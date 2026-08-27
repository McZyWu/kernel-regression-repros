#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repro_script="$script_dir/ifa_npugraph_update_repro.py"
summary_script="$script_dir/summarize_results.py"

old_container=""
new_container=""
output_dir=""
device=0
rounds=3
measurement_blocks=10
warmup=20
iters=150
eager_iters=100
records=48
seq_len=3500
cpu_set=""

usage() {
  cat <<'EOF'
Run the IFA NPU Graph reproducer alternately in two existing containers.

Usage:
  run_container_matrix.sh \
    --old-container CONTAINER \
    --new-container CONTAINER \
    --output-dir DIR \
    [--device N] [--rounds N] [--measurement-blocks N] \
    [--warmup N] [--iters N] [--eager-iters N] \
    [--records N] [--seq-len N] [--cpu-set CPU-LIST]

The host must have docker, npu-smi, and python3. Both containers must already
be running and expose the requested physical NPU. The runner refuses to start
an individual measurement while any NPU process is visible on the host.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --old-container) old_container=${2:?missing value}; shift 2 ;;
    --new-container) new_container=${2:?missing value}; shift 2 ;;
    --output-dir) output_dir=${2:?missing value}; shift 2 ;;
    --device) device=${2:?missing value}; shift 2 ;;
    --rounds) rounds=${2:?missing value}; shift 2 ;;
    --measurement-blocks) measurement_blocks=${2:?missing value}; shift 2 ;;
    --warmup) warmup=${2:?missing value}; shift 2 ;;
    --iters) iters=${2:?missing value}; shift 2 ;;
    --eager-iters) eager_iters=${2:?missing value}; shift 2 ;;
    --records) records=${2:?missing value}; shift 2 ;;
    --seq-len) seq_len=${2:?missing value}; shift 2 ;;
    --cpu-set) cpu_set=${2:?missing value}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$old_container" || -z "$new_container" || -z "$output_dir" ]]; then
  echo "--old-container, --new-container, and --output-dir are required" >&2
  usage >&2
  exit 2
fi

for required_command in docker find npu-smi python3 sha256sum; do
  command -v "$required_command" >/dev/null || {
    echo "required host command not found: $required_command" >&2
    exit 2
  }
done

for value_name in device rounds measurement_blocks warmup iters eager_iters records seq_len; do
  value=${!value_name}
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "$value_name must be a non-negative integer: $value" >&2
    exit 2
  fi
done
if (( rounds < 1 || measurement_blocks < 1 || iters < 1 || eager_iters < 1 || records < 1 || seq_len < 1 )); then
  echo "rounds, measurement-blocks, iters, eager-iters, records, and seq-len must be positive" >&2
  exit 2
fi

if [[ -e "$output_dir" && ! -d "$output_dir" ]]; then
  echo "output path exists and is not a directory: $output_dir" >&2
  exit 2
fi
if [[ -d "$output_dir" && -n $(find "$output_dir" -mindepth 1 -maxdepth 1 -print -quit) ]]; then
  echo "output directory must be new or empty: $output_dir" >&2
  exit 2
fi
mkdir -p "$output_dir"
output_dir=$(cd -- "$output_dir" && pwd)

npu_process_ids() {
  npu-smi info 2>/dev/null \
    | awk -F'|' '$3 ~ /^[[:space:]]*[0-9]+[[:space:]]*$/ {
        gsub(/[[:space:]]/, "", $3)
        print $3
      }' \
    | sort -nu
}

require_idle_npus() {
  local process_ids
  process_ids=$(npu_process_ids)
  if [[ -n "$process_ids" ]]; then
    echo "NPU processes are already running; refusing to contaminate the measurement:" >&2
    echo "$process_ids" >&2
    return 75
  fi
}

for container in "$old_container" "$new_container"; do
  if [[ $(docker inspect -f '{{.State.Running}}' "$container" 2>/dev/null || true) != true ]]; then
    echo "container is not running: $container" >&2
    exit 2
  fi
  if [[ -n "$cpu_set" ]]; then
    docker exec "$container" taskset -c "$cpu_set" true
  fi
done

sha256sum "$repro_script" >"$output_dir/reproducer.sha256"
{
  echo "started=$(date --iso-8601=seconds)"
  echo "host=$(hostname)"
  echo "kernel=$(uname -a)"
  echo "old_container=$old_container"
  echo "new_container=$new_container"
  echo "device=$device"
  echo "rounds=$rounds"
  echo "measurement_blocks=$measurement_blocks"
  echo "warmup=$warmup"
  echo "iters=$iters"
  echo "eager_iters=$eager_iters"
  echo "records=$records"
  echo "seq_len=$seq_len"
  echo "cpu_set=${cpu_set:-unbound}"
  for container in "$old_container" "$new_container"; do
    docker inspect -f \
      'container={{.Name}}|image={{.Config.Image}}|image_id={{.Image}}|status={{.State.Status}}' \
      "$container"
    docker exec -i "$container" python3 - <<'PY'
import json
import platform

import torch
import torch_npu

print(json.dumps({
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_npu": getattr(torch_npu, "__version__", "unknown"),
    "torch_npu_file": torch_npu.__file__,
}, sort_keys=True))
PY
  done
} >"$output_dir/environment.txt" 2>&1

run_one() {
  local round=$1
  local label=$2
  local container=$3
  local stem
  stem=$(printf '%s/round_%02d_%s' "$output_dir" "$round" "$label")

  require_idle_npus || {
    npu-smi info >"${stem}_busy_npu_smi.txt" 2>&1 || true
    return 75
  }

  echo "BEGIN $(date --iso-8601=seconds) round=$round stack=$label container=$container" \
    | tee -a "$output_dir/harness.log"
  npu-smi info >"${stem}_before_npu_smi.txt" 2>&1 || true

  local -a python_command=(python3 -)
  if [[ -n "$cpu_set" ]]; then
    python_command=(taskset -c "$cpu_set" python3 -)
  fi

  local started=$SECONDS
  set +e
  docker exec -i \
    -e NPU_DEVICE_INDEX="$device" \
    -e PYTHONHASHSEED=0 \
    -e OMP_NUM_THREADS=1 \
    -e MKL_NUM_THREADS=1 \
    "$container" \
    "${python_command[@]}" \
      --device "$device" \
      --records "$records" \
      --unique-tensors \
      --seq-len "$seq_len" \
      --warmup "$warmup" \
      --iters "$iters" \
      --measurement-blocks "$measurement_blocks" \
      --eager-iters "$eager_iters" \
      <"$repro_script" >"${stem}.json" 2>"${stem}.stderr.log"
  local status=$?
  set -e
  local elapsed=$((SECONDS - started))

  echo "END $(date --iso-8601=seconds) round=$round stack=$label status=$status elapsed_s=$elapsed" \
    | tee -a "$output_dir/harness.log"
  if [[ $status -ne 0 ]]; then
    echo "measurement failed; see ${stem}.stderr.log" >&2
    return "$status"
  fi
  sleep 2
}

require_idle_npus
for round in $(seq 1 "$rounds"); do
  if (( round % 2 == 1 )); then
    order=("old:$old_container" "new:$new_container")
  else
    order=("new:$new_container" "old:$old_container")
  fi
  for item in "${order[@]}"; do
    label=${item%%:*}
    container=${item#*:}
    run_one "$round" "$label" "$container"
  done
done

echo "COMPLETE $(date --iso-8601=seconds) rounds=$rounds" \
  | tee -a "$output_dir/harness.log"
python3 "$summary_script" "$output_dir" | tee "$output_dir/summary.md"
