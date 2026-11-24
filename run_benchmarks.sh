#!/usr/bin/env bash
# Runs every benchmark workload except PT-KVCache and writes each one's
# reported TOTAL total_ms (from benchmark/common/timing.py's TimingRecorder
# summary) to a CSV file.
#
# Usage:
#   ./run_benchmarks.sh [output.csv]
#
# Output defaults to benchmark_totals.csv in the repo root.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

OUTPUT="${1:-$ROOT/benchmark_totals.csv}"

# name -> script, in run order. PT-KVCache is intentionally excluded.
WORKLOAD_NAMES=(pt_dense pt_multitenant pt_sparse pt_multireader pt_quantized pt_vision)
declare -A WORKLOAD_SCRIPTS=(
    [pt_dense]="benchmark/pt_dense/model.py"
    [pt_multitenant]="benchmark/pt_multitenant/model.py"
    [pt_sparse]="benchmark/pt_sparse/model.py"
    [pt_multireader]="benchmark/pt_multireader/model.py"
    [pt_quantized]="benchmark/pt_quantized/model.py"
    [pt_vision]="benchmark/pt_vision/model.py"
)

echo "workload,total_ms" > "$OUTPUT"

for name in "${WORKLOAD_NAMES[@]}"; do
    script="${WORKLOAD_SCRIPTS[$name]}"
    echo "Running $name ($script)..." >&2

    output="$(python3 "$ROOT/$script" 2>&1)"
    status=$?

    if [[ $status -ne 0 ]]; then
        echo "  $name FAILED (exit $status):" >&2
        echo "$output" >&2
        echo "${name}," >> "$OUTPUT"
        continue
    fi

    total_ms="$(printf '%s\n' "$output" | awk '/^  TOTAL/{print $3; exit}')"
    if [[ -z "$total_ms" ]]; then
        echo "  $name: no TOTAL total_ms found in output" >&2
        echo "${name}," >> "$OUTPUT"
        continue
    fi

    echo "  $name: total_ms=$total_ms"
    echo "${name},${total_ms}" >> "$OUTPUT"
done

echo "Wrote $OUTPUT" >&2
