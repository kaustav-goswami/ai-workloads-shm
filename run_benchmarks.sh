#!/usr/bin/env bash
# Runs every writer/reader benchmark pair in this (v0.3 dmalloc-backed)
# variant and writes each one's wall-clock total_ms to a CSV file, in the
# same shape as ai-workloads-shm-default/run_benchmarks.sh so the two are
# directly comparable.
#
# PT-KVCache is excluded: it depends on region.resize() (in-place growth of
# an already-published named object), which v0.3's dmalloc engine has no
# primitive for (d_publish() is a one-shot copy-in; there is no realloc).
# See shm_alloc_python.py's SHMRegion.resize() and WIP.md.
#
# Each workload is writer.py + reader.py run as a pair: the writer is
# started in the background, the reader is run in the foreground (it polls
# the sync variable for the writer to finish publishing), and total_ms is
# the wall-clock time from starting the writer to the writer exiting after
# the reader signals done.
#
# Usage:
#   ./run_benchmarks.sh [output.csv]

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

OUTPUT="${1:-$ROOT/benchmark_totals.csv}"

WORKLOAD_NAMES=(pt_dense pt_multitenant pt_sparse pt_multireader pt_quantized pt_vision)

echo "workload,total_ms" > "$OUTPUT"

for name in "${WORKLOAD_NAMES[@]}"; do
    shm_path="/ml_bench_${name}_$$"
    echo "Running $name (writer+reader)..." >&2

    start_ns=$(date +%s%N)
    python3 "$ROOT/benchmark/$name/writer.py" --path "$shm_path" >/tmp/${name}_writer.log 2>&1 &
    writer_pid=$!

    reader_output="$(python3 "$ROOT/benchmark/$name/reader.py" --path "$shm_path" 2>&1)"
    reader_status=$?
    wait "$writer_pid"
    writer_status=$?
    end_ns=$(date +%s%N)

    rm -f "/dev/shm${shm_path}" 2>/dev/null

    if [[ $reader_status -ne 0 || $writer_status -ne 0 ]]; then
        echo "  $name FAILED (reader exit $reader_status, writer exit $writer_status):" >&2
        echo "$reader_output" >&2
        cat "/tmp/${name}_writer.log" >&2
        echo "${name}," >> "$OUTPUT"
        continue
    fi

    total_ms=$(( (end_ns - start_ns) / 1000000 ))
    echo "  $name: total_ms=$total_ms"
    echo "${name},${total_ms}" >> "$OUTPUT"
done

echo "Wrote $OUTPUT" >&2
