#!/usr/bin/env bash
# Builds allocator/build/libshm_alloc.so from the ROOT repo's v0.3 dmalloc
# engine (src/shm_slim.c, src/dmalloc.c, src/shm_ns.c) instead of the
# allocator/ git submodule's own v2 sources (src/shm_alloc.c, src/shm_ns.c).
#
# shm_alloc_python.py in this directory still loads
# "./allocator/build/libshm_alloc.so" -- only what that .so is built FROM
# has changed, not where it lives, so the Python-side load path needed no
# edit.
#
# Usage:
#   ./build_dmalloc_lib.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
OUT_DIR="$HERE/allocator/build"

mkdir -p "$OUT_DIR"

gcc -shared -fPIC -std=c11 -Wall -Wextra -g -D_POSIX_C_SOURCE=200809L \
    -I"$ROOT/include" -I"$ROOT/src" \
    "$ROOT/src/shm_slim.c" "$ROOT/src/dmalloc.c" "$ROOT/src/shm_ns.c" \
    -o "$OUT_DIR/libshm_alloc.so" -pthread -lrt

echo "Built $OUT_DIR/libshm_alloc.so from v0.3 dmalloc engine sources."
