# PyTorch Shared Memory / Disaggregated Memory Weights Allocator

This directory contains the final version of the code and documentation showing how to build and use shared-memory PyTorch weights with the `disagg-shmem-allocator`.

See **[EXAMPLE.md](EXAMPLE.md)** for all seven benchmark workloads, their sync protocols, and multi-process run instructions.

## Directory Structure

*   `allocator/`: Git submodule (`disagg-shmem-allocator`, checked out under the local path `allocator/`) containing the custom C shared-memory allocator, its own unit tests, and `build/libshm_alloc.so` once compiled.
*   `shm_alloc_python.py`: Python `ctypes` bindings for the allocator shared library — region open/close, `alloc`/`free`/`resize`/`lookup`, and wrapping raw addresses directly as PyTorch tensors (zero-copy).
*   `tests/`:
    *   `writer.py` / `reader.py`: classic two-process smoke test.
    *   `test_benchmarks.py`: automated correctness test suite covering the smoke test and all seven benchmark workloads (see [Automated Testing](#automated-testing)).
*   `benchmark/`: Seven fragmentation/scaling evaluation workloads (PT-Dense, PT-MultiTenant, PT-Sparse, PT-KVCache, PT-MultiReader, PT-Quantized, PT-Vision), each with its own `model.py` / `weights_loader.py` / `writer.py` / `reader.py`.
    *   `benchmark/common/`: shared helpers — `shm_weights.py` (region setup, sync protocol, generic alloc/attach helpers), `seeds.py` (deterministic per-tensor-name RNG seeding), `timing.py` (latency recording/reporting for every SHM op).

---

## How to Build the Allocator Shared Library

To build the allocator, you must first compile the C subproject as a shared library (`.so` file):

1.  Initialize the git submodule if it hasn't been initialized:
    ```bash
    git submodule update --init --recursive
    ```
2.  Navigate to the submodule directory and compile the shared library:
    ```bash
    cd allocator
    gcc -shared -fPIC -std=c11 -Wall -Wextra -g -D_POSIX_C_SOURCE=200809L -Iinclude -Isrc src/shm_alloc.c src/shm_ns.c -o build/libshm_alloc.so -pthread -lrt
    ```
3.  Go back to the `final-version` directory:
    ```bash
    cd ..
    ```

Set `PYTHONPATH` to the `final-version` directory before running any script:

```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
```

(Optional) Run the allocator's own C unit tests to sanity-check the build:

```bash
cd allocator && make clean && make test && cd ..
```

---

## Quick Start: Classic Smoke Test

### Option A: Using POSIX Shared Memory (Default)

1.  **Terminal 1 (Writer)**:
    ```bash
    python3 tests/writer.py --backend shm --path /pytorch_model_weights_shm
    ```
2.  **Terminal 2 (Reader)**:
    ```bash
    python3 tests/reader.py --backend shm --path /pytorch_model_weights_shm
    ```

### Option B: Using DAX / CXL Disaggregated Memory

1.  **Terminal 1 (Writer)**:
    ```bash
    python3 tests/writer.py --backend dax --path /dev/dax0.0
    ```
2.  **Terminal 2 (Reader)**:
    ```bash
    python3 tests/reader.py --backend dax --path /dev/dax0.0
    ```

---

## Benchmark Workloads

| Workload | What it exercises | Writer | Reader |
| :--- | :--- | :--- | :--- |
| PT-Dense | Monolithic dense transformer, sequential SHM allocation | `benchmark/pt_dense/writer.py` | `benchmark/pt_dense/reader.py` |
| PT-MultiTenant | LoRA + MoE, heavily segmented/fragmented allocation | `benchmark/pt_multitenant/writer.py` | `benchmark/pt_multitenant/reader.py` |
| PT-Sparse | Centralized DLRM embedding tables, `embedding_bag` lookups | `benchmark/pt_sparse/writer.py` | `benchmark/pt_sparse/reader.py` |
| PT-KVCache | Per-layer KV cache that **grows one token at a time via `shm_resize`** | `benchmark/pt_kvcache/writer.py` | `benchmark/pt_kvcache/reader.py` |
| PT-MultiReader | **N concurrent reader processes** attaching the same weights at once | `benchmark/pt_multireader/writer.py` | `benchmark/pt_multireader/reader.py` |
| PT-Quantized | Int8 weight + fp32 per-channel scale, sharing quantized tensors | `benchmark/pt_quantized/writer.py` | `benchmark/pt_quantized/reader.py` |
| PT-Vision | Conv-only CNN — few large 4D tensors instead of many small ones | `benchmark/pt_vision/writer.py` | `benchmark/pt_vision/reader.py` |

Example (PT-Dense):

```bash
python3 benchmark/pt_dense/writer.py --backend shm --path /pt_dense_shm
python3 benchmark/pt_dense/reader.py --backend shm --path /pt_dense_shm
```

Every writer/reader prints a per-operation timing summary (alloc/copy on the writer, lookup/attach on the reader) via `benchmark/common/timing.py` — see [EXAMPLE.md](EXAMPLE.md) for details and for the extra flags PT-KVCache (`--max-new-tokens`) and PT-MultiReader (`--num-readers`) accept.

Full design notes, sync protocols, and scaling guidance are in [EXAMPLE.md](EXAMPLE.md).

---

## Automated Testing

`tests/test_benchmarks.py` runs the classic smoke test plus all seven benchmark workloads end to end (writer + reader, real subprocesses, real shared memory) and checks that both sides exit cleanly and the reader reports success:

```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
python3 tests/test_benchmarks.py
```

It waits on each writer's shared-memory region actually appearing (no fixed sleeps), so timing is adaptive to machine load. It uses the standard library `unittest` — no extra dependencies required.
