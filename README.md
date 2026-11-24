# PyTorch Weights on the Default Allocator

This directory contains seven benchmark workloads (plus a classic smoke
test) that build small PyTorch models, initialize their weights
deterministically, and run a forward pass - all using ordinary tensors
allocated with **PyTorch's default (heap) CPU allocator**. There is no
shared-memory allocator, no zero-copy attach step, and no writer/reader
process split: each workload is a single Python file you run directly.

See **[EXAMPLE.md](EXAMPLE.md)** for details on all seven workloads.

## Directory Structure

*   `tests/`:
    *   `smoke_test.py`: minimal single-process smoke test (tiny 2-layer MLP with fixed weights).
    *   `test_benchmarks.py`: automated correctness test suite that runs the smoke test and all seven benchmark workloads (see [Automated Testing](#automated-testing)).
*   `benchmark/`: Seven benchmark workloads (PT-Dense, PT-MultiTenant, PT-Sparse, PT-KVCache, PT-MultiReader, PT-Quantized, PT-Vision), each a single `benchmark/<name>/model.py` file.
    *   `benchmark/common/`: shared helpers - `seeds.py` (deterministic per-tensor-name RNG seeding), `timing.py` (latency recording/reporting for build/init/forward operations).

* `benchmark_totals.csv` is automatically produced after `run_benchmarks.sh` is called. 
---

## Quick Start

No build step is required - just Python and PyTorch.

```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
python3 tests/smoke_test.py
```

Expected output ends with:

```
Output tensor: tensor([[37., 37.]], grad_fn=<AddmmBackward0>)
SUCCESS! Model weights initialized and forward pass verified correctly.
```

---

## Benchmark Workloads

| Workload | What it exercises | Script |
| :--- | :--- | :--- |
| PT-Dense | Monolithic dense transformer, sequential weight allocation | `benchmark/pt_dense/model.py` |
| PT-MultiTenant | LoRA + MoE, many small heterogeneous weight tensors | `benchmark/pt_multitenant/model.py` |
| PT-Sparse | DLRM-style embedding tables, `embedding_bag` lookups | `benchmark/pt_sparse/model.py` |
| PT-KVCache | Per-layer KV cache that **grows one token at a time via `torch.cat`** | `benchmark/pt_kvcache/model.py` |
| PT-MultiReader | **N concurrent reader threads** running forward passes against one shared model instance | `benchmark/pt_multireader/model.py` |
| PT-Quantized | Int8 weight + fp32 per-channel scale, runtime dequantization | `benchmark/pt_quantized/model.py` |
| PT-Vision | Conv-only CNN - few large 4D tensors instead of many small ones | `benchmark/pt_vision/model.py` |

Example (PT-Dense):

```bash
python3 benchmark/pt_dense/model.py
```

Every workload prints a per-operation timing summary (build/init/forward, or
per-step timings for PT-KVCache) via `benchmark/common/timing.py` - see
[EXAMPLE.md](EXAMPLE.md) for details and the extra flags PT-KVCache
(`--max-new-tokens`) and PT-MultiReader (`--num-readers`) accept.

---

## Automated Testing

`tests/test_benchmarks.py` runs the classic smoke test plus all seven
benchmark workloads end to end (each as its own subprocess) and checks that
every script exits cleanly and prints a `SUCCESS` marker:

```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
python3 tests/test_benchmarks.py
```

Uses the standard library `unittest` - no extra dependencies required.
