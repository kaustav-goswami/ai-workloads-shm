# PyTorch Default-Allocator Benchmark Workloads

This document describes the seven benchmark workloads. Each workload is a
single, self-contained script that builds a small PyTorch model, initializes
its weights deterministically, and runs a forward pass (or, for PT-KVCache,
an incremental decode loop; for PT-MultiReader, several concurrent reader
threads) - entirely in one process, using ordinary tensors allocated with
**PyTorch's default CPU allocator**. There is no shared-memory region, no
custom allocator, and no writer/reader process split.

## Design Principles

| Concern | Approach |
| :--- | :--- |
| **Weights** | Model parameters and buffers are ordinary `torch.Tensor`s created by the model's `nn.Module` constructors, living in normal process heap memory. |
| **Growth (PT-KVCache)** | The per-layer K/V cache grows by concatenating a new tensor each step (`torch.cat`); PyTorch's default allocator handles the reallocation. |
| **Concurrency (PT-MultiReader)** | N threads in the same process run forward passes against one shared model instance - PyTorch's default allocator already makes the weight tensors visible to every thread without any explicit attach step. |
| **Determinism** | Every workload initializes weights (and inputs) from deterministic per-tensor-name seeds (`benchmark/common/seeds.py`), so runs are reproducible. |
| **Timing** | Every workload records per-operation latency (build/init/forward, or per-step for PT-KVCache) via `benchmark/common/timing.py` and prints a summary table. |

## Directory Layout

```
final-version/
├── EXAMPLE.md                   # this file
├── README.md                    # quick start
├── tests/
│   ├── smoke_test.py            # minimal single-process smoke test
│   └── test_benchmarks.py       # automated correctness suite (see below)
└── benchmark/
    ├── common/
    │   ├── seeds.py              # deterministic per-tensor-name seeding
    │   └── timing.py             # TimingRecorder: per-op latency capture + summary
    ├── pt_dense/model.py          # PT-Dense: monolithic LLaMA-style GEMM
    ├── pt_multitenant/model.py    # PT-MultiTenant: shared backbone + LoRA + MoE
    ├── pt_sparse/model.py         # PT-Sparse: DLRM embedding tables
    ├── pt_kvcache/model.py        # PT-KVCache: growing per-layer KV cache via torch.cat
    ├── pt_multireader/model.py    # PT-MultiReader: N concurrent reader threads
    ├── pt_quantized/model.py      # PT-Quantized: int8 weight + fp32 scale
    └── pt_vision/model.py         # PT-Vision: conv-only CNN, few large 4D tensors
```

Each `benchmark/<name>/model.py` contains everything for that workload: the
`nn.Module` definitions, `default_config()`, `build_model()`,
`initialize_weights()`, input generation, and a `main()` that runs the
workload end to end and prints a `SUCCESS` marker.

---

## Setup

No build step or native library is required - just Python and PyTorch.

```bash
export PYTHONPATH=/path/to/final-version:$PYTHONPATH
```

---

## Workload Summary

| Workload ID | Script | Model | Weight Layout | Primary Operator / Mechanism |
| :--- | :--- | :--- | :--- | :--- |
| **PT-Dense** | `benchmark/pt_dense/model.py` | LLaMA-style decoder (SwiGLU FFN, RMSNorm) | All layer weights, built once | Dense `nn.Linear` / GEMM |
| **PT-MultiTenant** | `benchmark/pt_multitenant/model.py` | LLaMA backbone + LoRA (`r=16`, `alpha=32` on `q_proj`/`v_proj`) + 8-expert Top-2 MoE | Backbone, LoRA adapters, expert blocks - many small heterogeneous tensors | Adapter injection + expert routing |
| **PT-Sparse** | `benchmark/pt_sparse/model.py` | Meta DLRM-style (8 embedding tables + bottom/top MLP) | 8 `nn.Embedding` tables (~40 MiB at default local scale) | `embedding_bag` row lookups |
| **PT-KVCache** | `benchmark/pt_kvcache/model.py` | 2-layer incremental decoder (attention + SwiGLU MLP) | Static attention/MLP weights, plus a per-layer K/V cache that grows one token at a time | `torch.cat`-based cache growth |
| **PT-MultiReader** | `benchmark/pt_multireader/model.py` | Smaller LLaMA-style backbone (reuses PT-Dense's classes) | One shared model instance | N reader **threads** running forward passes concurrently |
| **PT-Quantized** | `benchmark/pt_quantized/model.py` | Same LLaMA-style decoder shape as PT-Dense | Per-channel **int8 `qweight` + fp32 `scale`** buffers instead of fp32 weights | Runtime dequantization (`qweight.float() * scale`) inside `F.linear` |
| **PT-Vision** | `benchmark/pt_vision/model.py` | Small conv-only CNN (stem + residual conv blocks + FC head) | A handful of large 4D tensors instead of many small 2D ones | `nn.Conv2d` |

Default model sizes are reduced for fast local runs. The structural patterns
(monolithic vs segmented vs centralized sparse vs dynamically-growing vs
concurrently-read vs quantized vs conv-shaped) match the full-scale
specification.

---

## 1. PT-Dense - Monolithic Dense Transformer

All transformer weights (`layers.N.self_attn.q_proj.weight`, etc.) are
allocated as part of one `LlamaDenseModel` instance. The script builds the
model, initializes weights deterministically, runs a forward pass on a
random local activation tensor shaped `[batch, seq_len, hidden_size]`, and
prints the output shape/mean plus a timing summary.

```bash
python3 benchmark/pt_dense/model.py
```

---

## 2. PT-MultiTenant - LoRA + MoE Ensemble

Builds a `MultiTenantModel`: backbone attention/MLP norms and projections,
LoRA `lora_a`/`lora_b` matrices, and an 8-expert Top-2 MoE gate + expert
FFNs - all as ordinary parameters of one model instance. Runs a forward pass
that computes softmax routing weights and dispatches tokens to experts.

```bash
python3 benchmark/pt_multitenant/model.py
```

---

## 3. PT-Sparse - DLRM Embedding Tables

Builds a `DLRMModel` with 8 embedding tables plus bottom/top MLPs. Runs a
forward pass with a random batch of dense features, sparse indices, and
pooling offsets through `embedding_bag`.

```bash
python3 benchmark/pt_sparse/model.py --help  # no extra flags; region-size flags removed
python3 benchmark/pt_sparse/model.py
```

---

## 4. PT-KVCache - Dynamically Growing KV Cache

**What it tests that the other workloads don't:** PT-Dense/MultiTenant/
Sparse/Quantized/Vision all build their weight tensors once, at their final
size. Real inference serving is exactly the opposite: a KV cache starts
empty and grows by one token's worth of bytes on every decode step.

**Model:** a 2-layer decoder (`IncrementalDecoderLayer`: RMSNorm + attention
+ SwiGLU MLP, reusing PT-Dense's `LlamaAttention`/`LlamaMLP`/`RMSNorm`
classes). Static attention/MLP weights are initialized once, exactly like
PT-Dense. What's new is a per-layer K/V cache that grows one token at a
time via `torch.cat([k_cache, new_k], dim=2)` - PyTorch's default allocator
handles the reallocation, so there's no manual resize/relocate bookkeeping.

**Run loop:** for each decode step, the script generates a deterministic
per-step input token, calls `model.decode_step(x_t, kv_caches)` for every
layer, and keeps the returned (grown) K/V tensors as the caches for the next
step. Per-step latency is recorded via `TimingRecorder`.

```bash
python3 benchmark/pt_kvcache/model.py --max-new-tokens 24
```

`--max-new-tokens` controls how many decode steps run (default 24).

---

## 5. PT-MultiReader - Concurrent Reader Scaling

**What it tests that the other workloads don't:** every other workload runs
forward passes from a single call site. Real disaggregated-inference
deployments have N reader replicas accessing the same weights at once.
PT-MultiReader reuses PT-Dense's architecture (at a smaller scale) and
launches `--num-readers` (default 4) Python **threads**, each of which
independently runs a forward pass against the *same* model instance.

**Why threads instead of processes:** with a custom shared-memory
allocator, concurrent readers had to be separate OS processes that each
attached the region. With PyTorch's default allocator, weight tensors
already live in ordinary process memory - every thread in the process sees
them with zero copying and no attach step. Threads are therefore both
simpler and a more direct illustration of what "the default allocator
already gives you sharing for free, within a process" means.

**What's reported:** one row per thread (`elapsed_ms`, `out_mean`) plus
aggregate min/max/mean latency across all reader threads, and the wall-clock
time to launch, run, and join all of them.

```bash
python3 benchmark/pt_multireader/model.py --num-readers 8
```

---

## 6. PT-Quantized - Int8 Weight + Fp32 Scale

**What it tests that the other workloads don't:** every other workload uses
fp16/fp32 tensors. Real quantized-inference deployments (GPTQ/AWQ-style) use
much smaller int8 (or lower) weights alongside small fp32/fp16 scale
tensors. PT-Quantized builds a `QuantLinear`-based model whose `qweight`
(int8, per-output-channel symmetric quantization, no zero-point) and `scale`
(fp32, one value per output channel) are ordinary buffers, replacing
PT-Dense's single fp32 weight tensor per projection.

**What it reports:** the fp32-vs-quantized footprint and the resulting
compression ratio (~4x at the default per-channel int8 scheme), plus the
quantization error (`max abs diff`, relative to peak fp32 magnitude) between
the quantized forward pass and an independently-computed fp32 reference
forward pass on the same input.

```bash
python3 benchmark/pt_quantized/model.py
```

---

## 7. PT-Vision - Conv-Only CNN

**What it tests that the other workloads don't:** PT-Dense/MultiTenant/
Sparse/KVCache/Quantized all use mostly 2D `Linear`-shaped tensors (plus
embedding tables). PT-Vision uses a small conv-only CNN (`stem` + 4 residual
`ConvBlock`s + FC head) whose weights are **4D** (`[out_ch, in_ch, kh, kw]`)
and comparatively few but large - a genuinely different allocation shape.

```bash
python3 benchmark/pt_vision/model.py
```

---

## Classic Smoke Test (`tests/smoke_test.py`)

The original minimal linear-network demo: builds a 2-layer MLP, sets its
weights to fixed constants, runs a forward pass on a fixed input, and checks
the output against a known-good expected value.

```bash
python3 tests/smoke_test.py
```

Expected output ends with:

```
Output tensor: tensor([[37., 37.]], grad_fn=<AddmmBackward0>)
SUCCESS! Model weights initialized and forward pass verified correctly.
```

---

## Timing

`benchmark/common/timing.py` provides `TimingRecorder`, a small
`perf_counter`-based utility used by every workload:

```python
recorder = TimingRecorder(label="my workload")
with recorder.time("forward"):
    ...
recorder.print_summary()   # prints n / total / mean / p50 / p99 / max per op, in ms
```

Every `main()` tags `build`, `init`, and `forward` samples (or, for
PT-KVCache, one `step` sample per decode step) so the printed summary breaks
latency down by operation.

---

## Automated Testing

`tests/test_benchmarks.py` runs the classic smoke test and all seven
benchmark workloads as subprocesses and asserts a clean exit code plus a
`SUCCESS` marker in stdout:

```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
python3 tests/test_benchmarks.py
```

Uses only the standard library (`unittest`) - no extra dependencies.

---

## Extending to Production Scale

To approach full LLaMA-7B / 8.16 GB DLRM footprint:

1. Increase dimensions in each workload's `default_config()` inside
   `benchmark/<name>/model.py`.
2. Use FP16 (`pt_dense`, `pt_multitenant`, `pt_multireader`) or FP32
   (`pt_sparse`, `pt_kvcache`, `pt_quantized`'s reference model, `pt_vision`)
   consistently; PT-Quantized's weights are always int8 + fp32 regardless of
   the reference model's working dtype.

Each `benchmark/<name>/model.py` is the single place to adjust
initialization, model shape, and default sizes for that workload.
