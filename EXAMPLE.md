# PyTorch Shared-Memory Fragmentation Evaluation Examples

This document describes the seven benchmark workloads that exercise the
`disagg-shmem-allocator` with PyTorch. Each workload uses a **writer** process
that loads model weights into a shared region and a **reader** process that
attaches those weights and runs a forward pass with **local activations only**
(except PT-MultiReader, which attaches from many reader processes at once).

## Design Principles

| Concern | Approach |
| :--- | :--- |
| **Shared memory** | Model weights (and embedding tables / KV caches / quantized tensors where noted) are allocated by name in the shared region and attached as zero-copy PyTorch tensors. |
| **Process-local memory** | Activations, inputs, batch indices, offsets, dense/top MLP weights (PT-Sparse), and routing logits (PT-MultiTenant) stay in ordinary heap memory. |
| **Multi-process testing** | Run a writer in one terminal and a reader in another on the same machine. A `sync_ready` flag coordinates startup and completion. Weight initialization uses deterministic per-name seeds so writer and reader agree across processes. |
| **Weight loaders** | Each workload has its own `weights_loader.py` module that owns model construction, deterministic initialization, SHM write, and SHM attach helpers. |
| **Timing** | Every workload records per-operation latency (alloc/copy on the writer, lookup/attach on the reader, plus resize/wait/read for PT-KVCache) via `benchmark/common/timing.py` and prints a summary table. |

## Directory Layout

```
final-version/
├── shm_alloc_python.py          # ctypes bindings to libshm_alloc.so
├── EXAMPLE.md                   # this file
├── README.md                    # build instructions
├── allocator/                   # git submodule: the C allocator + its own C unit tests
├── tests/
│   ├── writer.py                # classic two-process smoke test (writer)
│   ├── reader.py                # classic two-process smoke test (reader)
│   └── test_benchmarks.py       # automated correctness suite (see below)
└── benchmark/
    ├── common/
    │   ├── shm_weights.py       # shared region helpers and sync protocol
    │   ├── seeds.py             # deterministic per-tensor-name seeding
    │   └── timing.py            # TimingRecorder: per-op latency capture + summary
    ├── pt_dense/                 # PT-Dense: monolithic LLaMA-style GEMM
    ├── pt_multitenant/           # PT-MultiTenant: shared backbone + LoRA + MoE
    ├── pt_sparse/                 # PT-Sparse: DLRM embedding tables
    ├── pt_kvcache/                # PT-KVCache: growing per-layer KV cache via shm_resize
    ├── pt_multireader/            # PT-MultiReader: N concurrent reader processes
    ├── pt_quantized/              # PT-Quantized: int8 weight + fp32 scale sharing
    └── pt_vision/                 # PT-Vision: conv-only CNN, few large 4D tensors
```

Each benchmark directory contains:

- `model.py` — PyTorch module definition
- `weights_loader.py` — build, initialize, write-to-SHM, attach-from-SHM
- `writer.py` — creates the region and populates shared weights
- `reader.py` — attaches shared weights and validates a forward pass

---

## Build

From `final-version/`:

```bash
git submodule update --init --recursive
cd allocator
gcc -shared -fPIC -std=c11 -Wall -Wextra -g -D_POSIX_C_SOURCE=200809L \
  -Iinclude -Isrc src/shm_alloc.c src/shm_ns.c -o build/libshm_alloc.so -pthread -lrt
cd ..
```

All commands below assume:

```bash
export PYTHONPATH=/path/to/final-version:$PYTHONPATH
```

---

## Workload Summary

| Workload ID | Directory | Model | Shared Weights | Local Variables | Primary Operator / Mechanism |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PT-Dense** | `benchmark/pt_dense/` | LLaMA-style decoder (SwiGLU FFN, RMSNorm) | All layer weights (monolithic sequential SHM allocations) | Activations, inputs | Dense `nn.Linear` / GEMM |
| **PT-MultiTenant** | `benchmark/pt_multitenant/` | LLaMA backbone + LoRA (`r=16`, `alpha=32` on `q_proj`/`v_proj`) + 8-expert Top-2 MoE | Backbone, LoRA adapters, and expert blocks (segmented SHM objects) | Activations, routing logits computed locally | Adapter injection + expert routing |
| **PT-Sparse** | `benchmark/pt_sparse/` | Meta DLRM-style (8 embedding tables + bottom/top MLP) | 8 `nn.Embedding` tables (~40 MiB at default local scale) | Bottom/top MLP weights, dense features, sparse indices/offsets | `embedding_bag` row lookups |
| **PT-KVCache** | `benchmark/pt_kvcache/` | 2-layer incremental decoder (attention + SwiGLU MLP) | Static attention/MLP weights (written once) **plus** a per-layer K/V cache that grows one token at a time | The token stream itself (deterministic per-step) | `shm_resize()` grow-in-place / grow-with-relocation, under an optimistic-concurrency read protocol |
| **PT-MultiReader** | `benchmark/pt_multireader/` | Smaller LLaMA-style backbone (reuses PT-Dense's classes) | All backbone weights, written once by a single writer | Activations | N reader **processes** calling `shm_lookup_by_name`/`shm_ptr` concurrently against the same live region |
| **PT-Quantized** | `benchmark/pt_quantized/` | Same LLaMA-style decoder shape as PT-Dense | Per-channel **int8 `qweight` + fp32 `scale`** buffers (two SHM objects per projection) instead of fp32 weights | Activations | Runtime dequantization (`qweight.float() * scale`) inside `F.linear` |
| **PT-Vision** | `benchmark/pt_vision/` | Small conv-only CNN (stem + residual conv blocks + FC head) | All conv/FC weights — a handful of large 4D tensors instead of many small 2D ones | Input image batch | `nn.Conv2d` |

Default model sizes are reduced for local multi-process testing. The structural
patterns (monolithic vs segmented vs centralized sparse vs dynamically-growing
vs concurrently-read vs quantized vs conv-shaped) match the full-scale
specification.

---

## 1. PT-Dense — Monolithic Dense Transformer

**Sharing model:** 100% of transformer weights are written sequentially into the
shared region as individually named tensors (`layers.N.self_attn.q_proj.weight`, etc.).

**Local data:** Random activation tensor shaped `[batch, seq_len, hidden_size]`.

### Run (POSIX shared memory)

Terminal 1 — writer:

```bash
python3 benchmark/pt_dense/writer.py --backend shm --path /pt_dense_shm
```

Terminal 2 — reader:

```bash
python3 benchmark/pt_dense/reader.py --backend shm --path /pt_dense_shm
```

The reader rebuilds a reference model with the same deterministic weight init,
attaches SHM weights, runs forward on local activations, and compares against the
reference output.

---

## 2. PT-MultiTenant — LoRA + MoE Ensemble

**Sharing model:**

- **Shared (read-only views on reader):** backbone attention/MLP norms and projections, LoRA `lora_a`/`lora_b` matrices, MoE gate and per-expert FFN weights.
- **Segmentation:** each tensor is a separate named SHM object, producing a fragmented address layout within the region.
- **Local:** activations and softmax routing weights computed during forward.

### Run

Terminal 1:

```bash
python3 benchmark/pt_multitenant/writer.py --backend shm --path /pt_multitenant_shm
```

Terminal 2:

```bash
python3 benchmark/pt_multitenant/reader.py --backend shm --path /pt_multitenant_shm
```

---

## 3. PT-Sparse — DLRM Embedding Tables

**Sharing model:**

- **Shared:** all 8 embedding tables (`tables.N.weight`).
- **Local:** bottom MLP and top MLP weights, dense input features, sparse indices, and pooling offsets.

This mirrors centralized shared embedding storage with concurrent random row access
via `embedding_bag`.

### Run

Terminal 1:

```bash
python3 benchmark/pt_sparse/writer.py --backend shm --path /pt_sparse_shm --region-size-gb 1
```

Terminal 2:

```bash
python3 benchmark/pt_sparse/reader.py --backend shm --path /pt_sparse_shm
```

Increase `--region-size-gb` if you scale `num_embeddings` in `benchmark/pt_sparse/model.py`.

---

## 4. PT-KVCache — Dynamically Growing KV Cache

**What it tests that the other three don't:** none of PT-Dense/MultiTenant/Sparse
ever call `shm_resize()` — every object is allocated once, at its final size, and
never touched again. Real inference serving is exactly the opposite: a KV cache
starts empty and grows by one token's worth of bytes on every decode step, which
is precisely the workload that can trigger grow-in-place vs
grow-with-relocation behavior in the allocator (see [Known Issues](#known-issues-found-and-fixed-during-development)
below — this workload is what surfaced a real allocator bug).

**Model:** a 2-layer decoder (`IncrementalDecoderLayer`: RMSNorm + attention +
SwiGLU MLP, reusing PT-Dense's `LlamaAttention`/`LlamaMLP`/`RMSNorm` classes for
their submodules). Static attention/MLP weights are written to SHM once, exactly
like PT-Dense. What's new is a per-layer K/V cache that the writer grows one
token at a time.

**Layout constraint (important):** the SHM-resident K/V buffer is stored as
`[sequence_length, num_heads, head_dim]` — sequence length **must** be the
outermost dimension. `shm_resize()`'s grow-with-relocation path preserves data
by copying raw bytes from the front of the old buffer into the front of the new
one; if the *growing* dimension were anywhere but outermost (e.g. the more
natural `[batch, heads, length, dim]` attention layout), growing it would shift
every later element's byte offset except the first, silently corrupting
already-written tokens. `benchmark/pt_kvcache/weights_loader.py`'s
`to_shm_layout()`/`from_shm_layout()` convert between this SHM layout and the
model's normal `[batch, heads, length, dim]` attention layout on every write/read.

**Sync protocol (extends the usual `sync_ready` handshake):**

1. Writer allocates static weights, an empty (1-token-capacity) K/V buffer and a
   `[version, current_length]` int64 pair per layer, and a `decode_start` scalar
   (0). It then signals `sync_ready = 1` (static weights ready).
2. Reader attaches static weights, then sets `decode_start = 1`.
3. Writer waits for `decode_start == 1` before generating any tokens, so the
   reader never misses the first steps.
4. For each decode step and each layer, the writer:
   `version += 1` (now odd = "update in progress") → `shm_resize()` both K and V
   → write the new token's slice → `current_length += 1` → `version += 1` (now
   even = "stable").
5. The reader polls `(version, current_length)`; once `version` is even and
   `current_length` has reached the target, it takes an **optimistic-concurrency
   snapshot**: read `version` before, `.clone()` the K/V tensors, read `version`
   after, and retry if they differ (a writer update happened mid-read and may
   have relocated the block).
6. After all steps, the reader signals `sync_ready = 2` as usual and the writer exits.

**Correctness check:** the reader computes the SAME decode sequence twice —
once via `model.decode_step_from_cache()`, which attends against K/V tensors
read verbatim from SHM, and once via a fully independent local reference model
(`ref_model.decode_step()`) that never touches SHM for its own KV cache. Both
use identical deterministic per-step inputs and identical (independently
re-initialized) weights, so any divergence between the two paths means the SHM
KV cache round-trip — not the model math — is wrong.

### Run

Terminal 1:

```bash
python3 benchmark/pt_kvcache/writer.py --backend shm --path /pt_kvcache_shm --max-new-tokens 24
```

Terminal 2:

```bash
python3 benchmark/pt_kvcache/reader.py --backend shm --path /pt_kvcache_shm --max-new-tokens 24
```

`--max-new-tokens` must match between writer and reader (default 24).

---

## 5. PT-MultiReader — Concurrent Reader Scaling

**What it tests that the other workloads don't:** every other workload is
strictly one writer, one reader. Real disaggregated-inference deployments have
N reader replicas attaching to the same weight region at once. PT-MultiReader
reuses PT-Dense's architecture (at a smaller scale, so N torch-import-heavy
subprocesses stay fast to start) and has a single `reader.py` invocation spawn
`--num-readers` (default 4) independent OS **processes** via
`multiprocessing.get_context("fork")`, each of which does its own
`shm_lookup_by_name` + `shm_ptr` + forward pass + validation, completely
independently of the others.

**Sharing model:** identical to PT-Dense — one writer publishes the full
backbone once; the only thing that's different is how many processes attach
to it, and whether they can do so concurrently without corrupting each
other's reads (the underlying allocator serializes directory access via
`region_lock()`/`region_unlock()`, but reads of already-written payload data
need no lock at all since nothing is mutating it).

**What's reported:** the reader prints one row per subprocess (`ok`,
`max_diff`, `attach_ms`, `total_ms`) plus aggregate min/max/mean attach
latency across the whole reader fleet, and the whole wall-clock time to
launch, attach, validate, and join all N readers.

### Run

Terminal 1:

```bash
python3 benchmark/pt_multireader/writer.py --backend shm --path /pt_multireader_shm
```

Terminal 2:

```bash
python3 benchmark/pt_multireader/reader.py --backend shm --path /pt_multireader_shm --num-readers 8
```

---

## 6. PT-Quantized — Int8 Weight + Fp32 Scale Sharing

**What it tests that the other workloads don't:** every other workload shares
fp16/fp32 tensors. Real quantized-inference deployments (GPTQ/AWQ-style) share
much smaller int8 (or lower) weights alongside small fp32/fp16 scale tensors.
PT-Quantized shares a `QuantLinear` layer's `qweight` (int8, per-output-channel
symmetric quantization, no zero-point) and `scale` (fp32, one value per output
channel) as **two separate SHM objects per projection**, replacing PT-Dense's
single fp32 weight tensor per projection.

Because `qweight`/`scale` are registered as PyTorch **buffers** (not
`Parameter`s) on `QuantLinear`, they show up in `model.state_dict()` just like
any other tensor — so the writer/reader reuse the exact same generic
`copy_state_dict_to_shm`/`attach_state_dict_from_shm` helpers every other
workload uses. No bespoke SHM plumbing was needed for the new dtype; only
`shm_alloc_python.py`'s dtype table needed an `int8`/`uint8` entry.

**Correctness check:** the reader builds its own fp32 reference model (same
deterministic PT-Dense init) and quantizes it **locally**, then runs that
against the same input and compares to the SHM-driven output — this isolates
"did the int8 bytes and scales transfer correctly through SHM" (should match
almost exactly) from "how much error does quantization itself introduce"
(reported separately, informationally, against the *true* fp32 forward pass).

### Run

Terminal 1:

```bash
python3 benchmark/pt_quantized/writer.py --backend shm --path /pt_quantized_shm
```

Terminal 2:

```bash
python3 benchmark/pt_quantized/reader.py --backend shm --path /pt_quantized_shm
```

The writer prints the fp32-vs-quantized footprint and the resulting compression
ratio (~4x at the default per-channel int8 scheme); the reader prints the
quantization error relative to the true fp32 model.

---

## 7. PT-Vision — Conv-Only CNN

**What it tests that the other workloads don't:** PT-Dense/MultiTenant/Sparse/
KVCache/Quantized all share mostly 2D `Linear`-shaped tensors (plus embedding
tables). PT-Vision shares a small conv-only CNN (`stem` + 4 residual
`ConvBlock`s + FC head) whose weights are **4D** (`[out_ch, in_ch, kh, kw]`)
and comparatively few but large, a genuinely different allocation shape to
pack into the region.

**Sharing model:** all conv/FC weights and biases are shared; the input image
batch is process-local.

### Run

Terminal 1:

```bash
python3 benchmark/pt_vision/writer.py --backend shm --path /pt_vision_shm
```

Terminal 2:

```bash
python3 benchmark/pt_vision/reader.py --backend shm --path /pt_vision_shm
```

---

## Classic Smoke Test (`tests/`)

The original minimal linear-network demo validates basic SHM integration:

Terminal 1:

```bash
python3 tests/writer.py --backend shm --path /pytorch_model_weights_shm
```

Terminal 2:

```bash
python3 tests/reader.py --backend shm --path /pytorch_model_weights_shm
```

Expected reader output: `[[37.0, 37.0]]`.

---

## DAX / Disaggregated Memory

All writer/reader scripts accept `--backend dax --path /dev/dax0.0`. DAX regions
cannot be unlinked; the writer leaves the mapping in place on exit. `shm_resize()`
(used by PT-KVCache) works identically on DAX — it only ever moves data within
the already-mapped window — but the window's total size is fixed by the device,
so make sure it's large enough up front.

---

## Sync Protocol

Every workload coordinates through a shared `sync_ready` int64 scalar:

| Value | Meaning |
| :---: | :--- |
| `0` | Writer initializing |
| `1` | Weights ready; reader may attach |
| `2` | Reader finished validation; writer may exit |

PT-KVCache layers a second, finer-grained protocol on top of this (see
[section 4](#4-pt-kvcache--dynamically-growing-kv-cache)): a `decode_start`
handshake before any tokens are generated, and a per-layer
`[version, current_length]` pair used as an optimistic-concurrency lock around
the growing K/V buffers.

---

## Timing

`benchmark/common/timing.py` provides `TimingRecorder`, a small
`perf_counter`-based utility used by every writer/reader:

```python
recorder = TimingRecorder(label="my writer")
with recorder.time("alloc", tensor_name):
    ...
recorder.print_summary()   # prints n / total / mean / p50 / p99 / max per op, in ms
```

`benchmark/common/shm_weights.py`'s `alloc_and_write_tensor()` /
`lookup_and_attach_tensor()` (and the `copy_state_dict_to_shm()` /
`attach_state_dict_from_shm()` helpers built on top of them) accept an optional
`recorder=` and tag every sample with the tensor's name, so the printed summary
breaks latency down by operation (`alloc`, `copy`, `lookup`, and for PT-KVCache
`resize`/`write`/`wait`/`read`).

---

## Known Issues Found and Fixed During Development

### 1. `shm_resize()` dropping data after a coalesce-without-split (C allocator)

Building PT-KVCache — the first workload in this repo to actually call
`shm_resize()` repeatedly under realistic fragmentation — surfaced a real bug
in the vendored allocator (`allocator/src/shm_alloc.c`):

**Bug:** `freelist_insert()` coalesces adjacent free blocks by growing the
absorbing block's `total_size`, but never refreshes its `payload_cap`. When
`heap_alloc_block()` later hands out that coalesced block *without* splitting
it further (the leftover remainder was smaller than `BLOCK_MIN`), the
returned block kept its **stale, pre-coalesce** `payload_cap`. A subsequent
`shm_resize()` relocation of that same object then used that stale, too-small
`payload_cap` as its `memcpy()` length — silently truncating the copy and
dropping any payload bytes written past the stale capacity.

**Fix:** `heap_alloc_block()`'s "absorb without splitting" branch now resets
`payload_cap = total_size - sizeof(shm_block_hdr_t)` before returning the
block, matching what the split branch already did.

**Regression test:** `allocator/tests/test_resize.c` gained
`test_grow_relocation_after_coalesce_absorb()`, which deliberately engineers
this exact scenario (allocate → free two adjacent neighbors so they coalesce →
grow into the coalesced block without a split → pin its old slot so the next
grow is forced to relocate again) and asserts both halves of previously
written data survive. Verified to fail without the fix and pass with it.

### 2. Intermittent "region too small for N guard pages" on region creation (Python bindings)

The automated test suite (below) occasionally - roughly 1 in 5 full runs -
failed a *different*, randomly-chosen workload each time with a nonsensical
`shm_alloc: region <size> too small for <huge garbage number> guard pages`
error from `shm_region_open()`.

**Bug:** `shm_alloc_python.py`'s `ShmRegionOpenOpts` ctypes struct was missing
the C struct's trailing `guard_pages` field (`shm_region_open_opts_t` in
`allocator/include/shm_alloc.h` has five fields; the Python mirror only
declared four). `ctypes.Structure` sizes itself to exactly its declared
fields, so the C side - which casts a pointer to the *real*, five-field
struct - read `opts->guard_pages` from memory past the end of what Python
had allocated: undefined behavior that usually happened to read zero, and
occasionally read adjacent heap garbage as a giant `guard_pages` count,
which then failed the region's size sanity check for no real reason.

**Fix:** added the missing `("guard_pages", ctypes.c_uint32)` field so the
Python struct's layout matches the C struct field-for-field.

**Verification:** ran the full `tests/test_benchmarks.py` suite five
consecutive times after the fix (0 failures across 40 workload runs), versus
two failures - each on a different, non-reproducible workload - across the
handful of runs immediately before the fix.

---

## Automated Testing

`tests/test_benchmarks.py` runs the classic smoke test and all seven benchmark
workloads as real writer/reader subprocess pairs and asserts clean exit codes
plus a `SUCCESS` marker in the reader's output:

```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
python3 tests/test_benchmarks.py
```

It polls for each writer's `/dev/shm` region to actually appear rather than
sleeping a fixed amount, so it adapts to machine load instead of being flaky
under it. Uses only the standard library (`unittest`) — no extra dependencies.

---

## Extending to Production Scale

To approach full LLaMA-7B / 8.16 GB DLRM footprint:

1. Increase dimensions in each workload's `default_config()` inside `model.py`.
2. Set `--region-size-gb` large enough for the weight footprint.
3. Use FP16 (`pt_dense`, `pt_multitenant`, `pt_multireader`) or FP32
   (`pt_sparse`, `pt_kvcache`, `pt_quantized`'s reference model, `pt_vision`)
   consistently; PT-Quantized's shared tensors are always int8 + fp32
   regardless of the reference model's working dtype.

The `weights_loader.py` module for each workload is the single place to adjust
initialization, SHM object naming, and which tensors are shared vs local.
