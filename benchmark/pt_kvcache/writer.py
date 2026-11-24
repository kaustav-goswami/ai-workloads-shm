import argparse
import time

import torch

from benchmark.common.shm_weights import (
    add_backend_args,
    allocate_sync,
    open_writer_region,
    signal_writer_ready,
    wait_for_reader,
)
from benchmark.common.timing import TimingRecorder
from benchmark.pt_kvcache.model import default_config
from benchmark.pt_kvcache.weights_loader import (
    build_model,
    initialize_weights,
    kv_k_name,
    kv_meta_name,
    kv_shm_shape,
    kv_tensor_nbytes,
    kv_v_name,
    required_region_bytes,
    to_shm_layout,
    token_hidden_state,
    write_weights_to_shm,
)


def main():
    parser = argparse.ArgumentParser(
        description="PT-KVCache writer: multi-layer incremental decode that grows a "
        "per-layer K/V cache in shared memory one token at a time via shm_resize."
    )
    add_backend_args(parser)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override the number of decode steps from default_config().",
    )
    parser.add_argument(
        "--presize",
        action="store_true",
        help="Allocate each layer's K/V cache once, sized for --max-new-tokens up "
        "front, and write into a fixed slot per step instead of calling "
        "region.resize() every step. Required on this v0.3 engine build: "
        "resize() is unconditionally unimplemented here (see "
        "shm_alloc_python.py's SHMRegion.resize()) -- this is not a smaller "
        "version of the same measurement, it measures a fixed-size zero-copy "
        "write instead of incremental resize-and-copy growth.",
    )
    args = parser.parse_args()

    config = default_config()
    if args.max_new_tokens is not None:
        config["max_new_tokens"] = args.max_new_tokens

    print("PT-KVCache writer: building incremental decoder and initializing weights...")
    model = build_model(config)
    initialize_weights(model)
    weight_bytes = required_region_bytes(model)

    region, use_dax, path = open_writer_region(args, weight_bytes, extra_bytes=64 * 1024 * 1024)
    print(f"PT-KVCache writer: backend={args.backend} path={path}")

    weight_recorder = TimingRecorder(label="PT-KVCache writer (static weights)")
    names = write_weights_to_shm(region, model, recorder=weight_recorder)
    print(f"PT-KVCache writer: wrote {len(names)} static attention/MLP tensors.")
    weight_recorder.print_summary()

    num_layers = config["num_layers"]
    dtype = config["dtype"]

    kv_state = []
    for layer_idx in range(num_layers):
        # Presized: allocate the FULL final buffer up front (one d_publish),
        # since this engine has no resize primitive at all. Non-presized:
        # keep the original length=1 starting allocation for when resize()
        # is restored.
        nbytes = kv_tensor_nbytes(config, config["max_new_tokens"] if args.presize else 1)
        k_id, k_off = region.alloc(kv_k_name(layer_idx), nbytes)
        v_id, v_off = region.alloc(kv_v_name(layer_idx), nbytes)
        meta_id, meta_off = region.alloc(kv_meta_name(layer_idx), 16)
        meta = region.get_tensor(meta_off, [2], torch.int64, writable=True)
        meta[0] = 0  # version: even == stable
        meta[1] = 0  # current_length (tokens committed so far)
        kv_state.append({"k_id": k_id, "v_id": v_id, "k_off": k_off, "v_off": v_off, "meta": meta})
    print(f"PT-KVCache writer: allocated {num_layers} {'presized' if args.presize else 'empty'} per-layer K/V caches.")

    decode_start_id, decode_start_off = region.alloc("decode_start", 8)
    decode_start = region.get_tensor(decode_start_off, [1], torch.int64, writable=True)
    decode_start[0] = 0

    sync_tensor = allocate_sync(region)
    signal_writer_ready(sync_tensor)
    print("PT-KVCache writer: static weights + empty KV caches ready; waiting for reader to attach...")

    while decode_start[0].item() != 1:
        time.sleep(0.02)
    print(
        f"PT-KVCache writer: reader attached; decoding {config['max_new_tokens']} tokens "
        f"across {num_layers} layers with a growing SHM KV cache..."
    )

    label = "PT-KVCache writer (kv presized write)" if args.presize else "PT-KVCache writer (kv resize + write)"
    kv_recorder = TimingRecorder(label=label)
    kv_caches = [(None, None) for _ in range(num_layers)]

    if args.presize:
        # Full-size view of each layer's fixed buffer, resolved once since
        # the offset never moves (no resize -> no relocation).
        full_shape = kv_shm_shape(config, config["max_new_tokens"])
        for state in kv_state:
            state["k_full"] = region.get_tensor(state["k_off"], full_shape, dtype, writable=True)
            state["v_full"] = region.get_tensor(state["v_off"], full_shape, dtype, writable=True)

    for step in range(config["max_new_tokens"]):
        x = token_hidden_state(step, config)
        new_length = step + 1
        for layer_idx, layer in enumerate(model.layers):
            k_cache, v_cache = kv_caches[layer_idx]
            with torch.no_grad():
                x, k_all, v_all = layer.step(x, k_cache, v_cache)
            kv_caches[layer_idx] = (k_all, v_all)

            state = kv_state[layer_idx]
            meta = state["meta"]

            if args.presize:
                with kv_recorder.time("write", f"layer{layer_idx}"):
                    meta[0] = int(meta[0].item()) + 1  # odd: update in progress
                    # Fixed slot `step` (not the always-last-row append the
                    # resize path uses) -- the buffer is already full length.
                    state["k_full"][step : step + 1, :, :] = to_shm_layout(k_all[:, :, -1:, :])
                    state["v_full"][step : step + 1, :, :] = to_shm_layout(v_all[:, :, -1:, :])
                    meta[1] = new_length
                    meta[0] = int(meta[0].item()) + 1  # even: stable again
            else:
                with kv_recorder.time("resize", f"layer{layer_idx}"):
                    meta[0] = int(meta[0].item()) + 1  # odd: update in progress
                    nbytes = kv_tensor_nbytes(config, new_length)
                    k_off, _ = region.resize(state["k_id"], nbytes)
                    v_off, _ = region.resize(state["v_id"], nbytes)
                with kv_recorder.time("write", f"layer{layer_idx}"):
                    shm_shape = kv_shm_shape(config, new_length)
                    k_tensor = region.get_tensor(k_off, shm_shape, dtype, writable=True)
                    v_tensor = region.get_tensor(v_off, shm_shape, dtype, writable=True)
                    # New length is appended as the new *outermost* row, so this
                    # write never disturbs the byte positions of earlier tokens.
                    k_tensor[-1:, :, :] = to_shm_layout(k_all[:, :, -1:, :])
                    v_tensor[-1:, :, :] = to_shm_layout(v_all[:, :, -1:, :])
                    meta[1] = new_length
                    meta[0] = int(meta[0].item()) + 1  # even: stable again

    print(f"PT-KVCache writer: completed {config['max_new_tokens']} decode steps.")
    kv_recorder.print_summary()

    print("PT-KVCache writer: waiting for reader to finish validating the full trajectory...")
    try:
        wait_for_reader(sync_tensor)
        print("PT-KVCache writer: reader completed validation.")
    except KeyboardInterrupt:
        print("PT-KVCache writer: interrupted.")
    finally:
        region.close(unlink=True)


if __name__ == "__main__":
    main()
