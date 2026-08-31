import argparse
import time

import torch

from benchmark.common.shm_weights import (
    add_backend_args,
    open_reader_region,
    signal_reader_done,
    wait_for_writer,
)
from benchmark.common.timing import TimingRecorder
from benchmark.pt_kvcache.model import default_config
from benchmark.pt_kvcache.weights_loader import (
    attach_weights_from_shm,
    build_model,
    from_shm_layout,
    initialize_weights,
    kv_k_name,
    kv_meta_name,
    kv_shm_shape,
    kv_v_name,
    token_hidden_state,
)


def _read_meta(region, layer_idx):
    _, off = region.lookup_by_name(kv_meta_name(layer_idx))
    return region.get_tensor(off, [2], torch.int64, writable=False)


def wait_for_length(region, layer_idx, target_length, poll_seconds=0.01, max_wait_seconds=30.0):
    waited = 0.0
    while True:
        meta = _read_meta(region, layer_idx)
        version = int(meta[0].item())
        length = int(meta[1].item())
        if version % 2 == 0 and length >= target_length:
            return
        time.sleep(poll_seconds)
        waited += poll_seconds
        if waited > max_wait_seconds:
            raise TimeoutError(
                f"PT-KVCache reader: timed out waiting for layer {layer_idx} "
                f"to reach length {target_length}"
            )


def read_kv_consistent(region, layer_idx, length, config, dtype, retries=200):
    """Optimistic-concurrency read: snapshot the cache, then confirm the
    writer's version counter didn't change (and therefore didn't relocate or
    overwrite the block) while we were reading. Retries on a torn read.
    """
    shape = kv_shm_shape(config, length)
    for _ in range(retries):
        meta_before = _read_meta(region, layer_idx)
        version_before = int(meta_before[0].item())
        if version_before % 2 != 0:
            continue  # writer is mid-update; try again
        _, k_off = region.lookup_by_name(kv_k_name(layer_idx))
        _, v_off = region.lookup_by_name(kv_v_name(layer_idx))
        k = region.get_tensor(k_off, shape, dtype, writable=False).clone()
        v = region.get_tensor(v_off, shape, dtype, writable=False).clone()
        version_after = int(_read_meta(region, layer_idx)[0].item())
        if version_after == version_before:
            return from_shm_layout(k), from_shm_layout(v)
    raise RuntimeError(
        f"PT-KVCache reader: could not obtain a consistent read for layer {layer_idx} "
        f"after {retries} retries"
    )


def main():
    parser = argparse.ArgumentParser(
        description="PT-KVCache reader: attach static weights, then poll and validate "
        "the growing per-layer K/V cache one decode step at a time."
    )
    add_backend_args(parser)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Must match the value passed to writer.py.",
    )
    args = parser.parse_args()

    config = default_config()
    if args.max_new_tokens is not None:
        config["max_new_tokens"] = args.max_new_tokens

    region, use_dax, path = open_reader_region(args)
    print(f"PT-KVCache reader: attached to backend={args.backend} path={path}")

    _, sync_offset = region.lookup_by_name("sync_ready")
    sync_tensor = region.get_tensor(sync_offset, [1], torch.int64, writable=True)
    wait_for_writer(sync_tensor)

    model = build_model(config)
    weight_recorder = TimingRecorder(label="PT-KVCache reader (static weights)")
    attach_weights_from_shm(region, model, recorder=weight_recorder)
    print("PT-KVCache reader: static attention/MLP weights attached (read-only).")
    weight_recorder.print_summary()

    num_layers = config["num_layers"]
    dtype = config["dtype"]

    _, decode_start_off = region.lookup_by_name("decode_start")
    decode_start = region.get_tensor(decode_start_off, [1], torch.int64, writable=True)
    decode_start[0] = 1
    print("PT-KVCache reader: signaled writer to start decoding; polling KV cache growth...")

    ref_model = build_model(config)
    initialize_weights(ref_model)

    kv_recorder = TimingRecorder(label="PT-KVCache reader (kv wait + read)")
    ref_kv_caches = [(None, None) for _ in range(num_layers)]
    max_diff_overall = 0.0

    for step in range(config["max_new_tokens"]):
        x_t = token_hidden_state(step, config)
        target_length = step + 1

        shm_caches = []
        for layer_idx in range(num_layers):
            with kv_recorder.time("wait", f"layer{layer_idx}"):
                wait_for_length(region, layer_idx, target_length)
            with kv_recorder.time("read", f"layer{layer_idx}"):
                k, v = read_kv_consistent(region, layer_idx, target_length, config, dtype)
            shm_caches.append((k, v))

        with torch.no_grad():
            shm_out = model.decode_step_from_cache(x_t, shm_caches)
            ref_out, ref_kv_caches = ref_model.decode_step(x_t, ref_kv_caches)

        diff = (shm_out - ref_out).abs().max().item()
        max_diff_overall = max(max_diff_overall, diff)
        if not torch.allclose(shm_out, ref_out, rtol=1e-4, atol=1e-4):
            raise RuntimeError(
                f"PT-KVCache reader: FAILURE at decode step {step} - max diff {diff} "
                "(SHM-driven output diverged from the independent local reference)"
            )

    print(
        f"PT-KVCache reader: validated all {config['max_new_tokens']} decode steps "
        f"across {num_layers} layers (max diff observed: {max_diff_overall:.3e})."
    )
    kv_recorder.print_summary()
    print("PT-KVCache reader: SUCCESS - dynamic SHM KV cache growth validated end-to-end.")

    signal_reader_done(sync_tensor)
    region.close(unlink=False)


if __name__ == "__main__":
    main()
