import argparse

import torch

from benchmark.common.shm_weights import (
    add_backend_args,
    open_reader_region,
    signal_reader_done,
    wait_for_writer,
)
from benchmark.common.timing import TimingRecorder
from benchmark.pt_dense.model import default_config
from benchmark.pt_dense.weights_loader import (
    attach_weights_from_shm,
    build_model,
    initialize_weights,
    local_activation_input,
    reference_forward,
)


def main():
    parser = argparse.ArgumentParser(
        description="PT-Dense reader: attach shared LLaMA-style weights and run GEMM forward pass."
    )
    add_backend_args(parser)
    args = parser.parse_args()

    region, use_dax, path = open_reader_region(args)
    print(f"PT-Dense reader: attached to backend={args.backend} path={path} dax={use_dax}")

    sync_id, sync_offset = region.lookup_by_name("sync_ready")
    sync_tensor = region.get_tensor(sync_offset, [1], torch.int64, writable=True)
    wait_for_writer(sync_tensor)

    model = build_model()
    recorder = TimingRecorder(label="PT-Dense reader")
    attach_weights_from_shm(region, model, recorder=recorder)
    print("PT-Dense reader: shared weights attached (read-only views).")
    recorder.print_summary()

    x = local_activation_input()
    print(f"PT-Dense reader: running forward pass with local activations {tuple(x.shape)}...")
    with torch.no_grad():
        out = model(x)

    ref_model = build_model()
    initialize_weights(ref_model)
    expected = reference_forward(ref_model, x)
    if torch.allclose(out, expected, rtol=1e-2, atol=1e-2):
        print("PT-Dense reader: SUCCESS - dense GEMM forward pass validated.")
    else:
        max_diff = (out - expected).abs().max().item()
        raise RuntimeError(f"PT-Dense reader: FAILURE - max diff {max_diff}")

    signal_reader_done(sync_tensor)
    region.close(unlink=False)


if __name__ == "__main__":
    main()
