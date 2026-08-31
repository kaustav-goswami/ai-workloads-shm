import argparse

import torch

from benchmark.common.shm_weights import (
    add_backend_args,
    open_reader_region,
    signal_reader_done,
    wait_for_writer,
)
from benchmark.common.timing import TimingRecorder
from benchmark.pt_sparse.weights_loader import (
    attach_weights_from_shm,
    build_model,
    initialize_weights,
    local_batch,
    reference_forward,
)


def main():
    parser = argparse.ArgumentParser(
        description="PT-Sparse reader: attach shared embedding tables and run EmbeddingBag lookups."
    )
    add_backend_args(parser)
    args = parser.parse_args()

    region, use_dax, path = open_reader_region(args)
    print(f"PT-Sparse reader: attached to backend={args.backend} path={path}")

    _, sync_offset = region.lookup_by_name("sync_ready")
    sync_tensor = region.get_tensor(sync_offset, [1], torch.int64, writable=True)
    wait_for_writer(sync_tensor)

    model = build_model()
    initialize_weights(model)
    recorder = TimingRecorder(label="PT-Sparse reader")
    attach_weights_from_shm(region, model, recorder=recorder)
    print("PT-Sparse reader: shared embedding tables attached; dense/top MLP weights are local.")
    recorder.print_summary()

    dense, sparse_indices, sparse_offsets = local_batch()
    with torch.no_grad():
        out = model(dense, sparse_indices, sparse_offsets)

    ref_model = build_model()
    initialize_weights(ref_model)
    attach_weights_from_shm(region, ref_model)
    expected = reference_forward(ref_model, dense, sparse_indices, sparse_offsets)
    if torch.allclose(out, expected, rtol=1e-4, atol=1e-4):
        print("PT-Sparse reader: SUCCESS - EmbeddingBag sparse lookup validated.")
    else:
        max_diff = (out - expected).abs().max().item()
        raise RuntimeError(f"PT-Sparse reader: FAILURE - max diff {max_diff}")

    signal_reader_done(sync_tensor)
    region.close(unlink=False)


if __name__ == "__main__":
    main()
