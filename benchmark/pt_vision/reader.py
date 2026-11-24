import argparse

import torch

from benchmark.common.shm_weights import (
    add_backend_args,
    open_reader_region,
    signal_reader_done,
    wait_for_writer,
)
from benchmark.common.timing import TimingRecorder
from benchmark.pt_vision.weights_loader import (
    attach_weights_from_shm,
    build_model,
    initialize_weights,
    local_image_batch,
    reference_forward,
)


def main():
    parser = argparse.ArgumentParser(
        description="PT-Vision reader: attach shared conv weights and run an image forward pass."
    )
    add_backend_args(parser)
    parser.add_argument(
        "--keep-writer-alive",
        action="store_true",
        help="Skip signal_reader_done() so the writer keeps waiting instead of "
        "exiting and unlinking the region -- lets this reader be run repeatedly "
        "against the same still-alive writer.",
    )
    args = parser.parse_args()

    region, use_dax, path = open_reader_region(args)
    print(f"PT-Vision reader: attached to backend={args.backend} path={path} dax={use_dax}")

    sync_id, sync_offset = region.lookup_by_name("sync_ready")
    sync_tensor = region.get_tensor(sync_offset, [1], torch.int64, writable=True)
    wait_for_writer(sync_tensor)

    model = build_model()
    recorder = TimingRecorder(label="PT-Vision reader")
    attach_weights_from_shm(region, model, recorder=recorder)
    print("PT-Vision reader: shared conv/fc weights attached (read-only views).")
    recorder.print_summary()

    x = local_image_batch()
    print(f"PT-Vision reader: running forward pass with local image batch {tuple(x.shape)}...")
    with torch.no_grad():
        out = model(x)

    ref_model = build_model()
    initialize_weights(ref_model)
    expected = reference_forward(ref_model, x)
    if torch.allclose(out, expected, rtol=1e-4, atol=1e-4):
        print("PT-Vision reader: SUCCESS - conv forward pass validated.")
    else:
        max_diff = (out - expected).abs().max().item()
        raise RuntimeError(f"PT-Vision reader: FAILURE - max diff {max_diff}")

    if not args.keep_writer_alive:
        signal_reader_done(sync_tensor)
    region.close(unlink=False)


if __name__ == "__main__":
    main()
