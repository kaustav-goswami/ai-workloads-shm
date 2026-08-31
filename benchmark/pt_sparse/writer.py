import argparse

from benchmark.common.shm_weights import (
    add_backend_args,
    allocate_sync,
    open_writer_region,
    signal_writer_ready,
    wait_for_reader,
)
from benchmark.common.timing import TimingRecorder
from benchmark.pt_sparse.weights_loader import (
    build_model,
    initialize_weights,
    required_region_bytes,
    write_weights_to_shm,
)


def main():
    parser = argparse.ArgumentParser(
        description="PT-Sparse writer: load centralized DLRM embedding tables into shared memory."
    )
    add_backend_args(parser)
    args = parser.parse_args()

    print("PT-Sparse writer: building DLRM model...")
    model = build_model()
    initialize_weights(model)
    weight_bytes = required_region_bytes(model)

    region, use_dax, path = open_writer_region(args, weight_bytes)
    print(
        f"PT-Sparse writer: backend={args.backend} path={path} "
        f"shared_tables={weight_bytes / (1024 * 1024):.2f} MiB"
    )

    recorder = TimingRecorder(label="PT-Sparse writer")
    groups = write_weights_to_shm(region, model, recorder=recorder)
    print(
        f"PT-Sparse writer: wrote {len(groups['shared'])} shared embedding tables; "
        f"{len(groups['local'])} dense/top MLP weights remain process-local."
    )
    recorder.print_summary()

    sync_tensor = allocate_sync(region)
    signal_writer_ready(sync_tensor)
    print("PT-Sparse writer: waiting for reader process...")

    try:
        wait_for_reader(sync_tensor)
        print("PT-Sparse writer: reader completed validation.")
    except KeyboardInterrupt:
        print("PT-Sparse writer: interrupted.")
    finally:
        region.close(unlink=True)


if __name__ == "__main__":
    main()
