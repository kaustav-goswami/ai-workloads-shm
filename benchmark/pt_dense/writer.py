import argparse
import time

from benchmark.common.shm_weights import (
    add_backend_args,
    allocate_sync,
    open_writer_region,
    signal_writer_ready,
    wait_for_reader,
)
from benchmark.common.timing import TimingRecorder
from benchmark.pt_dense.weights_loader import (
    build_model,
    initialize_weights,
    required_region_bytes,
    write_weights_to_shm,
)


def main():
    parser = argparse.ArgumentParser(
        description="PT-Dense writer: load monolithic LLaMA-style weights into shared memory."
    )
    add_backend_args(parser)
    args = parser.parse_args()

    print("PT-Dense writer: building local model and initializing weights...")
    model = build_model()
    initialize_weights(model)
    weight_bytes = required_region_bytes(model)

    print(
        f"PT-Dense writer: opening shared region ({weight_bytes / (1024 * 1024):.2f} MiB weights)..."
    )
    region, use_dax, path = open_writer_region(args, weight_bytes)
    print(f"PT-Dense writer: backend={args.backend} path={path} dax={use_dax}")

    recorder = TimingRecorder(label="PT-Dense writer")
    names = write_weights_to_shm(region, model, recorder=recorder)
    print(f"PT-Dense writer: wrote {len(names)} monolithic weight tensors to shared memory.")
    recorder.print_summary()

    sync_tensor = allocate_sync(region)
    signal_writer_ready(sync_tensor)
    print("PT-Dense writer: waiting for reader process...")

    try:
        wait_for_reader(sync_tensor)
        print("PT-Dense writer: reader completed validation.")
    except KeyboardInterrupt:
        print("PT-Dense writer: interrupted.")
    finally:
        region.close(unlink=True)


if __name__ == "__main__":
    main()
