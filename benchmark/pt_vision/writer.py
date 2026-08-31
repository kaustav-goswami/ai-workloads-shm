import argparse

from benchmark.common.shm_weights import (
    add_backend_args,
    allocate_sync,
    open_writer_region,
    signal_writer_ready,
    wait_for_reader,
)
from benchmark.common.timing import TimingRecorder
from benchmark.pt_vision.weights_loader import (
    build_model,
    initialize_weights,
    required_region_bytes,
    write_weights_to_shm,
)


def main():
    parser = argparse.ArgumentParser(
        description="PT-Vision writer: load a small conv-only CNN's weights "
        "(a handful of large 4D conv tensors) into shared memory."
    )
    add_backend_args(parser)
    args = parser.parse_args()

    print("PT-Vision writer: building conv model and initializing weights...")
    model = build_model()
    initialize_weights(model)
    weight_bytes = required_region_bytes(model)

    print(f"PT-Vision writer: opening shared region ({weight_bytes / (1024 * 1024):.2f} MiB weights)...")
    region, use_dax, path = open_writer_region(args, weight_bytes)
    print(f"PT-Vision writer: backend={args.backend} path={path} dax={use_dax}")

    recorder = TimingRecorder(label="PT-Vision writer")
    names = write_weights_to_shm(region, model, recorder=recorder)
    print(f"PT-Vision writer: wrote {len(names)} conv/fc tensors to shared memory.")
    recorder.print_summary()

    sync_tensor = allocate_sync(region)
    signal_writer_ready(sync_tensor)
    print("PT-Vision writer: waiting for reader process...")

    try:
        wait_for_reader(sync_tensor)
        print("PT-Vision writer: reader completed validation.")
    except KeyboardInterrupt:
        print("PT-Vision writer: interrupted.")
    finally:
        region.close(unlink=True)


if __name__ == "__main__":
    main()
