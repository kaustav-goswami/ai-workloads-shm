import argparse

from benchmark.common.shm_weights import (
    add_backend_args,
    allocate_sync,
    open_writer_region,
    signal_writer_ready,
    wait_for_reader,
)
from benchmark.common.timing import TimingRecorder
from benchmark.pt_multireader.weights_loader import (
    build_model,
    initialize_weights,
    required_region_bytes,
    write_weights_to_shm,
)


def main():
    parser = argparse.ArgumentParser(
        description="PT-MultiReader writer: writes one shared backbone that many "
        "concurrent reader processes will attach to at once."
    )
    add_backend_args(parser)
    args = parser.parse_args()

    print("PT-MultiReader writer: building shared backbone and initializing weights...")
    model = build_model()
    initialize_weights(model)
    weight_bytes = required_region_bytes(model)

    region, use_dax, path = open_writer_region(args, weight_bytes)
    print(f"PT-MultiReader writer: backend={args.backend} path={path}")

    recorder = TimingRecorder(label="PT-MultiReader writer")
    names = write_weights_to_shm(region, model, recorder=recorder)
    print(f"PT-MultiReader writer: wrote {len(names)} shared backbone tensors.")
    recorder.print_summary()

    sync_tensor = allocate_sync(region)
    signal_writer_ready(sync_tensor)
    print("PT-MultiReader writer: waiting for all concurrent readers to finish...")

    try:
        wait_for_reader(sync_tensor)
        print("PT-MultiReader writer: all readers completed validation.")
    except KeyboardInterrupt:
        print("PT-MultiReader writer: interrupted.")
    finally:
        region.close(unlink=True)


if __name__ == "__main__":
    main()
