import argparse

from benchmark.common.shm_weights import (
    add_backend_args,
    allocate_sync,
    open_writer_region,
    signal_writer_ready,
    wait_for_reader,
)
from benchmark.common.timing import TimingRecorder
from benchmark.pt_multitenant.weights_loader import (
    build_model,
    initialize_weights,
    required_region_bytes,
    write_weights_to_shm,
)


def main():
    parser = argparse.ArgumentParser(
        description="PT-MultiTenant writer: shared backbone plus segmented LoRA/MoE weights."
    )
    add_backend_args(parser)
    args = parser.parse_args()

    print("PT-MultiTenant writer: building hybrid model...")
    model = build_model()
    initialize_weights(model)
    weight_bytes = required_region_bytes(model)

    region, use_dax, path = open_writer_region(args, weight_bytes)
    print(f"PT-MultiTenant writer: backend={args.backend} path={path}")

    recorder = TimingRecorder(label="PT-MultiTenant writer")
    groups = write_weights_to_shm(region, model, recorder=recorder)
    print(
        "PT-MultiTenant writer: wrote "
        f"{len(groups['backbone'])} backbone, "
        f"{len(groups['adapters'])} LoRA, "
        f"{len(groups['experts'])} expert tensors."
    )
    recorder.print_summary()

    sync_tensor = allocate_sync(region)
    signal_writer_ready(sync_tensor)
    print("PT-MultiTenant writer: waiting for reader process...")

    try:
        wait_for_reader(sync_tensor)
        print("PT-MultiTenant writer: reader completed validation.")
    except KeyboardInterrupt:
        print("PT-MultiTenant writer: interrupted.")
    finally:
        region.close(unlink=True)


if __name__ == "__main__":
    main()
