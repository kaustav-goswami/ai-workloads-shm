import argparse

from benchmark.common.shm_weights import (
    add_backend_args,
    allocate_sync,
    open_writer_region,
    signal_writer_ready,
    wait_for_reader,
)
from benchmark.common.timing import TimingRecorder
from benchmark.pt_quantized.weights_loader import (
    build_quantized_model,
    build_reference_model,
    quantize_model,
    required_region_bytes,
    write_weights_to_shm,
)


def main():
    parser = argparse.ArgumentParser(
        description="PT-Quantized writer: quantize an fp32 LLaMA-style decoder to "
        "per-channel int8 + fp32 scale and write the quantized tensors to shared memory."
    )
    add_backend_args(parser)
    args = parser.parse_args()

    print("PT-Quantized writer: building fp32 reference model and quantizing to int8...")
    ref_model = build_reference_model()
    quant_model = build_quantized_model()
    quantize_model(quant_model, ref_model)

    fp32_bytes = sum(t.numel() * t.element_size() for t in ref_model.state_dict().values())
    quant_bytes = required_region_bytes(quant_model)
    print(
        f"PT-Quantized writer: fp32 footprint={fp32_bytes / (1024 * 1024):.2f} MiB, "
        f"quantized footprint={quant_bytes / (1024 * 1024):.2f} MiB "
        f"({fp32_bytes / quant_bytes:.2f}x smaller)."
    )

    region, use_dax, path = open_writer_region(args, quant_bytes)
    print(f"PT-Quantized writer: backend={args.backend} path={path}")

    recorder = TimingRecorder(label="PT-Quantized writer")
    names = write_weights_to_shm(region, quant_model, recorder=recorder)
    print(f"PT-Quantized writer: wrote {len(names)} tensors (int8 qweight + fp32 scale per projection).")
    recorder.print_summary()

    sync_tensor = allocate_sync(region)
    signal_writer_ready(sync_tensor)
    print("PT-Quantized writer: waiting for reader process...")

    try:
        wait_for_reader(sync_tensor)
        print("PT-Quantized writer: reader completed validation.")
    except KeyboardInterrupt:
        print("PT-Quantized writer: interrupted.")
    finally:
        region.close(unlink=True)


if __name__ == "__main__":
    main()
