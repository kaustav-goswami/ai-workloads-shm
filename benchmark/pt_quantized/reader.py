import argparse

import torch

from benchmark.common.shm_weights import (
    add_backend_args,
    open_reader_region,
    signal_reader_done,
    wait_for_writer,
)
from benchmark.common.timing import TimingRecorder
from benchmark.pt_quantized.weights_loader import (
    attach_weights_from_shm,
    build_quantized_model,
    build_reference_model,
    local_activation_input,
    quantize_model,
    reference_forward,
)


def main():
    parser = argparse.ArgumentParser(
        description="PT-Quantized reader: attach shared int8/scale weights and validate "
        "against an independently-quantized local reference."
    )
    add_backend_args(parser)
    args = parser.parse_args()

    region, use_dax, path = open_reader_region(args)
    print(f"PT-Quantized reader: attached to backend={args.backend} path={path}")

    _, sync_offset = region.lookup_by_name("sync_ready")
    sync_tensor = region.get_tensor(sync_offset, [1], torch.int64, writable=True)
    wait_for_writer(sync_tensor)

    quant_model = build_quantized_model()
    recorder = TimingRecorder(label="PT-Quantized reader")
    attach_weights_from_shm(region, quant_model, recorder=recorder)
    print("PT-Quantized reader: shared int8 qweight + fp32 scale tensors attached (read-only).")
    recorder.print_summary()

    x = local_activation_input()
    print(f"PT-Quantized reader: running forward pass with local activations {tuple(x.shape)}...")
    with torch.no_grad():
        out = quant_model(x)

    # Independent local quantization: rebuild the fp32 reference (same
    # deterministic seeds PT-Dense's init uses) and quantize it exactly the
    # same way, entirely without SHM. If the SHM-attached int8/scale tensors
    # were transferred correctly, this must match the SHM-driven output
    # almost exactly - divergence here means the shared int8/scale bytes
    # were corrupted or mismatched, not just "quantization noise".
    local_ref = build_reference_model()
    local_quant = build_quantized_model()
    quantize_model(local_quant, local_ref)
    expected = reference_forward(local_quant, x)

    if torch.allclose(out, expected, rtol=1e-4, atol=1e-4):
        max_diff = (out - expected).abs().max().item()
        print(
            f"PT-Quantized reader: SUCCESS - SHM-driven int8 forward matches independent "
            f"local quantization (max diff {max_diff:.3e})."
        )
    else:
        max_diff = (out - expected).abs().max().item()
        raise RuntimeError(f"PT-Quantized reader: FAILURE - max diff {max_diff}")

    # Purely informational: how much error does int8 quantization itself
    # introduce relative to the true fp32 model (not a pass/fail check).
    fp32_ref = build_reference_model()
    fp32_out = reference_forward(fp32_ref, x)
    quant_error = (out - fp32_out).abs().max().item()
    rel_error = quant_error / fp32_out.abs().max().item()
    print(
        f"PT-Quantized reader: int8 quantization error vs true fp32 - "
        f"max abs diff {quant_error:.4f} ({rel_error * 100:.2f}% of peak fp32 magnitude)."
    )

    signal_reader_done(sync_tensor)
    region.close(unlink=False)


if __name__ == "__main__":
    main()
