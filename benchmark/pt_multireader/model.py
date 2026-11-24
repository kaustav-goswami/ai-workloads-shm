import argparse
import threading
import time

import torch

from benchmark.common.seeds import deterministic_seed
from benchmark.common.timing import TimingRecorder
from benchmark.pt_dense.model import LlamaDenseModel

# PT-MultiReader validates concurrent *reader* access, not model size, so it
# reuses PT-Dense's architecture at a smaller scale.
SharedBackboneModel = LlamaDenseModel


def default_config():
    return {
        "hidden_size": 256,
        "num_layers": 3,
        "num_heads": 8,
        "intermediate_size": 688,
        "dtype": torch.float16,
        "batch_size": 2,
        "seq_len": 8,
        "num_readers": 4,
    }


def build_model(config=None):
    config = default_config() if config is None else config
    model = SharedBackboneModel(
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        intermediate_size=config["intermediate_size"],
    )
    return model.to(config["dtype"])


def initialize_weights(model):
    with torch.no_grad():
        for name, param in model.named_parameters():
            generator = torch.Generator(device="cpu")
            generator.manual_seed(deterministic_seed(name))
            if "norm" in name or (name.endswith(".weight") and param.ndim == 1):
                param.fill_(1.0)
            else:
                param.copy_(torch.randn(param.shape, generator=generator, dtype=param.dtype) * 0.02)


def local_activation_input(config, reader_id):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(deterministic_seed(f"reader_input_{reader_id}"))
    return torch.randn(
        config["batch_size"],
        config["seq_len"],
        config["hidden_size"],
        generator=generator,
        dtype=config["dtype"],
    )


def reader_worker(reader_id, model, config, results, lock, recorder):
    """Runs in its own thread against the SAME model instance. PyTorch's
    default allocator keeps the model's weight tensors in ordinary process
    memory, so every thread in this process already sees them without any
    explicit attach/copy step - this is what zero-copy sharing looks like
    without a custom shared-memory allocator.
    """
    start = time.perf_counter()
    x = local_activation_input(config, reader_id)
    with torch.no_grad():
        out = model(x)
    elapsed_s = time.perf_counter() - start
    elapsed_ms = elapsed_s * 1000
    with lock:
        recorder.record("forward", elapsed_s, item=f"reader{reader_id}")
        results.append(
            {
                "reader_id": reader_id,
                "elapsed_ms": elapsed_ms,
                "out_mean": out.float().mean().item(),
            }
        )


def main():
    parser = argparse.ArgumentParser(
        description="PT-MultiReader: N concurrent reader threads running forward passes "
        "against one shared model instance allocated with PyTorch's default CPU allocator."
    )
    parser.add_argument(
        "--num-readers",
        type=int,
        default=None,
        help="Override the number of concurrent reader threads from default_config().",
    )
    args = parser.parse_args()

    config = default_config()
    num_readers = args.num_readers if args.num_readers is not None else config["num_readers"]

    print("PT-MultiReader: building shared backbone once...")
    model = build_model(config)
    initialize_weights(model)
    model.eval()

    print(f"PT-MultiReader: launching {num_readers} concurrent reader threads...")
    results = []
    lock = threading.Lock()
    recorder = TimingRecorder(label="PT-MultiReader")
    threads = [
        threading.Thread(target=reader_worker, args=(i, model, config, results, lock, recorder))
        for i in range(num_readers)
    ]

    wall_start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall_ms = (time.perf_counter() - wall_start) * 1000

    results.sort(key=lambda r: r["reader_id"])
    print(f"\nPT-MultiReader: all {num_readers} readers finished in {wall_ms:.2f} ms wall time.")
    print(f"{'reader':<8}{'elapsed_ms':<12}{'out_mean':<12}")
    for r in results:
        print(f"{r['reader_id']:<8}{r['elapsed_ms']:<12.4f}{r['out_mean']:<12.6f}")
    recorder.print_summary()

    elapsed_times = [r["elapsed_ms"] for r in results]
    print(
        f"\nPT-MultiReader: latency across {len(elapsed_times)} concurrent readers - "
        f"min={min(elapsed_times):.4f}ms max={max(elapsed_times):.4f}ms "
        f"mean={sum(elapsed_times) / len(elapsed_times):.4f}ms"
    )
    print(
        "PT-MultiReader: SUCCESS - all concurrent readers ran forward passes against "
        "the shared model weights."
    )


if __name__ == "__main__":
    main()
