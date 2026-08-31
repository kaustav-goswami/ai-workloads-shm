import argparse
import multiprocessing
import time

import torch

from benchmark.common.shm_weights import (
    add_backend_args,
    open_reader_region,
    signal_reader_done,
    wait_for_writer,
)
from benchmark.common.timing import TimingRecorder
from benchmark.pt_multireader.model import default_config
from benchmark.pt_multireader.weights_loader import (
    attach_weights_from_shm,
    build_model,
    initialize_weights,
    local_activation_input,
    reference_forward,
)


def reader_worker(reader_id, backend, path, result_queue):
    """Runs in its own process: independently attaches the shared backbone
    from SHM, validates a forward pass, and reports latency back to the
    parent. Exercises N processes doing concurrent shm_lookup_by_name /
    shm_ptr calls against the same live region.
    """
    try:
        start = time.perf_counter()

        class _Args:
            pass

        args = _Args()
        args.backend = backend
        args.path = path
        region, _, _ = open_reader_region(args)

        _, sync_offset = region.lookup_by_name("sync_ready")
        sync_tensor = region.get_tensor(sync_offset, [1], torch.int64, writable=True)
        wait_for_writer(sync_tensor)

        model = build_model()
        recorder = TimingRecorder(label=f"reader-{reader_id}")
        attach_start = time.perf_counter()
        attach_weights_from_shm(region, model, recorder=recorder)
        attach_ms = (time.perf_counter() - attach_start) * 1000

        x = local_activation_input()
        with torch.no_grad():
            out = model(x)

        ref_model = build_model()
        initialize_weights(ref_model)
        expected = reference_forward(ref_model, x)
        ok = torch.allclose(out, expected, rtol=1e-2, atol=1e-2)
        max_diff = (out - expected).abs().max().item()

        region.close(unlink=False)
        total_ms = (time.perf_counter() - start) * 1000
        result_queue.put(
            {
                "reader_id": reader_id,
                "ok": ok,
                "max_diff": max_diff,
                "attach_ms": attach_ms,
                "total_ms": total_ms,
                "lookup_stats": recorder.stats("lookup"),
                "error": None,
            }
        )
    except Exception as exc:  # noqa: BLE001 - report failures across the process boundary
        result_queue.put({"reader_id": reader_id, "ok": False, "error": repr(exc)})


def main():
    parser = argparse.ArgumentParser(
        description="PT-MultiReader reader: spawns N concurrent reader processes that "
        "all attach the same shared backbone at once."
    )
    add_backend_args(parser)
    parser.add_argument(
        "--num-readers",
        type=int,
        default=None,
        help="Override the number of concurrent reader processes from default_config().",
    )
    args = parser.parse_args()

    config = default_config()
    num_readers = args.num_readers if args.num_readers is not None else config["num_readers"]

    # Block here until the writer has published weights, so every spawned
    # worker starts racing for shm_lookup_by_name at (approximately) the same
    # instant instead of trickling in one at a time.
    region, use_dax, path = open_reader_region(args)
    print(f"PT-MultiReader reader: attached to backend={args.backend} path={path}")
    _, sync_offset = region.lookup_by_name("sync_ready")
    sync_tensor = region.get_tensor(sync_offset, [1], torch.int64, writable=True)
    wait_for_writer(sync_tensor)
    region.close(unlink=False)

    print(f"PT-MultiReader reader: launching {num_readers} concurrent reader processes...")
    ctx = multiprocessing.get_context("fork")
    result_queue = ctx.Queue()
    workers = [
        ctx.Process(target=reader_worker, args=(i, args.backend, args.path, result_queue))
        for i in range(num_readers)
    ]

    wall_start = time.perf_counter()
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    wall_ms = (time.perf_counter() - wall_start) * 1000

    results = [result_queue.get() for _ in workers]
    results.sort(key=lambda r: r["reader_id"])

    print(f"\nPT-MultiReader reader: all {num_readers} readers finished in {wall_ms:.2f} ms wall time.")
    print(f"{'reader':<8}{'ok':<6}{'max_diff':<12}{'attach_ms':<12}{'total_ms':<10}")
    all_ok = True
    for r in results:
        if r.get("error"):
            all_ok = False
            print(f"{r['reader_id']:<8}{'ERR':<6}{r['error']}")
            continue
        all_ok = all_ok and r["ok"]
        print(
            f"{r['reader_id']:<8}{str(r['ok']):<6}{r['max_diff']:<12.3e}"
            f"{r['attach_ms']:<12.4f}{r['total_ms']:<10.4f}"
        )

    if not all_ok:
        raise RuntimeError("PT-MultiReader reader: FAILURE - at least one concurrent reader failed")

    attach_times = [r["attach_ms"] for r in results if not r.get("error")]
    print(
        f"\nPT-MultiReader reader: attach latency across {len(attach_times)} concurrent readers - "
        f"min={min(attach_times):.4f}ms max={max(attach_times):.4f}ms "
        f"mean={sum(attach_times) / len(attach_times):.4f}ms"
    )
    print("PT-MultiReader reader: SUCCESS - all concurrent readers attached and validated independently.")

    region, use_dax, path = open_reader_region(args)
    _, sync_offset = region.lookup_by_name("sync_ready")
    sync_tensor = region.get_tensor(sync_offset, [1], torch.int64, writable=True)
    signal_reader_done(sync_tensor)
    region.close(unlink=False)


if __name__ == "__main__":
    main()
