"""Automated correctness tests for every writer/reader workload pair in this repo.

Each test launches a workload's writer.py as a background process, waits for
its POSIX shared-memory region to actually appear under /dev/shm (rather than
a fixed sleep - process startup time, especially the first torch import in a
fresh process, varies with machine load), then runs the matching reader.py to
completion and checks that both sides exited cleanly and the reader printed
its own "SUCCESS" marker.

Run with:
    PYTHONPATH=. python3 tests/test_benchmarks.py
or:
    PYTHONPATH=. python3 -m unittest discover -s tests -p 'test_*.py' -v

Requires the allocator shared library to already be built:
    cd allocator && gcc -shared -fPIC -std=c11 -Wall -Wextra -g \\
        -D_POSIX_C_SOURCE=200809L -Iinclude -Isrc \\
        src/shm_alloc.c src/shm_ns.c -o build/libshm_alloc.so -pthread -lrt
"""

import os
import subprocess
import sys
import time
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SHM_DIR = Path("/dev/shm")
LIB_PATH = REPO_ROOT / "allocator" / "build" / "libshm_alloc.so"


def _env():
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{existing}" if existing else str(REPO_ROOT)
    return env


def _wait_for_shm_path(shm_name, timeout, poll=0.05):
    target = SHM_DIR / shm_name
    deadline = time.time() + timeout
    while time.time() < deadline:
        if target.exists():
            return True
        time.sleep(poll)
    return target.exists()


def run_workload_pair(
    test_case,
    name,
    writer_argv,
    reader_argv,
    shm_path,
    writer_timeout=30,
    reader_timeout=60,
):
    """Runs one writer/reader pair end to end and asserts success.

    `shm_path` is the POSIX shm name passed to both sides via --path (e.g.
    "/pt_dense_test_1234"), used to detect writer readiness and to clean up.
    """
    shm_name = shm_path.lstrip("/")
    leftover = SHM_DIR / shm_name
    if leftover.exists():
        leftover.unlink()

    env = _env()
    writer_proc = subprocess.Popen(
        [sys.executable] + writer_argv,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        appeared = _wait_for_shm_path(shm_name, timeout=writer_timeout)
        if not appeared:
            writer_proc.kill()
            out, _ = writer_proc.communicate(timeout=10)
            test_case.fail(
                f"[{name}] writer never created shm path {shm_path} within "
                f"{writer_timeout}s.\n--- writer output ---\n{out}"
            )

        try:
            reader_result = subprocess.run(
                [sys.executable] + reader_argv,
                cwd=str(REPO_ROOT),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=reader_timeout,
            )
        except subprocess.TimeoutExpired as exc:
            writer_out = _drain(writer_proc)
            test_case.fail(
                f"[{name}] reader did not finish within {reader_timeout}s.\n"
                f"--- reader output so far ---\n{exc.output}\n"
                f"--- writer output ---\n{writer_out}"
            )
            return

        try:
            writer_out, _ = writer_proc.communicate(timeout=writer_timeout)
        except subprocess.TimeoutExpired:
            writer_out = _drain(writer_proc)
            test_case.fail(
                f"[{name}] writer did not exit within {writer_timeout}s after "
                f"reader finished (should have unblocked on reader-done signal).\n"
                f"--- reader output ---\n{reader_result.stdout}\n"
                f"--- writer output ---\n{writer_out}"
            )
            return

        debug = (
            f"\n--- [{name}] reader output ---\n{reader_result.stdout}"
            f"\n--- [{name}] writer output ---\n{writer_out}"
        )
        test_case.assertEqual(
            reader_result.returncode, 0, f"[{name}] reader exited nonzero.{debug}"
        )
        test_case.assertIn(
            "SUCCESS", reader_result.stdout, f"[{name}] reader did not report SUCCESS.{debug}"
        )
        test_case.assertEqual(
            writer_proc.returncode, 0, f"[{name}] writer exited nonzero.{debug}"
        )
    finally:
        if writer_proc.poll() is None:
            writer_proc.kill()
            writer_proc.wait(timeout=10)
        if leftover.exists():
            leftover.unlink()


def _drain(proc, timeout=5):
    try:
        proc.kill()
        out, _ = proc.communicate(timeout=timeout)
        return out
    except Exception:  # noqa: BLE001 - best-effort diagnostic capture
        return "<could not capture writer output>"


@unittest.skipUnless(LIB_PATH.exists(), f"allocator not built at {LIB_PATH}; see module docstring")
class TestBenchmarkWorkloads(unittest.TestCase):
    """One test per writer/reader workload pair in the repo."""

    def _path(self, tag):
        return f"/{tag}_test_{os.getpid()}"

    def test_classic_smoke_test(self):
        path = self._path("classic")
        run_workload_pair(
            self,
            "classic-smoke-test",
            ["tests/writer.py", "--backend", "shm", "--path", path],
            ["tests/reader.py", "--backend", "shm", "--path", path],
            path,
        )

    def test_pt_dense(self):
        path = self._path("pt_dense")
        run_workload_pair(
            self,
            "pt_dense",
            ["benchmark/pt_dense/writer.py", "--backend", "shm", "--path", path],
            ["benchmark/pt_dense/reader.py", "--backend", "shm", "--path", path],
            path,
        )

    def test_pt_multitenant(self):
        path = self._path("pt_multitenant")
        run_workload_pair(
            self,
            "pt_multitenant",
            ["benchmark/pt_multitenant/writer.py", "--backend", "shm", "--path", path],
            ["benchmark/pt_multitenant/reader.py", "--backend", "shm", "--path", path],
            path,
        )

    def test_pt_sparse(self):
        path = self._path("pt_sparse")
        run_workload_pair(
            self,
            "pt_sparse",
            ["benchmark/pt_sparse/writer.py", "--backend", "shm", "--path", path, "--region-size-gb", "1"],
            ["benchmark/pt_sparse/reader.py", "--backend", "shm", "--path", path],
            path,
        )

    def test_pt_kvcache(self):
        path = self._path("pt_kvcache")
        run_workload_pair(
            self,
            "pt_kvcache",
            ["benchmark/pt_kvcache/writer.py", "--backend", "shm", "--path", path, "--max-new-tokens", "12"],
            ["benchmark/pt_kvcache/reader.py", "--backend", "shm", "--path", path, "--max-new-tokens", "12"],
            path,
            reader_timeout=60,
        )

    def test_pt_multireader(self):
        path = self._path("pt_multireader")
        run_workload_pair(
            self,
            "pt_multireader",
            ["benchmark/pt_multireader/writer.py", "--backend", "shm", "--path", path],
            ["benchmark/pt_multireader/reader.py", "--backend", "shm", "--path", path, "--num-readers", "3"],
            path,
            reader_timeout=60,
        )

    def test_pt_quantized(self):
        path = self._path("pt_quantized")
        run_workload_pair(
            self,
            "pt_quantized",
            ["benchmark/pt_quantized/writer.py", "--backend", "shm", "--path", path],
            ["benchmark/pt_quantized/reader.py", "--backend", "shm", "--path", path],
            path,
        )

    def test_pt_vision(self):
        path = self._path("pt_vision")
        run_workload_pair(
            self,
            "pt_vision",
            ["benchmark/pt_vision/writer.py", "--backend", "shm", "--path", path],
            ["benchmark/pt_vision/reader.py", "--backend", "shm", "--path", path],
            path,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
