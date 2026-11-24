"""Automated correctness tests for every benchmark workload in this repo.

Each workload is a single self-contained script (`benchmark/<name>/model.py`)
that builds a model, initializes weights deterministically, and runs a
forward pass (or, for PT-KVCache, an incremental decode loop; for
PT-MultiReader, several concurrent reader threads) - all using ordinary
tensors allocated with PyTorch's default CPU allocator. Every script prints
a "SUCCESS" marker on a clean run.

Run with:
    PYTHONPATH=. python3 tests/test_benchmarks.py
or:
    PYTHONPATH=. python3 -m unittest discover -s tests -p 'test_*.py' -v
"""

import os
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _env():
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{existing}" if existing else str(REPO_ROOT)
    return env


def run_workload(test_case, name, argv, timeout=60):
    """Runs one workload script to completion and asserts success."""
    result = subprocess.run(
        [sys.executable] + argv,
        cwd=str(REPO_ROOT),
        env=_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )
    debug = f"\n--- [{name}] output ---\n{result.stdout}"
    test_case.assertEqual(result.returncode, 0, f"[{name}] exited nonzero.{debug}")
    test_case.assertIn("SUCCESS", result.stdout, f"[{name}] did not report SUCCESS.{debug}")


class TestBenchmarkWorkloads(unittest.TestCase):
    """One test per benchmark workload script in the repo."""

    def test_classic_smoke_test(self):
        run_workload(self, "classic-smoke-test", ["tests/smoke_test.py"])

    def test_pt_dense(self):
        run_workload(self, "pt_dense", ["benchmark/pt_dense/model.py"])

    def test_pt_multitenant(self):
        run_workload(self, "pt_multitenant", ["benchmark/pt_multitenant/model.py"])

    def test_pt_sparse(self):
        run_workload(self, "pt_sparse", ["benchmark/pt_sparse/model.py"])

    def test_pt_kvcache(self):
        run_workload(
            self, "pt_kvcache", ["benchmark/pt_kvcache/model.py", "--max-new-tokens", "12"]
        )

    def test_pt_multireader(self):
        run_workload(
            self, "pt_multireader", ["benchmark/pt_multireader/model.py", "--num-readers", "3"]
        )

    def test_pt_quantized(self):
        run_workload(self, "pt_quantized", ["benchmark/pt_quantized/model.py"])

    def test_pt_vision(self):
        run_workload(self, "pt_vision", ["benchmark/pt_vision/model.py"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
