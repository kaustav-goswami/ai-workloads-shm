import time
from contextlib import contextmanager


class TimingRecorder:
    """Collects named latency samples (seconds) and prints/exports a summary.

    Every SHM operation that matters for a fragmentation/perf study - alloc,
    lookup, resize, attach - gets recorded under a short "op" tag plus an
    optional per-object label, so the summary can break down cost by op type.
    """

    def __init__(self, label=""):
        self.label = label
        self._samples = {}

    @contextmanager
    def time(self, op, item=""):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self._samples.setdefault(op, []).append((item, elapsed))

    def record(self, op, elapsed_seconds, item=""):
        self._samples.setdefault(op, []).append((item, elapsed_seconds))

    def total_seconds(self, op=None):
        if op is not None:
            return sum(e for _, e in self._samples.get(op, []))
        return sum(e for samples in self._samples.values() for _, e in samples)

    def count(self, op=None):
        if op is not None:
            return len(self._samples.get(op, []))
        return sum(len(s) for s in self._samples.values())

    def stats(self, op):
        samples = sorted(e for _, e in self._samples.get(op, []))
        n = len(samples)
        if n == 0:
            return None
        p50 = samples[n // 2]
        p99 = samples[min(n - 1, int(n * 0.99))]
        return {
            "n": n,
            "total_s": sum(samples),
            "mean_s": sum(samples) / n,
            "min_s": samples[0],
            "p50_s": p50,
            "p99_s": p99,
            "max_s": samples[-1],
        }

    def summary_lines(self):
        lines = []
        title = f"Timing summary{f' ({self.label})' if self.label else ''}"
        lines.append(title)
        lines.append("-" * len(title))
        if not self._samples:
            lines.append("  (no samples recorded)")
            return lines
        header = f"  {'op':<18}{'n':>6}{'total_ms':>12}{'mean_ms':>12}{'p50_ms':>10}{'p99_ms':>10}{'max_ms':>10}"
        lines.append(header)
        for op in sorted(self._samples):
            s = self.stats(op)
            lines.append(
                f"  {op:<18}{s['n']:>6}"
                f"{s['total_s'] * 1000:>12.3f}"
                f"{s['mean_s'] * 1000:>12.4f}"
                f"{s['p50_s'] * 1000:>10.4f}"
                f"{s['p99_s'] * 1000:>10.4f}"
                f"{s['max_s'] * 1000:>10.4f}"
            )
        lines.append(f"  {'TOTAL':<18}{self.count():>6}{self.total_seconds() * 1000:>12.3f}")
        return lines

    def print_summary(self):
        print("\n".join(self.summary_lines()))

    def as_dict(self):
        return {
            "label": self.label,
            "ops": {op: self.stats(op) for op in self._samples},
        }
