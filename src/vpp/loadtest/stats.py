"""Latency / error bookkeeping and trend analysis for the load harness."""

from __future__ import annotations

import math
from array import array
from collections import Counter
from typing import Any


def percentile(sorted_values: Any, q: float) -> float | None:
    """Nearest-rank percentile (``q`` in 0-100) of an already sorted sequence."""
    n = len(sorted_values)
    if n == 0:
        return None
    rank = max(1, math.ceil(q / 100.0 * n))
    return float(sorted_values[min(n, rank) - 1])


def _ms(v: float | None) -> float | None:
    return None if v is None else round(v * 1000.0, 2)


def latency_summary(values: Any) -> dict[str, float | None]:
    """p50/p95/p99/max of latencies given in seconds, reported in milliseconds."""
    s = sorted(values)
    return {
        "p50_ms": _ms(percentile(s, 50)),
        "p95_ms": _ms(percentile(s, 95)),
        "p99_ms": _ms(percentile(s, 99)),
        "max_ms": _ms(float(s[-1])) if s else None,
    }


class Recorder:
    """Counts operations of one kind: successes with their latency, and errors.

    Keeps every latency of the run (8 bytes each) for exact overall
    percentiles, plus the current sampling window, which :meth:`window`
    summarises and resets.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._all = array("d")
        self._window = array("d")
        self.ok_count = 0
        self.errors = 0
        self.skipped = 0  # ticks the generator could not issue (client saturated)
        self._window_ok = 0
        self._window_errors = 0
        self.error_kinds: Counter[str] = Counter()

    def ok(self, latency_s: float | None = None) -> None:
        self.ok_count += 1
        self._window_ok += 1
        if latency_s is not None:
            self._all.append(latency_s)
            self._window.append(latency_s)

    def error(self, kind: str) -> None:
        self.errors += 1
        self._window_errors += 1
        self.error_kinds[kind[:120]] += 1

    def skip(self) -> None:
        self.skipped += 1

    @property
    def total(self) -> int:
        return self.ok_count + self.errors

    def window(self, elapsed_s: float) -> dict[str, Any]:
        """Summarise the window that just ended and start a new one."""
        out: dict[str, Any] = {
            "ops": self._window_ok,
            "rate": round(self._window_ok / elapsed_s, 2) if elapsed_s > 0 else None,
            "errors": self._window_errors,
            **latency_summary(self._window),
        }
        self._window = array("d")
        self._window_ok = 0
        self._window_errors = 0
        return out

    def summary(self, duration_s: float) -> dict[str, Any]:
        total = self.total
        return {
            "ok": self.ok_count,
            "errors": self.errors,
            "skipped": self.skipped,
            "error_rate": round(self.errors / total, 6) if total else 0.0,
            "throughput_per_s": round(self.ok_count / duration_s, 2) if duration_s > 0 else None,
            **latency_summary(self._all),
            "error_kinds": dict(self.error_kinds.most_common(10)),
        }


def linear_slope(xs: list[float], ys: list[float | None]) -> float | None:
    """Least-squares slope of ys over xs (None with fewer than 3 points)."""
    pts = [(x, float(y)) for x, y in zip(xs, ys, strict=False) if y is not None]
    if len(pts) < 3:
        return None
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    var = sum((p[0] - mx) ** 2 for p in pts)
    if var == 0:
        return None
    return sum((p[0] - mx) * (p[1] - my) for p in pts) / var


def _median(values: list[float]) -> float | None:
    s = sorted(v for v in values if v is not None)
    if not s:
        return None
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2


def thirds_ratio(values: list[float | None]) -> float | None:
    """Median of the last third over the median of the first third."""
    vals = [v for v in values if v is not None]
    if len(vals) < 6:
        return None
    k = len(vals) // 3
    first, last = _median(vals[:k]), _median(vals[-k:])
    if not first or last is None:
        return None
    return round(last / first, 3)
