"""Synthetic price feed — deterministic sine + noise + peak markers.

Useful as a fallback when network feeds are unreachable, and as a fixture
for benchmarks and integration tests. Produces the same prices for the
same (start, end, seed) tuple — no I/O.
"""
from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone
from typing import Any

from .base import PriceFeed, PricePoint


class SyntheticFeed(PriceFeed):
    """Deterministic synthetic energy-price feed.

    Model
    -----
    ``price[t] = base + amplitude * sin(2π · (h - 8) / 24)
                 + peak_adder * 1[16 <= h < 21]
                 + noise``

    where ``h`` is the local hour of day. Defaults make a midday low
    around 02:00, peak ramp 16:00-21:00, and a small Gaussian noise
    component reproducible by ``seed``.
    """

    name = "synthetic"
    timezone = "UTC"

    def __init__(
        self,
        base: float = 0.12,
        amplitude: float = 0.04,
        peak_adder: float = 0.18,
        noise_std: float = 0.005,
        interval_minutes: int = 60,
        seed: int = 42,
    ) -> None:
        self.base = float(base)
        self.amplitude = float(amplitude)
        self.peak_adder = float(peak_adder)
        self.noise_std = float(noise_std)
        self.interval_minutes = int(interval_minutes)
        self.seed = int(seed)

    async def fetch(
        self, start: datetime, end: datetime, **kwargs: Any
    ) -> list[PricePoint]:
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        rng = random.Random(self.seed + int(start.timestamp()))
        step = timedelta(minutes=self.interval_minutes)
        out: list[PricePoint] = []
        t = start
        while t < end:
            h = t.hour + t.minute / 60.0
            sin_val = math.sin(2 * math.pi * (h - 8) / 24)
            peak = self.peak_adder if 16 <= h < 21 else 0.0
            noise = rng.gauss(0.0, self.noise_std)
            price = max(0.0, self.base + self.amplitude * sin_val + peak + noise)
            out.append(
                PricePoint(
                    timestamp=t,
                    price_per_kwh=round(price, 6),
                    feed=self.name,
                    metadata={"is_peak": peak > 0},
                )
            )
            t = t + step
        return out
