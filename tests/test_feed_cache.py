"""Tests for FeedCache TTL semantics (M4)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from vpp.tariffs.feeds import FeedCache, PricePoint


def _sample_points() -> list[PricePoint]:
    base = datetime(2024, 7, 1, tzinfo=timezone.utc)
    return [
        PricePoint(timestamp=base + timedelta(hours=h), price_per_kwh=0.10 + h * 0.01,
                   feed="t")
        for h in range(3)
    ]


def test_cache_hits_within_ttl():
    """A set/get within the TTL returns the same points."""
    cache = FeedCache(ttl_seconds=300)
    fake_now = {"t": 1_000_000.0}
    cache._now = lambda: fake_now["t"]

    start = datetime(2024, 7, 1, tzinfo=timezone.utc)
    end = start + timedelta(hours=3)
    pts = _sample_points()
    cache.set("synthetic", start, end, pts)

    fake_now["t"] += 100  # within 300s
    got = cache.get("synthetic", start, end)
    assert got is not None
    assert len(got) == 3
    assert got[0].price_per_kwh == pts[0].price_per_kwh


def test_cache_evicts_after_ttl():
    """After the TTL elapses, get() returns None."""
    cache = FeedCache(ttl_seconds=60)
    fake_now = {"t": 5_000_000.0}
    cache._now = lambda: fake_now["t"]

    start = datetime(2024, 8, 1, tzinfo=timezone.utc)
    end = start + timedelta(hours=1)
    cache.set("synthetic", start, end, _sample_points())

    # First get inside TTL — present.
    assert cache.get("synthetic", start, end) is not None

    fake_now["t"] += 120  # past 60s TTL
    assert cache.get("synthetic", start, end) is None
