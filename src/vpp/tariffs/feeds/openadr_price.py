"""OpenADR price-signal feed adapter.

Bridges OpenADR 2.0b ``DISTRIBUTE_EVENT`` messages carrying
``priceMultiplier`` (or ELECTRICITY_PRICE) signals into the uniform
:class:`PriceFeed` contract.

Design
------
OpenADR DR events arrive asynchronously from a VTN — we cannot "fetch a
window" the same way as OASIS. Instead this adapter:

1. Subscribes to events via :class:`vpp.protocols.openadr.OpenADRAdapter`.
2. Maintains an in-memory ring of (interval_start, signal_level) tuples.
3. ``fetch(start, end)`` filters that ring to the requested window and
   converts each signal to ``PricePoint`` using ``base_price * multiplier``
   for SIMPLE/priceMultiplier signals, or the raw level for
   ELECTRICITY_PRICE signals (already $/kWh).

For testing without a live VTN, callers may invoke :meth:`ingest_event`
directly with a :class:`DREvent`.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from .base import PriceFeed, PricePoint

try:
    from vpp.protocols.openadr import DREvent, DRSignalType
except Exception:  # pragma: no cover  (protocols extra optional)
    DREvent = None  # type: ignore[assignment]
    DRSignalType = None  # type: ignore[assignment]


class OpenADRPriceFeed(PriceFeed):
    """Price feed driven by OpenADR DISTRIBUTE_EVENT signals."""

    name = "openadr_price"
    timezone = "UTC"

    def __init__(
        self,
        base_price_per_kwh: float = 0.12,
        interval_minutes: int = 60,
    ) -> None:
        self.base_price_per_kwh = float(base_price_per_kwh)
        self.interval_minutes = int(interval_minutes)
        self._events: list[tuple[datetime, datetime, float, str]] = []
        # (start, end, signal_level, signal_type)

    def ingest_event(self, event: Any) -> None:
        """Record an OpenADR DR event for later windowed fetches.

        Accepts a :class:`vpp.protocols.openadr.DREvent` or any object
        exposing ``start_time`` (epoch seconds), ``duration_seconds``,
        ``signal_level``, ``signal_type``.
        """
        start_epoch = float(event.start_time)
        duration = int(event.duration_seconds)
        level = float(event.signal_level)
        # signal_type may be enum or str
        st = getattr(event.signal_type, "value", event.signal_type)
        start = datetime.fromtimestamp(start_epoch, tz=timezone.utc)
        end = start + timedelta(seconds=duration)
        self._events.append((start, end, level, str(st)))

    def clear(self) -> None:
        self._events.clear()

    async def fetch(
        self, start: datetime, end: datetime, **kwargs: Any
    ) -> list[PricePoint]:
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        step = timedelta(minutes=self.interval_minutes)
        out: list[PricePoint] = []
        t = start
        while t < end:
            price = self.base_price_per_kwh
            applied: Optional[tuple[float, str]] = None
            for ev_start, ev_end, level, stype in self._events:
                if ev_start <= t < ev_end:
                    if stype == "ELECTRICITY_PRICE":
                        price = level  # treat level as $/kWh directly
                    else:
                        # SIMPLE / priceMultiplier semantics
                        price = self.base_price_per_kwh * max(0.0, level)
                    applied = (level, stype)
                    break
            meta: dict[str, Any] = {}
            if applied is not None:
                meta = {"openadr_level": applied[0], "openadr_signal": applied[1]}
            out.append(
                PricePoint(
                    timestamp=t,
                    price_per_kwh=round(price, 6),
                    feed=self.name,
                    metadata=meta,
                )
            )
            t = t + step
        return out
