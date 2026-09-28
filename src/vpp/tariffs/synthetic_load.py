"""Deterministic synthetic load profiles for bill simulation.

These are *illustrative* shapes for comparing tariffs when no metered data is
at hand -- not a forecast of any real premise. Every simulation that uses one
reports ``load_summary.source == "synthetic"`` so a result is never mistaken
for a bill computed from a real meter.

Shapes (fraction of the average load, by local hour; weekdays/weekends):

* ``residential`` -- morning shoulder, evening peak 17:00-21:00, low overnight;
  weekends flatter and later.
* ``commercial``  -- business hours 08:00-18:00 on weekdays, ~35 % base load
  nights and weekends.

A small deterministic day-to-day modulation (+/-8 %, from a fixed sine) keeps
demand peaks realistic without randomness, so identical requests produce
identical bills.

Optional rooftop PV (``pv_kw`` DC-ish nameplate) uses a clear-sky bell curve
between 06:00 and 18:00 local, peaking at ``pv_kw`` at 12:00 and derated by a
fixed 0.8 performance ratio. Net load below zero is reported as export.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, tzinfo

from .meter import MeterTrace

PROFILES = ("residential", "commercial")

_RES_WEEKDAY = [
    0.55, 0.50, 0.48, 0.47, 0.48, 0.55, 0.80, 1.05, 0.95, 0.80, 0.75, 0.75,
    0.78, 0.80, 0.85, 0.95, 1.20, 1.60, 1.85, 1.90, 1.75, 1.45, 1.05, 0.75,
]  # fmt: skip
_RES_WEEKEND = [
    0.60, 0.55, 0.52, 0.50, 0.50, 0.52, 0.62, 0.80, 1.00, 1.10, 1.10, 1.05,
    1.00, 0.98, 1.00, 1.05, 1.20, 1.45, 1.65, 1.70, 1.60, 1.35, 1.00, 0.75,
]  # fmt: skip
_COM_WEEKDAY = [
    0.45, 0.45, 0.45, 0.45, 0.45, 0.50, 0.75, 1.20, 1.55, 1.65, 1.70, 1.75,
    1.75, 1.75, 1.70, 1.65, 1.55, 1.35, 0.95, 0.65, 0.55, 0.50, 0.48, 0.45,
]  # fmt: skip
_COM_WEEKEND = [0.45] * 6 + [0.5] * 12 + [0.45] * 6


def _normalized(shape: list[float]) -> list[float]:
    mean = sum(shape) / len(shape)
    return [v / mean for v in shape]


_SHAPES = {
    "residential": (_normalized(_RES_WEEKDAY), _normalized(_RES_WEEKEND)),
    "commercial": (_normalized(_COM_WEEKDAY), _normalized(_COM_WEEKEND)),
}

DEFAULT_AVG_KW = {"residential": 0.8, "commercial": 40.0}
PV_PERFORMANCE_RATIO = 0.8


def _pv_kw(pv_kw: float, local: datetime) -> float:
    hour = local.hour + local.minute / 60.0
    if pv_kw <= 0 or not (6.0 <= hour <= 18.0):
        return 0.0
    return pv_kw * PV_PERFORMANCE_RATIO * math.sin(math.pi * (hour - 6.0) / 12.0) ** 2


def synthetic_trace(
    *,
    start: datetime,
    days: int,
    tz: tzinfo,
    profile: str = "residential",
    avg_kw: float | None = None,
    pv_kw: float = 0.0,
    interval_minutes: int = 60,
) -> MeterTrace:
    """Build a deterministic synthetic :class:`MeterTrace` starting at ``start``."""
    if profile not in _SHAPES:
        raise ValueError(f"profile must be one of {PROFILES}")
    if start.tzinfo is None:
        start = start.replace(tzinfo=tz)
    avg = DEFAULT_AVG_KW[profile] if avg_kw is None else avg_kw
    weekday_shape, weekend_shape = _SHAPES[profile]
    step = timedelta(minutes=interval_minutes)
    hours = interval_minutes / 60.0
    n = int(days * 24 * 60 / interval_minutes)
    timestamps: list[datetime] = []
    imports: list[float] = []
    exports: list[float] = []
    for i in range(n):
        ts = start + i * step
        local = ts.astimezone(tz)
        shape = weekend_shape if local.weekday() >= 5 else weekday_shape
        day_index = (local.date() - start.astimezone(tz).date()).days
        modulation = 1.0 + 0.08 * math.sin(2 * math.pi * day_index / 7.3)
        load_kw = avg * shape[local.hour] * modulation
        net_kw = load_kw - _pv_kw(pv_kw, local)
        timestamps.append(ts)
        imports.append(round(max(net_kw, 0.0) * hours, 6))
        exports.append(round(max(-net_kw, 0.0) * hours, 6))
    return MeterTrace(
        timestamps=timestamps,
        import_kwh=imports,
        export_kwh=exports,
        interval_minutes=interval_minutes,
        tz=tz,
    )
