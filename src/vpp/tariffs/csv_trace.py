"""Parse an uploaded interval-meter CSV into a :class:`MeterTrace`.

Accepted layout: a header row, a ``timestamp`` column (aliases ``time``,
``datetime``, ``interval_start``) holding the interval *start* in ISO 8601,
and energy in one of these forms:

* ``import_kwh`` (alias ``kwh``) with optional ``export_kwh`` -- energy per
  interval;
* ``kw`` (alias ``demand_kw``, ``load_kw``) with optional ``export_kw`` --
  average power over the interval; a negative ``kw`` is treated as export.

Timestamps without an offset are interpreted in the simulation timezone.
The interval length is the most common gap between consecutive rows unless
given explicitly, and must be 1-60 minutes and divide an hour. Rows are
sorted; duplicate timestamps are rejected rather than silently summed.
"""

from __future__ import annotations

import csv
import io
from collections import Counter
from datetime import datetime, tzinfo
from itertools import pairwise

from .meter import MeterTrace

MAX_ROWS = 110_000  # a year of 5-minute data

_TS = ("timestamp", "time", "datetime", "interval_start", "start")
_KWH = ("import_kwh", "kwh", "energy_kwh")
_KW = ("kw", "demand_kw", "load_kw", "power_kw")
_EXP_KWH = ("export_kwh",)
_EXP_KW = ("export_kw",)


class CSVTraceError(ValueError):
    """The CSV cannot be turned into a meter trace; message is user-facing."""


def _find(header: list[str], names: tuple[str, ...]) -> int | None:
    for name in names:
        if name in header:
            return header.index(name)
    return None


def _num(value: str, row: int, col: str) -> float:
    try:
        return float(value) if value.strip() != "" else 0.0
    except ValueError as exc:
        raise CSVTraceError(f"row {row}: {col} value {value!r} is not a number") from exc


def parse_csv_trace(text: str, *, tz: tzinfo, interval_minutes: int | None = None) -> MeterTrace:
    reader = csv.reader(io.StringIO(text.lstrip("﻿")))
    try:
        header = [h.strip().lower() for h in next(reader)]
    except StopIteration as exc:
        raise CSVTraceError("CSV is empty") from exc
    ts_i = _find(header, _TS)
    if ts_i is None:
        raise CSVTraceError(f"CSV needs a timestamp column (one of {', '.join(_TS)})")
    kwh_i, kw_i = _find(header, _KWH), _find(header, _KW)
    if kwh_i is None and kw_i is None:
        raise CSVTraceError(
            f"CSV needs an energy column: one of {', '.join(_KWH)} (kWh per interval) "
            f"or {', '.join(_KW)} (average kW)"
        )
    exp_i = _find(header, _EXP_KWH if kwh_i is not None else _EXP_KW)

    rows: list[tuple[datetime, float, float]] = []
    for n, rec in enumerate(reader, start=2):
        if not rec or all(not c.strip() for c in rec):
            continue
        if len(rows) >= MAX_ROWS:
            raise CSVTraceError(f"CSV has more than {MAX_ROWS} rows")
        try:
            ts = datetime.fromisoformat(rec[ts_i].strip().replace("Z", "+00:00"))
        except (ValueError, IndexError) as exc:
            raise CSVTraceError(f"row {n}: cannot parse timestamp") from exc
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=tz)
        val_i = kwh_i if kwh_i is not None else kw_i
        assert val_i is not None
        value = _num(rec[val_i] if val_i < len(rec) else "", n, header[val_i])
        exp = _num(rec[exp_i], n, header[exp_i]) if exp_i is not None and exp_i < len(rec) else 0.0
        if value < 0:  # net meter convention: negative = export
            exp += -value
            value = 0.0
        if exp < 0:
            raise CSVTraceError(f"row {n}: export must not be negative")
        rows.append((ts, value, exp))

    if not rows:
        raise CSVTraceError("CSV has no data rows")
    rows.sort(key=lambda r: r[0])
    for a, b in pairwise(rows):
        if a[0] == b[0]:
            raise CSVTraceError(f"duplicate timestamp {a[0].isoformat()}")

    if interval_minutes is None:
        if len(rows) < 2:
            raise CSVTraceError("cannot infer the interval from a single row")
        gaps = Counter(round((b[0] - a[0]).total_seconds() / 60) for a, b in pairwise(rows))
        interval_minutes = gaps.most_common(1)[0][0]
    if interval_minutes <= 0 or interval_minutes > 60 or 60 % interval_minutes:
        raise CSVTraceError(f"interval of {interval_minutes} min is unsupported (must divide 60)")

    hours = interval_minutes / 60.0
    to_kwh = 1.0 if kwh_i is not None else hours
    return MeterTrace(
        timestamps=[r[0] for r in rows],
        import_kwh=[r[1] * to_kwh for r in rows],
        export_kwh=[r[2] * to_kwh for r in rows],
        interval_minutes=interval_minutes,
        tz=tz,
    )
