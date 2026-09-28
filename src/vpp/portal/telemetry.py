"""Resource telemetry history: storage helpers and downsampled queries.

Two tables carry per-resource time series:

* ``battery_states`` -- MQTT battery telemetry (``soc`` stored 0-100, or 0-1
  from some producers; normalised to a 0-1 fraction on read).
* ``resource_telemetry`` -- generic samples (Modbus polling, the ingest
  endpoint), ``state_of_charge`` already a 0-1 fraction.

:func:`query_history` time-buckets both in SQL (``AVG`` per bucket, so a 7-day
window of 5-second samples never leaves the database as 120k rows) and
merges them per bucket weighted by sample count. Buckets without samples are
omitted rather than zero-filled: a gap in the series means "no data", and
the chart should show it as such.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from sqlalchemy import Integer, and_, case, cast, func, literal, select

from vpp.db.models import BatteryStateModel, ResourceTelemetryModel

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

#: Named windows accepted by ``GET /resources/{id}/metrics?window=``.
WINDOWS: dict[str, timedelta] = {
    "1h": timedelta(hours=1),
    "6h": timedelta(hours=6),
    "24h": timedelta(hours=24),
    "7d": timedelta(days=7),
    "30d": timedelta(days=30),
}
MAX_SPAN = timedelta(days=366)
#: Bucket sizes are snapped up to one of these so bucket edges line up with
#: wall-clock boundaries (and consecutive polls return stable buckets).
_NICE_BUCKETS_S = (
    1, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600,
    7200, 10800, 21600, 43200, 86400,
)


def as_utc(ts: datetime) -> datetime:
    """Treat naive datetimes (SQLite round-trips) as UTC; convert aware ones."""
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def choose_bucket_seconds(span: timedelta, max_points: int) -> int:
    """Smallest 'nice' bucket that keeps ``span`` within ``max_points`` buckets."""
    raw = max(1, math.ceil(span.total_seconds() / max(1, max_points)))
    for b in _NICE_BUCKETS_S:
        if b >= raw:
            return b
    return math.ceil(raw / 86400) * 86400


def _normalised_soc(column):
    """0-100 percentages -> 0-1 fraction; 0-1 values pass through."""
    return case((column > 1.0, column / 100.0), else_=column)


def _bucket_expr(dialect: str, column, bucket_s: int):
    if dialect == "sqlite":
        # Integer / integer is integer (floor for non-negative epochs) in SQLite.
        epoch = cast(func.strftime("%s", column), Integer)
        return epoch.op("/")(literal(bucket_s, Integer))
    if dialect == "postgresql":
        return cast(func.floor(func.extract("epoch", column) / bucket_s), Integer)
    return None


@dataclass
class _Acc:
    power_sum: float = 0.0
    power_n: int = 0
    soc_sum: float = 0.0
    soc_n: int = 0


async def _bucketed(
    session: AsyncSession,
    *,
    ts_col,
    rid_col,
    power_col,
    soc_col,
    resource_id: str,
    start: datetime,
    end: datetime,
    bucket_s: int,
    acc: dict[int, _Acc],
) -> None:
    dialect = session.get_bind().dialect.name
    where = and_(rid_col == resource_id, ts_col >= start, ts_col < end)
    bucket = _bucket_expr(dialect, ts_col, bucket_s)
    if bucket is not None:
        stmt = (
            select(
                bucket.label("b"),
                func.avg(power_col),
                func.count(power_col),
                func.avg(soc_col),
                func.count(soc_col),
            )
            .where(where)
            .group_by("b")
        )
        for b, p_avg, p_n, s_avg, s_n in (await session.execute(stmt)).all():
            a = acc.setdefault(int(b), _Acc())
            if p_n:
                a.power_sum += float(p_avg) * p_n
                a.power_n += p_n
            if s_n:
                a.soc_sum += float(s_avg) * s_n
                a.soc_n += s_n
        return
    # Portable fallback for dialects without a known epoch function.
    stmt = select(ts_col, power_col, soc_col).where(where)
    for ts, p, s in (await session.execute(stmt)).all():
        b = int(as_utc(ts).timestamp()) // bucket_s
        a = acc.setdefault(b, _Acc())
        if p is not None:
            a.power_sum += float(p)
            a.power_n += 1
        if s is not None:
            a.soc_sum += float(s)
            a.soc_n += 1


async def query_history(
    session: AsyncSession,
    resource_id: str,
    *,
    start: datetime,
    end: datetime,
    bucket_s: int,
) -> list[dict[str, Any]]:
    """Downsampled ``[{timestamp, power, state_of_charge?, samples}]`` in ``[start, end)``."""
    start, end = as_utc(start), as_utc(end)
    acc: dict[int, _Acc] = {}
    await _bucketed(
        session,
        ts_col=BatteryStateModel.timestamp,
        rid_col=BatteryStateModel.resource_id,
        power_col=BatteryStateModel.power,
        soc_col=_normalised_soc(BatteryStateModel.soc),
        resource_id=resource_id, start=start, end=end, bucket_s=bucket_s, acc=acc,
    )
    await _bucketed(
        session,
        ts_col=ResourceTelemetryModel.timestamp,
        rid_col=ResourceTelemetryModel.resource_id,
        power_col=ResourceTelemetryModel.power_kw,
        soc_col=ResourceTelemetryModel.state_of_charge,
        resource_id=resource_id, start=start, end=end, bucket_s=bucket_s, acc=acc,
    )
    points: list[dict[str, Any]] = []
    for b in sorted(acc):
        a = acc[b]
        if a.power_n == 0 and a.soc_n == 0:
            continue
        point: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(b * bucket_s, tz=timezone.utc),
            "power": a.power_sum / a.power_n if a.power_n else None,
            "samples": max(a.power_n, a.soc_n),
        }
        if a.soc_n:
            point["state_of_charge"] = a.soc_sum / a.soc_n
        points.append(point)
    return points


async def latest_soc(session: AsyncSession, resource_ids: list[str]) -> dict[str, float]:
    """Most recent state of charge (0-1) per resource across both telemetry tables."""
    if not resource_ids:
        return {}
    newest: dict[str, tuple[datetime, float]] = {}

    def _offer(rid: str, ts: datetime | None, soc: float | None) -> None:
        if ts is None or soc is None:
            return
        ts = as_utc(ts)
        soc = float(soc)
        soc = soc / 100.0 if soc > 1.0 else soc
        cur = newest.get(rid)
        if cur is None or ts > cur[0]:
            newest[rid] = (ts, soc)

    bs = BatteryStateModel
    sub = (
        select(bs.resource_id, func.max(bs.timestamp).label("ts"))
        .where(bs.resource_id.in_(resource_ids))
        .group_by(bs.resource_id)
        .subquery()
    )
    stmt = select(bs.resource_id, bs.timestamp, bs.soc).join(
        sub, and_(bs.resource_id == sub.c.resource_id, bs.timestamp == sub.c.ts)
    )
    for rid, ts, soc in (await session.execute(stmt)).all():
        _offer(rid, ts, soc)

    rt = ResourceTelemetryModel
    sub2 = (
        select(rt.resource_id, func.max(rt.timestamp).label("ts"))
        .where(rt.resource_id.in_(resource_ids), rt.state_of_charge.is_not(None))
        .group_by(rt.resource_id)
        .subquery()
    )
    stmt2 = select(rt.resource_id, rt.timestamp, rt.state_of_charge).join(
        sub2, and_(rt.resource_id == sub2.c.resource_id, rt.timestamp == sub2.c.ts)
    )
    for rid, ts, soc in (await session.execute(stmt2)).all():
        _offer(rid, ts, soc)

    return {rid: soc for rid, (_ts, soc) in newest.items()}


async def record_samples(
    session: AsyncSession,
    resource_id: str,
    samples: list[tuple[datetime, float, float | None]],
    *,
    source: str,
) -> None:
    """Append ``(timestamp, power_kw, soc_fraction|None)`` samples (UTC-normalised)."""
    session.add_all(
        ResourceTelemetryModel(
            resource_id=resource_id,
            timestamp=as_utc(ts),
            power_kw=float(power),
            state_of_charge=soc,
            source=source,
        )
        for ts, power, soc in samples
    )
    await session.flush()
