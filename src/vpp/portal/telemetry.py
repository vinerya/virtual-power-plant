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

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from sqlalchemy import Integer, and_, case, cast, func, literal, select

from vpp.db.models import BatteryStateModel, ResourceModel, ResourceTelemetryModel

if TYPE_CHECKING:
    from collections.abc import Iterable

    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

#: Samples stamped further than this in the future are rejected (clock skew guard).
MAX_FUTURE_SKEW = timedelta(minutes=5)
#: Only samples at least this fresh update the resource's live ``current_power``.
LIVE_STATE_MAX_AGE = timedelta(minutes=15)

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
    1,
    5,
    10,
    15,
    30,
    60,
    120,
    300,
    600,
    900,
    1800,
    3600,
    7200,
    10800,
    21600,
    43200,
    86400,
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
        resource_id=resource_id,
        start=start,
        end=end,
        bucket_s=bucket_s,
        acc=acc,
    )
    await _bucketed(
        session,
        ts_col=ResourceTelemetryModel.timestamp,
        rid_col=ResourceTelemetryModel.resource_id,
        power_col=ResourceTelemetryModel.power_kw,
        soc_col=ResourceTelemetryModel.state_of_charge,
        resource_id=resource_id,
        start=start,
        end=end,
        bucket_s=bucket_s,
        acc=acc,
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
    newest_soc: dict[str, tuple[datetime, float]] = {}

    def _offer(rid: str, ts: datetime | None, soc: float | None) -> None:
        if ts is None or soc is None:
            return
        ts = as_utc(ts)
        soc = float(soc)
        soc = soc / 100.0 if soc > 1.0 else soc
        cur = newest_soc.get(rid)
        if cur is None or ts > cur[0]:
            newest_soc[rid] = (ts, soc)

    # One index seek per resource and table: correlated "newest row" lookups
    # (ORDER BY timestamp DESC LIMIT 1 on the (resource_id, timestamp)
    # indexes). A GROUP BY max(timestamp) instead read every sample of every
    # listed resource, so GET /resources slowed down linearly as telemetry
    # accumulated (~190 ms for 50 batteries at 1M rows on SQLite; found
    # with `vpp loadtest`, see docs/performance.md).
    bs = BatteryStateModel
    rt = ResourceTelemetryModel
    res = ResourceModel

    def newest(model: Any, ts_col: Any, value_col: Any, *extra: Any) -> Any:
        return (
            select(ts_col, value_col)
            .where(model.resource_id == res.id, *extra)
            .order_by(ts_col.desc())
            .limit(1)
        )

    bs_q = newest(bs, bs.timestamp, bs.soc)
    rt_q = newest(rt, rt.timestamp, rt.state_of_charge, rt.state_of_charge.is_not(None))
    stmt = select(
        res.id,
        bs_q.with_only_columns(bs.timestamp).scalar_subquery(),
        bs_q.with_only_columns(bs.soc).scalar_subquery(),
        rt_q.with_only_columns(rt.timestamp).scalar_subquery(),
        rt_q.with_only_columns(rt.state_of_charge).scalar_subquery(),
    ).where(res.id.in_(resource_ids))
    for rid, bs_ts, bs_soc, rt_ts, rt_soc in (await session.execute(stmt)).all():
        _offer(rid, bs_ts, bs_soc)
        _offer(rid, rt_ts, rt_soc)

    return {rid: soc for rid, (_ts, soc) in newest_soc.items()}


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


def is_future(ts: datetime, now: datetime) -> bool:
    """True if ``ts`` is beyond the allowed clock skew (the sample is rejected)."""
    return as_utc(ts) > now + MAX_FUTURE_SKEW


def is_live(ts: datetime, now: datetime) -> bool:
    """True if a sample stamped ``ts`` is fresh enough to update live state.

    Older samples are history only, so a backfill never overwrites the present.
    """
    return now - as_utc(ts) <= LIVE_STATE_MAX_AGE


def newest_live_power(
    samples: Iterable[tuple[datetime, float]], now: datetime
) -> tuple[datetime, float] | None:
    """The newest ``(timestamp, power_kw)`` sample if it may become the live
    ``current_power``, else ``None`` (no samples, or the newest one is too old).

    Shared by the ingestion paths (HTTP endpoint, MQTT) so they agree on
    which sample sets the live value.
    """
    newest = max(samples, key=lambda x: as_utc(x[0]), default=None)
    if newest is None or not is_live(newest[0], now):
        return None
    return newest


async def publish_resource_updated(data: dict[str, Any], *, source: str) -> None:
    """Broadcast ``RESOURCE_UPDATED`` (WebSocket ``resource_updates``, alerts,
    metrics). A broadcast failure is logged, never raised: it must not drop
    the telemetry write that preceded it."""
    from vpp.events import Event, EventType, get_event_bus

    try:
        await get_event_bus().publish(
            Event(event_type=EventType.RESOURCE_UPDATED, data=data, source=source)
        )
    except Exception:
        logger.exception("Failed to publish RESOURCE_UPDATED for %s", data.get("resource_id"))
