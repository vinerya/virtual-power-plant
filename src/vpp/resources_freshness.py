"""Telemetry freshness of resources: how old their latest data is.

A resource's last telemetry time is the newest sample stored for it in
either telemetry table (``resource_telemetry``: HTTP ingest endpoint and
Modbus polling; ``battery_states``: MQTT battery telemetry). There is no
separate "last seen" column: the samples themselves are the record.

The data counts as **stale** when it is older than the resource's staleness
threshold, chosen in this order:

1. ``metadata.telemetry_stale_after_s`` on the resource (``0`` turns the
   check off for that resource);
2. for a resource polled over Modbus (``metadata.modbus`` with a positive
   ``poll_interval_s``, default 5 s): ``poll_interval_s *``
   ``VPP_TELEMETRY_STALE_MODBUS_POLL_MULTIPLIER`` (when that is positive);
3. ``VPP_TELEMETRY_STALE_AFTER_SECONDS`` (default 300; ``0`` turns the check
   off).

A resource that has never reported anything is ``no_data``, not stale: a
resource described by hand, without a telemetry source, must not raise
alarms.

Who uses it:

* the resource API (``telemetry_*`` and ``data_status`` fields of
  ``GET /api/v1/resources``);
* the alert service's freshness check (built-in "Resource telemetry stale"
  rule, :mod:`vpp.alert_service`);
* anything else that wants to know whether to trust a resource's live
  values, e.g. :func:`stale_resource_ids` for the optimiser.

Everything here except :func:`latest_telemetry_at` and
:func:`stale_resource_ids` is pure (no I/O).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from sqlalchemy.ext.asyncio import AsyncSession

#: Per-resource override of the staleness threshold (seconds) in ``metadata``.
METADATA_KEY = "telemetry_stale_after_s"
#: ``ModbusAdapter``'s default poll interval (seconds).
DEFAULT_MODBUS_POLL_INTERVAL_S = 5.0

DataStatus = Literal["fresh", "stale", "no_data", "unchecked"]


@dataclass(frozen=True)
class FreshnessSettings:
    """Staleness thresholds (see the module docstring for how they combine)."""

    stale_after_s: float = 300.0
    modbus_poll_multiplier: float = 10.0

    @classmethod
    def from_settings(cls, settings: Any = None) -> FreshnessSettings:
        """Read ``VPP_TELEMETRY_STALE_*`` from *settings* (default: the app settings)."""
        if settings is None:
            from vpp.settings import get_settings

            settings = get_settings()
        if isinstance(settings, FreshnessSettings):
            return settings
        return cls(
            stale_after_s=float(getattr(settings, "telemetry_stale_after_seconds", 300.0)),
            modbus_poll_multiplier=float(
                getattr(settings, "telemetry_stale_modbus_poll_multiplier", 10.0)
            ),
        )


@dataclass(frozen=True)
class TelemetryFreshness:
    """Freshness of one resource's telemetry at a point in time."""

    last_at: datetime | None
    age_s: float | None
    stale_after_s: float | None  # None: the check is off for this resource
    stale: bool
    data_status: DataStatus

    def as_fields(self) -> dict[str, Any]:
        """The resource API's ``telemetry_*`` / ``data_status`` fields."""
        return {
            "telemetry_last_at": self.last_at,
            "telemetry_age_s": self.age_s,
            "telemetry_stale_after_s": self.stale_after_s,
            "telemetry_stale": self.stale,
            "data_status": self.data_status,
        }


def _as_utc(ts: datetime) -> datetime:
    """SQLite returns naive datetimes; everything is stored in UTC."""
    return ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts.astimezone(timezone.utc)


def _positive_number(value: Any) -> float | None:
    """``value`` as a float when it is a finite number >= 0, else ``None``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    v = float(value)
    return v if v >= 0.0 and v == v and v != float("inf") else None


def resource_metadata(resource: Any) -> dict[str, Any]:
    """The metadata dict of a resource row, API dict or fleet asset."""
    if isinstance(resource, dict):
        raw = resource.get("metadata", resource.get("metadata_json"))
    else:
        raw = getattr(resource, "metadata", None)
        if not isinstance(raw, dict):
            raw = getattr(resource, "metadata_json", None)
    if isinstance(raw, str):
        try:
            raw = json.loads(raw) if raw else {}
        except ValueError:
            return {}
    return raw if isinstance(raw, dict) else {}


def stale_after_s(metadata: Mapping[str, Any] | None, settings: Any = None) -> float | None:
    """The staleness threshold (seconds) for a resource with *metadata*.

    ``None`` means the check is off for this resource.
    """
    cfg = FreshnessSettings.from_settings(settings)
    metadata = metadata or {}
    override = _positive_number(metadata.get(METADATA_KEY))
    if override is not None:
        return override or None
    modbus = metadata.get("modbus")
    if isinstance(modbus, dict) and cfg.modbus_poll_multiplier > 0:
        raw = modbus.get("poll_interval_s", DEFAULT_MODBUS_POLL_INTERVAL_S)
        try:
            poll = float(raw)
        except (TypeError, ValueError):
            poll = DEFAULT_MODBUS_POLL_INTERVAL_S
        if poll > 0:
            return poll * cfg.modbus_poll_multiplier
    return cfg.stale_after_s if cfg.stale_after_s > 0 else None


def telemetry_freshness(
    last_at: datetime | None,
    threshold_s: float | None,
    now: datetime | None = None,
) -> TelemetryFreshness:
    """Freshness of data last received at *last_at*, with threshold *threshold_s*."""
    if last_at is None:
        return TelemetryFreshness(None, None, threshold_s, False, "no_data")
    now = _as_utc(now) if now is not None else datetime.now(timezone.utc)
    last_at = _as_utc(last_at)
    # Samples may carry timestamps slightly in the future (the HTTP endpoint
    # accepts up to 5 minutes ahead): that is age 0, not negative.
    age = max(0.0, (now - last_at).total_seconds())
    if threshold_s is None:
        return TelemetryFreshness(last_at, age, None, False, "unchecked")
    stale = age > threshold_s
    return TelemetryFreshness(last_at, age, threshold_s, stale, "stale" if stale else "fresh")


def freshness_of(
    resource: Any,
    last_at: datetime | None,
    now: datetime | None = None,
    settings: Any = None,
) -> TelemetryFreshness:
    """:func:`telemetry_freshness` with the resource's own threshold."""
    return telemetry_freshness(last_at, stale_after_s(resource_metadata(resource), settings), now)


def is_stale(
    resource: Any,
    now: datetime | None = None,
    settings: Any = None,
    *,
    last_telemetry_at: datetime | None,
) -> bool:
    """Whether *resource*'s telemetry is older than its staleness threshold.

    *resource* is anything with metadata: a ``ResourceModel`` row
    (``metadata_json``), a ``FleetAsset`` or an API dict (``metadata``).
    *last_telemetry_at* comes from :func:`latest_telemetry_at`. A resource
    that never reported (``None``) or whose check is off is not stale.
    """
    return freshness_of(resource, last_telemetry_at, now, settings).stale


async def latest_telemetry_at(
    session: AsyncSession, resource_ids: Iterable[str]
) -> dict[str, datetime]:
    """Newest telemetry timestamp (UTC) per resource, across both tables.

    Resources without any sample are left out. One index seek per resource
    and table (``ORDER BY timestamp DESC LIMIT 1`` on the ``(resource_id,
    timestamp)`` indexes), like :func:`vpp.portal.telemetry.latest_soc`.
    """
    ids = list(dict.fromkeys(str(r) for r in resource_ids))
    if not ids:
        return {}
    from sqlalchemy import select

    from vpp.db.models import BatteryStateModel, ResourceModel, ResourceTelemetryModel

    res = ResourceModel

    def newest(model: Any) -> Any:
        return (
            select(model.timestamp)
            .where(model.resource_id == res.id)
            .order_by(model.timestamp.desc())
            .limit(1)
            .scalar_subquery()
        )

    out: dict[str, datetime] = {}
    # Bounded IN lists: SQLite caps bound parameters per statement.
    for start in range(0, len(ids), 500):
        chunk = ids[start : start + 500]
        stmt = select(res.id, newest(ResourceTelemetryModel), newest(BatteryStateModel)).where(
            res.id.in_(chunk)
        )
        for rid, rt_ts, bs_ts in (await session.execute(stmt)).all():
            stamps = [_as_utc(t) for t in (rt_ts, bs_ts) if t is not None]
            if stamps:
                out[rid] = max(stamps)
    return out


async def stale_resource_ids(
    session: AsyncSession,
    resources: Iterable[Any],
    now: datetime | None = None,
    settings: Any = None,
) -> set[str]:
    """Ids of *resources* (rows, assets or dicts with ``id``) whose data is stale."""
    items = list(resources)
    ids = [str(r["id"] if isinstance(r, dict) else r.id) for r in items]
    last = await latest_telemetry_at(session, ids)
    return {
        rid
        for rid, r in zip(ids, items, strict=True)
        if is_stale(r, now, settings, last_telemetry_at=last.get(rid))
    }


__all__ = [
    "METADATA_KEY",
    "DataStatus",
    "FreshnessSettings",
    "TelemetryFreshness",
    "freshness_of",
    "is_stale",
    "latest_telemetry_at",
    "resource_metadata",
    "stale_after_s",
    "stale_resource_ids",
    "telemetry_freshness",
]
