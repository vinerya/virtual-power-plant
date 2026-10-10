"""MQTT battery telemetry ingestion (M5).

Bridges :class:`~vpp.protocols.mqtt.MQTTAdapter` messages on
``vpp/{site_id}/{resource_type}/{resource_id}/{metric}`` topics into
``battery_states`` rows. Before this module existed, nothing in production
ever wrote to that table: the degradation updater's DB-backed telemetry
fetch (``_fetch_recent_soc_window`` in ``vpp.api.app``) already read
from it correctly, but the table stayed empty forever, so the periodic SOH
updater never had a real window to work with.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from vpp.db.models import ResourceModel
    from vpp.protocols.base import ProtocolMessage
    from vpp.protocols.mqtt import MQTTAdapter

logger = logging.getLogger(__name__)

# Payload fields, beyond the required ``soc``, that map 1:1 onto
# BatteryStateRepository.record() keyword arguments.
_OPTIONAL_NUMERIC_FIELDS = ("soh", "temperature", "voltage", "current", "power")
#: Column values for fields a payload leaves out (as BatteryStateRepository.record).
_DEFAULTS = {"soh": 100.0, "temperature": 25.0, "voltage": 0.0, "current": 0.0, "power": 0.0}

#: Most messages :meth:`MQTTTelemetryIngestor.run` writes in one transaction.
MAX_BATCH = 500

#: ``YYYY-MM-DD`` then ``T`` or a space: an ISO 8601 date-time, not a number in a string.
_ISO_DATETIME = re.compile(r"^\d{4}-\d{2}-\d{2}[Tt ]")


@functools.cache
def _aware_datetime() -> Any:
    """Pydantic's datetime parser (as the HTTP endpoint uses), zone required."""
    from pydantic import AwareDatetime, TypeAdapter

    return TypeAdapter(AwareDatetime)


class InvalidTimestampError(ValueError):
    """A payload timestamp that cannot be read unambiguously."""


def parse_payload_timestamp(payload: dict[str, Any]) -> datetime | None:
    """The sample time a payload carries, in UTC, or ``None`` if it has none.

    Accepted (exactly one of):

    * ``"timestamp"``: an ISO 8601 date-time string **with** ``Z`` or a UTC
      offset (``2026-10-10T12:00:00Z``, ``2026-10-10T14:00:00.5+02:00``), or
      a JSON number of **seconds** since the Unix epoch (``1791633600``,
      fractions allowed);
    * ``"timestamp_ms"``: a JSON number of **milliseconds** since the epoch.

    A string without a zone is rejected rather than guessed (a device
    reporting local time would land hours off), as are numeric strings,
    booleans and non-finite numbers. Milliseconds sent in ``timestamp`` read
    as a date tens of thousands of years ahead and are rejected (here, or
    by the caller's future-skew check), so the units cannot be confused
    silently.

    Raises :class:`InvalidTimestampError` for anything else.
    """
    has_s, has_ms = "timestamp" in payload, "timestamp_ms" in payload
    if not has_s and not has_ms:
        return None
    if has_s and has_ms:
        raise InvalidTimestampError("send either timestamp or timestamp_ms, not both")
    key = "timestamp" if has_s else "timestamp_ms"
    raw = payload[key]
    if isinstance(raw, str) and key == "timestamp":
        if not _ISO_DATETIME.match(raw.strip()):
            raise InvalidTimestampError(f"timestamp {raw!r} is not an ISO 8601 date-time")
        from pydantic import ValidationError

        try:
            parsed: datetime = _aware_datetime().validate_python(raw.strip())
        except ValidationError:
            raise InvalidTimestampError(
                f"timestamp {raw!r} is not an ISO 8601 date-time with Z or a UTC offset"
            ) from None
        return parsed.astimezone(timezone.utc)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise InvalidTimestampError(f"{key} {raw!r} is not a number")
    seconds = float(raw) / 1000.0 if key == "timestamp_ms" else float(raw)
    if not math.isfinite(seconds):
        raise InvalidTimestampError(f"{key} {raw!r} is not finite")
    try:
        return datetime.fromtimestamp(seconds, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        raise InvalidTimestampError(
            f"{key} {raw!r} is out of range"
            + (" (milliseconds belong in timestamp_ms)" if key == "timestamp" else "")
        ) from None


@dataclass
class _Sample:
    message: ProtocolMessage
    resource_id: str
    soc: float
    #: Sample time (UTC): the payload's timestamp, else when it was received.
    ts: datetime
    #: Fresh enough to update live state (see vpp.portal.telemetry.is_live).
    live: bool
    fields: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TelemetryTopic:
    """Parsed ``vpp/{site_id}/{resource_type}/{resource_id}/{metric}`` topic."""

    site_id: str
    resource_type: str
    resource_id: str
    metric: str


def parse_telemetry_topic(topic: str) -> TelemetryTopic | None:
    """Parse a telemetry topic, or return ``None`` if it doesn't match."""
    parts = topic.split("/")
    if len(parts) != 5 or parts[0] != "vpp" or not all(parts[1:]):
        return None
    _, site_id, resource_type, resource_id, metric = parts
    return TelemetryTopic(site_id, resource_type, resource_id, metric)


class MQTTTelemetryIngestor:
    """Persists battery telemetry arriving over MQTT into ``battery_states``.

    Only messages for a ``battery``-type resource whose payload includes a
    ``soc`` field are persisted -- other battery topics (faults, config
    acks, etc.) are ignored rather than written as bogus zero-filled rows.

    Timestamps and live state follow the HTTP telemetry endpoint
    (``POST /api/v1/resources/{id}/telemetry``), using the same helpers in
    :mod:`vpp.portal.telemetry`: a sample is stamped with the payload's
    ``timestamp``/``timestamp_ms`` (see :func:`parse_payload_timestamp`) or,
    without one, with its receive time; a sample more than
    ``MAX_FUTURE_SKEW`` ahead is dropped; and only samples at most
    ``LIVE_STATE_MAX_AGE`` old count as live. For each resource the newest
    live sample carrying ``power`` sets ``ResourceModel.current_power``.
    Each live sample written publishes ``RESOURCE_UPDATED`` on the shared
    event bus (with ``current_power_kw`` on the one that set the live
    power), so WebSocket clients on ``resource_updates`` see it; older
    samples are history only and are not broadcast.
    """

    def __init__(
        self,
        adapter: MQTTAdapter,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        self._adapter = adapter
        self._session_factory = session_factory

    async def ingest_message(self, message: ProtocolMessage) -> bool:
        """Parse and persist one message. Returns True if a row was written."""
        return await self.ingest_batch([message]) == 1

    def _parse(self, message: ProtocolMessage, now: datetime) -> _Sample | None:
        parsed = parse_telemetry_topic(message.topic)
        if parsed is None or parsed.resource_type != "battery":
            return None

        payload = message.payload
        if not isinstance(payload, dict) or "soc" not in payload:
            return None

        try:
            soc = float(payload["soc"])
            kwargs = {
                field: float(payload[field])
                for field in _OPTIONAL_NUMERIC_FIELDS
                if field in payload
            }
        except (TypeError, ValueError):
            logger.warning(
                "Malformed MQTT telemetry payload on topic=%s: %r", message.topic, payload
            )
            return None
        if not all(math.isfinite(v) for v in (soc, *kwargs.values())):
            logger.warning(
                "Non-finite value in MQTT telemetry payload on topic=%s: %r",
                message.topic,
                payload,
            )
            return None

        try:
            ts = parse_payload_timestamp(payload)
        except InvalidTimestampError as exc:
            logger.warning("MQTT telemetry on topic=%s dropped: %s", message.topic, exc)
            return None
        from vpp.portal.telemetry import is_future, is_live

        if ts is None:
            ts = datetime.fromtimestamp(message.timestamp, tz=timezone.utc)
        elif is_future(ts, now):
            logger.warning(
                "MQTT telemetry on topic=%s dropped: timestamp %s is in the future",
                message.topic,
                ts.isoformat(),
            )
            return None
        return _Sample(message, parsed.resource_id, soc, ts, is_live(ts, now), kwargs)

    async def ingest_batch(self, messages: list[ProtocolMessage]) -> int:
        """Persist several messages in one transaction; returns rows written.

        One ``SELECT`` resolves every resource in the batch and one commit
        writes all rows, instead of a lookup and a commit per message: under
        load those per-message round trips, not the database, capped MQTT
        ingestion at ~90 messages/s on SQLite (``vpp loadtest``, see
        docs/performance.md). Messages that are malformed, not battery
        telemetry, or for an unknown or non-battery resource are skipped (and
        logged), as before. Each row is stamped with the payload's timestamp
        or, without one, the time its message was received; the newest fresh
        ``power`` per resource becomes its live ``current_power`` (see the
        class docstring).
        """
        now = datetime.now(timezone.utc)
        samples = [s for s in (self._parse(m, now) for m in messages) if s is not None]
        if not samples:
            return 0

        # Local imports to avoid a hard import-time dependency between
        # protocols and db (the optional `db` extra) for callers that only
        # need the parser.
        from sqlalchemy import select

        from vpp.db.engine import run_in_session
        from vpp.db.models import BatteryStateModel, ResourceModel
        from vpp.portal.telemetry import newest_live_power, publish_resource_updated

        live_power: dict[str, _Sample] = {}

        async def write(session: AsyncSession) -> list[_Sample]:
            ids = {s.resource_id for s in samples}
            batteries: dict[str, ResourceModel] = {
                r.id: r
                for r in (
                    await session.execute(
                        select(ResourceModel).where(
                            ResourceModel.id.in_(ids), ResourceModel.resource_type == "battery"
                        )
                    )
                ).scalars()
            }
            written = [s for s in samples if s.resource_id in batteries]
            session.add_all(
                BatteryStateModel(
                    resource_id=s.resource_id,
                    soc=s.soc,
                    timestamp=s.ts,
                    **{**_DEFAULTS, **s.fields},
                )
                for s in written
            )
            # Live power, decided exactly as the HTTP endpoint decides it.
            live_power.clear()
            by_resource: dict[str, list[_Sample]] = {}
            for s in written:
                if "power" in s.fields:
                    by_resource.setdefault(s.resource_id, []).append(s)
            for rid, candidates in by_resource.items():
                pairs = [(s.ts, s.fields["power"]) for s in candidates]
                newest = newest_live_power(pairs, now)
                if newest is None:
                    continue
                batteries[rid].current_power = newest[1]
                live_power[rid] = candidates[pairs.index(newest)]
            return written

        # Shielded: stopping ingestion (shutdown, lease loss) mid-commit must
        # not leak a connection that still holds SQLite's write lock.
        written = await run_in_session(self._session_factory, write)
        if len(written) < len(samples):
            stored = {id(s) for s in written}
            for s in samples:
                if id(s) not in stored:
                    logger.warning(
                        "MQTT telemetry for unknown/non-battery resource %r on topic=%s; dropping",
                        s.resource_id,
                        s.message.topic,
                    )

        for s in written:
            if not s.live:
                continue  # history only, like an HTTP backfill: no live broadcast
            data: dict[str, Any] = {"resource_id": s.resource_id, "soc": s.soc, **s.fields}
            if live_power.get(s.resource_id) is s:
                data["current_power_kw"] = s.fields["power"]
            await publish_resource_updated(data, source="mqtt.telemetry_ingestion")

        return len(written)

    async def run(self, *, max_batch: int = MAX_BATCH) -> None:
        """Drain the adapter's queue forever, persisting what has arrived.

        Waits for one message, then takes whatever else is already queued
        (up to ``max_batch``) and writes it in one transaction, so a backlog
        is worked off in a few commits while a quiet stream still writes each
        message as soon as it arrives. If a batch fails, its messages are
        retried one by one, so a single malformed or DB-erroring message is
        logged and skipped -- matching the degradation loop's "one bad tick
        doesn't kill the loop" convention -- and never takes down the whole
        ingestion stream.
        """
        async for message in self._adapter.receive_forever():
            batch = [message, *self._adapter.receive_nowait(max_batch - 1)]
            try:
                await self.ingest_batch(batch)
            except asyncio.CancelledError:
                raise
            except Exception:
                if len(batch) == 1:
                    logger.exception("MQTT telemetry ingestion failed for topic=%s", message.topic)
                    continue
                for item in batch:
                    try:
                        await self.ingest_message(item)
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        logger.exception(
                            "MQTT telemetry ingestion failed for topic=%s", item.topic
                        )
