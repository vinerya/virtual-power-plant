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
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

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


@dataclass
class _Sample:
    message: ProtocolMessage
    resource_id: str
    soc: float
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
    A successful write also publishes ``RESOURCE_UPDATED`` on the shared
    event bus, so connected WebSocket clients on the ``resource_updates``
    channel see the update live.
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

    def _parse(self, message: ProtocolMessage) -> _Sample | None:
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
        return _Sample(message, parsed.resource_id, soc, kwargs)

    async def ingest_batch(self, messages: list[ProtocolMessage]) -> int:
        """Persist several messages in one transaction; returns rows written.

        One ``SELECT`` resolves every resource in the batch and one commit
        writes all rows, instead of a lookup and a commit per message: under
        load those per-message round trips, not the database, capped MQTT
        ingestion at ~90 messages/s on SQLite (``vpp loadtest``, see
        docs/performance.md). Messages that are malformed, not battery
        telemetry, or for an unknown or non-battery resource are skipped (and
        logged), as before. Each row is stamped with the time its message was
        received.
        """
        samples = [s for s in (self._parse(m) for m in messages) if s is not None]
        if not samples:
            return 0

        # Local imports to avoid a hard import-time dependency between
        # protocols and db/events for callers that only need the parser.
        from sqlalchemy import select

        from vpp.db.engine import run_in_session
        from vpp.db.models import BatteryStateModel, ResourceModel
        from vpp.events import Event, EventType, get_event_bus

        async def write(session: AsyncSession) -> list[_Sample]:
            ids = {s.resource_id for s in samples}
            batteries = set(
                (
                    await session.execute(
                        select(ResourceModel.id).where(
                            ResourceModel.id.in_(ids), ResourceModel.resource_type == "battery"
                        )
                    )
                ).scalars()
            )
            written = [s for s in samples if s.resource_id in batteries]
            session.add_all(
                BatteryStateModel(
                    resource_id=s.resource_id,
                    soc=s.soc,
                    timestamp=datetime.fromtimestamp(s.message.timestamp, tz=timezone.utc),
                    **{**_DEFAULTS, **s.fields},
                )
                for s in written
            )
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

        bus = get_event_bus()
        for s in written:
            try:
                await bus.publish(
                    Event(
                        event_type=EventType.RESOURCE_UPDATED,
                        data={"resource_id": s.resource_id, "soc": s.soc, **s.fields},
                        source="mqtt.telemetry_ingestion",
                    )
                )
            except Exception:  # a broadcast failure must not drop the DB write
                logger.exception("Failed to publish RESOURCE_UPDATED for %s", s.resource_id)

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
