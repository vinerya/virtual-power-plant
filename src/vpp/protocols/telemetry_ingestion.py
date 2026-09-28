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

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from vpp.protocols.base import ProtocolMessage
    from vpp.protocols.mqtt import MQTTAdapter

logger = logging.getLogger(__name__)

# Payload fields, beyond the required ``soc``, that map 1:1 onto
# BatteryStateRepository.record() keyword arguments.
_OPTIONAL_NUMERIC_FIELDS = ("soh", "temperature", "voltage", "current", "power")


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
        parsed = parse_telemetry_topic(message.topic)
        if parsed is None or parsed.resource_type != "battery":
            return False

        payload = message.payload
        if not isinstance(payload, dict) or "soc" not in payload:
            return False

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
            return False

        # Local imports to avoid a hard import-time dependency between
        # protocols and db/events for callers that only need the parser.
        from vpp.db.repositories import BatteryStateRepository, ResourceRepository
        from vpp.events import Event, EventType, get_event_bus

        async with self._session_factory() as session:
            resource = await ResourceRepository.get_by_id(session, parsed.resource_id)
            if resource is None or resource.resource_type != "battery":
                logger.warning(
                    "MQTT telemetry for unknown/non-battery resource %r on topic=%s; dropping",
                    parsed.resource_id,
                    message.topic,
                )
                return False

            await BatteryStateRepository.record(
                session, resource_id=parsed.resource_id, soc=soc, **kwargs
            )
            await session.commit()

        try:
            await get_event_bus().publish(
                Event(
                    event_type=EventType.RESOURCE_UPDATED,
                    data={"resource_id": parsed.resource_id, "soc": soc, **kwargs},
                    source="mqtt.telemetry_ingestion",
                )
            )
        except Exception:  # a broadcast failure must not drop the DB write
            logger.exception("Failed to publish RESOURCE_UPDATED for %s", parsed.resource_id)

        return True

    async def run(self) -> None:
        """Drain the adapter's queue forever, persisting each message.

        A single malformed or DB-erroring message is logged and skipped --
        matching the degradation loop's "one bad tick doesn't kill the
        loop" convention -- so it never takes down the whole ingestion
        stream.
        """
        async for message in self._adapter.receive_forever():
            try:
                await self.ingest_message(message)
            except Exception:
                logger.exception("MQTT telemetry ingestion failed for topic=%s", message.topic)
