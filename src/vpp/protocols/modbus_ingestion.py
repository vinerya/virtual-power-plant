"""Modbus inverter/meter telemetry ingestion.

Bridges :class:`~vpp.protocols.modbus.ModbusAdapter` register polls into
live resource state. Before this module existed, nothing in production
ever instantiated a ``ModbusAdapter`` at all -- the adapter's own polling
loop (:meth:`ModbusAdapter.poll_once`, ``_poll_loop``) and its correct use
of the base class's ``subscribe``/``_dispatch`` pub-sub were already fully
built and tested in isolation, but never actually connected to a real
device or a resource in the running app.

Unlike MQTT (one broker, topic-routed to many resources), Modbus is
inherently per-physical-device: each inverter/meter is its own TCP or RTU
connection. Rather than a new DB column or schema migration, a resource
opts into Modbus polling by storing its connection config in its own
free-form ``metadata`` under a ``"modbus"`` key, e.g.::

    {
      "modbus": {
        "mode": "tcp", "host": "192.168.1.50", "port": 502,
        "device_profile": "sma_sunnyboy", "poll_interval_s": 5.0,
        "power_register": "ac_power"
      }
    }

``power_register`` (default ``"ac_power"``) names which polled register
this module writes to ``ResourceModel.current_power`` (converted W -> kW).
Every other key is passed straight through to
:meth:`ModbusAdapter.configure`.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vpp.protocols.base import ProtocolMessage

logger = logging.getLogger(__name__)

DEFAULT_POWER_REGISTER = "ac_power"


def modbus_config_for_resource(metadata: dict[str, Any]) -> dict[str, Any] | None:
    """Return the resource's ``modbus`` config dict, or None if absent/invalid."""
    config = metadata.get("modbus")
    if not isinstance(config, dict):
        return None
    return config


class ModbusResourcePersister:
    """Persists one resource's polled Modbus values onto its live state.

    Subscribed to a single :class:`ModbusAdapter` instance's dispatch
    (wildcard ``"*"``), this updates ``ResourceModel.current_power`` from
    the configured power register on every poll and publishes
    ``RESOURCE_UPDATED`` so connected WebSocket clients see it live --
    mirroring :class:`~vpp.protocols.telemetry_ingestion.MQTTTelemetryIngestor`'s
    event-publishing convention for the battery/MQTT case.
    """

    def __init__(
        self,
        resource_id: str,
        session_factory,
        power_register: str = DEFAULT_POWER_REGISTER,
    ) -> None:
        self._resource_id = resource_id
        self._session_factory = session_factory
        self._power_register = power_register

    async def handle_message(self, message: ProtocolMessage) -> None:
        watts = message.payload.get(self._power_register)
        if watts is None:
            return
        try:
            kw = float(watts) / 1000.0
        except (TypeError, ValueError):
            logger.warning(
                "Non-numeric Modbus register %r=%r for resource %s; dropping",
                self._power_register,
                watts,
                self._resource_id,
            )
            return

        # Local imports to avoid a hard import-time dependency between
        # protocols and db/events for callers that only need the config parser.
        from vpp.db.repositories import ResourceRepository
        from vpp.events import Event, EventType, get_event_bus

        async with self._session_factory() as session:
            updated = await ResourceRepository.update(session, self._resource_id, current_power=kw)
            if updated is None:
                logger.warning(
                    "Modbus telemetry for unknown resource %r; dropping", self._resource_id
                )
                return
            # Keep history too, so GET /resources/{id}/metrics has a series
            # for Modbus devices (battery_states only covers MQTT batteries).
            from vpp.portal.telemetry import record_samples

            await record_samples(
                session,
                self._resource_id,
                [(datetime.fromtimestamp(message.timestamp, tz=timezone.utc), kw, None)],
                source="modbus",
            )
            await session.commit()

        try:
            await get_event_bus().publish(
                Event(
                    event_type=EventType.RESOURCE_UPDATED,
                    data={"resource_id": self._resource_id, "current_power_kw": kw},
                    source="modbus.telemetry_ingestion",
                )
            )
        except Exception:  # a broadcast failure must not drop the DB write
            logger.exception("Failed to publish RESOURCE_UPDATED for %s", self._resource_id)
