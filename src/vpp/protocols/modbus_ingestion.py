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
``soc_register`` (optional) names a polled state-of-charge register in
percent (e.g. SunSpec model 124 ``ChaState``, see
:func:`vpp.protocols.modbus.sunspec_model_124_registers`); its value is
recorded in the resource telemetry, where the dispatch optimiser reads a
battery's state of charge from. Every other key is passed straight through
to :meth:`ModbusAdapter.configure`.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vpp.protocols.base import ProtocolMessage

logger = logging.getLogger(__name__)

DEFAULT_POWER_REGISTER = "ac_power"
#: Keys of a resource's ``modbus`` config consumed by the VPP, not the adapter.
NON_ADAPTER_KEYS = ("power_register", "soc_register", "control")


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
        soc_register: str | None = None,
    ) -> None:
        self._resource_id = resource_id
        self._session_factory = session_factory
        self._power_register = power_register
        self._soc_register = soc_register

    def _number(self, payload: dict[str, Any], register: str | None) -> float | None:
        if register is None:
            return None
        value = payload.get(register)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            logger.warning(
                "Non-numeric Modbus register %r=%r for resource %s; dropping",
                register,
                value,
                self._resource_id,
            )
            return None

    async def handle_message(self, message: ProtocolMessage) -> None:
        watts = self._number(message.payload, self._power_register)
        kw = watts / 1000.0 if watts is not None else None
        soc_pct = self._number(message.payload, self._soc_register)
        soc = soc_pct / 100.0 if soc_pct is not None and 0.0 <= soc_pct <= 100.0 else None
        if kw is None and soc is None:
            return

        # Cancelling a task while it awaits a DB call interrupts SQLAlchemy
        # mid-statement: the connection is invalidated without a rollback and
        # (on SQLite) keeps the write lock until it is garbage collected,
        # blocking every other writer. So the write runs in its own task that
        # a cancelled poll (e.g. the ingestion loop stopping) waits for: it
        # commits or rolls back and closes its session, then the
        # cancellation proceeds.
        write = asyncio.ensure_future(self._persist(message, kw, soc))
        try:
            await asyncio.shield(write)
        except asyncio.CancelledError:
            with contextlib.suppress(Exception):
                await write
            raise

    async def _persist(
        self, message: ProtocolMessage, kw: float | None, soc: float | None
    ) -> None:
        # Local imports to avoid a hard import-time dependency between
        # protocols and db/events for callers that only need the config parser.
        from vpp.db.repositories import ResourceRepository
        from vpp.events import Event, EventType, get_event_bus

        async with self._session_factory() as session:
            if kw is not None:
                updated = await ResourceRepository.update(
                    session, self._resource_id, current_power=kw
                )
            else:
                updated = await ResourceRepository.get_by_id(session, self._resource_id)
            if updated is None:
                logger.warning(
                    "Modbus telemetry for unknown resource %r; dropping", self._resource_id
                )
                return
            # Keep history too, so GET /resources/{id}/metrics has a series
            # for Modbus devices (battery_states only covers MQTT batteries),
            # and the optimiser finds the polled state of charge.
            from vpp.portal.telemetry import record_samples

            power = kw if kw is not None else float(updated.current_power or 0.0)
            await record_samples(
                session,
                self._resource_id,
                [(datetime.fromtimestamp(message.timestamp, tz=timezone.utc), power, soc)],
                source="modbus",
            )
            await session.commit()

        data: dict[str, Any] = {"resource_id": self._resource_id, "current_power_kw": power}
        if soc is not None:
            data["soc"] = soc
        try:
            await get_event_bus().publish(
                Event(
                    event_type=EventType.RESOURCE_UPDATED,
                    data=data,
                    source="modbus.telemetry_ingestion",
                )
            )
        except Exception:  # a broadcast failure must not drop the DB write
            logger.exception("Failed to publish RESOURCE_UPDATED for %s", self._resource_id)
