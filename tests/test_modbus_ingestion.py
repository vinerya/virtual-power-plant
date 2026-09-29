"""Tests for Modbus inverter/meter telemetry ingestion.

Before this feature, nothing in production ever instantiated a
ModbusAdapter -- its polling loop and pub-sub dispatch were fully built
and correct in isolation, but never connected to a real device or a
resource in the running app.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from vpp.db.engine import get_session_factory
from vpp.db.repositories import ResourceRepository
from vpp.events import Event, EventType, get_event_bus, reset_event_bus
from vpp.protocols.base import ProtocolMessage
from vpp.protocols.modbus_ingestion import (
    DEFAULT_POWER_REGISTER,
    ModbusResourcePersister,
    modbus_config_for_resource,
)

# ---------------------------------------------------------------------------
# modbus_config_for_resource
# ---------------------------------------------------------------------------


def test_modbus_config_for_resource_present():
    metadata = {"modbus": {"host": "192.168.1.50", "port": 502}}
    assert modbus_config_for_resource(metadata) == {"host": "192.168.1.50", "port": 502}


def test_modbus_config_for_resource_absent():
    assert modbus_config_for_resource({}) is None
    assert modbus_config_for_resource({"other": "stuff"}) is None


def test_modbus_config_for_resource_wrong_type():
    assert modbus_config_for_resource({"modbus": "not-a-dict"}) is None
    assert modbus_config_for_resource({"modbus": ["a", "list"]}) is None


# ---------------------------------------------------------------------------
# ModbusResourcePersister.handle_message
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_message_persists_current_power(db_session, app):
    reset_event_bus()
    resource = await ResourceRepository.create(
        db_session,
        name=f"modbus-inverter-{datetime.now().timestamp()}",
        resource_type="solar",
        rated_power=5.0,
    )
    await db_session.commit()

    persister = ModbusResourcePersister(resource.id, get_session_factory())
    await persister.handle_message(
        ProtocolMessage(
            topic="modbus/SMA Sunny Boy", payload={"ac_power": 2500.0}, source="modbus"
        )
    )

    # handle_message commits through its own session; db_session's identity
    # map still holds the pre-update object, so refresh before asserting.
    await db_session.refresh(resource)
    assert resource.current_power == pytest.approx(2.5)


@pytest.mark.asyncio
async def test_handle_message_uses_custom_power_register(db_session, app):
    reset_event_bus()
    resource = await ResourceRepository.create(
        db_session,
        name=f"modbus-meter-{datetime.now().timestamp()}",
        resource_type="solar",
        rated_power=5.0,
    )
    await db_session.commit()

    persister = ModbusResourcePersister(
        resource.id, get_session_factory(), power_register="power_total"
    )
    await persister.handle_message(
        ProtocolMessage(
            topic="modbus/Generic Power Meter",
            payload={"power_total": 1000.0},
            source="modbus",
        )
    )

    await db_session.refresh(resource)
    assert resource.current_power == pytest.approx(1.0)


def test_default_power_register_is_ac_power():
    assert DEFAULT_POWER_REGISTER == "ac_power"


@pytest.mark.asyncio
async def test_handle_message_skips_missing_register(db_session, app):
    resource = await ResourceRepository.create(
        db_session,
        name=f"modbus-nokey-{datetime.now().timestamp()}",
        resource_type="solar",
        rated_power=5.0,
    )
    resource.current_power = 9.0
    await db_session.commit()

    persister = ModbusResourcePersister(resource.id, get_session_factory())
    await persister.handle_message(
        ProtocolMessage(topic="modbus/x", payload={"dc_power": 100.0}, source="modbus")
    )

    await db_session.refresh(resource)
    assert resource.current_power == pytest.approx(9.0)  # untouched


@pytest.mark.asyncio
async def test_handle_message_skips_non_numeric_value(db_session, app):
    resource = await ResourceRepository.create(
        db_session,
        name=f"modbus-badval-{datetime.now().timestamp()}",
        resource_type="solar",
        rated_power=5.0,
    )
    resource.current_power = 9.0
    await db_session.commit()

    persister = ModbusResourcePersister(resource.id, get_session_factory())
    await persister.handle_message(
        ProtocolMessage(topic="modbus/x", payload={"ac_power": "not-a-number"}, source="modbus")
    )

    await db_session.refresh(resource)
    assert resource.current_power == pytest.approx(9.0)  # untouched


@pytest.mark.asyncio
async def test_handle_message_skips_unknown_resource():
    persister = ModbusResourcePersister("does-not-exist", get_session_factory())
    # Must not raise.
    await persister.handle_message(
        ProtocolMessage(topic="modbus/x", payload={"ac_power": 100.0}, source="modbus")
    )


@pytest.mark.asyncio
async def test_handle_message_publishes_resource_updated_event(db_session, app):
    reset_event_bus()
    bus = get_event_bus()
    resource = await ResourceRepository.create(
        db_session,
        name=f"modbus-evt-{datetime.now().timestamp()}",
        resource_type="solar",
        rated_power=5.0,
    )
    await db_session.commit()

    received: list[Event] = []

    async def _handler(event: Event) -> None:
        received.append(event)

    bus.subscribe(_handler)

    persister = ModbusResourcePersister(resource.id, get_session_factory())
    await persister.handle_message(
        ProtocolMessage(topic="modbus/x", payload={"ac_power": 3000.0}, source="modbus")
    )

    assert len(received) == 1
    assert received[0].event_type == EventType.RESOURCE_UPDATED
    assert received[0].data["resource_id"] == resource.id
    assert received[0].data["current_power_kw"] == pytest.approx(3.0)


@pytest.mark.asyncio
async def test_handle_message_records_telemetry_history(db_session, app):
    """Each poll is also appended to resource_telemetry for /metrics history."""
    from sqlalchemy import select

    from vpp.db.models import ResourceTelemetryModel

    reset_event_bus()
    resource = await ResourceRepository.create(
        db_session,
        name=f"modbus-history-{datetime.now().timestamp()}",
        resource_type="solar",
        rated_power=5.0,
    )
    await db_session.commit()

    persister = ModbusResourcePersister(resource.id, get_session_factory())
    msg = ProtocolMessage(topic="modbus/x", payload={"ac_power": 1200.0}, source="modbus")
    await persister.handle_message(msg)

    rows = (
        (
            await db_session.execute(
                select(ResourceTelemetryModel).where(
                    ResourceTelemetryModel.resource_id == resource.id
                )
            )
        )
        .scalars()
        .all()
    )
    assert len(rows) == 1
    assert rows[0].power_kw == pytest.approx(1.2)
    assert rows[0].source == "modbus"
    assert rows[0].state_of_charge is None


@pytest.mark.asyncio
async def test_handle_message_records_polled_soc_for_the_optimizer(db_session, app):
    """``soc_register`` (e.g. SunSpec ChaState, %) lands in resource telemetry,
    where the dispatch optimiser reads a battery's state of charge from."""
    from vpp.portal.telemetry import latest_soc

    reset_event_bus()
    resource = await ResourceRepository.create(
        db_session,
        name=f"modbus-battery-{datetime.now().timestamp()}",
        resource_type="battery",
        rated_power=5.0,
    )
    await db_session.commit()
    persister = ModbusResourcePersister(
        resource.id, get_session_factory(), power_register="battery_w", soc_register="cha_state"
    )
    # SoC only (no power register polled): recorded, current_power untouched.
    await persister.handle_message(
        ProtocolMessage(topic="modbus/custom", payload={"cha_state": 51.9}, source="modbus")
    )
    assert (await latest_soc(db_session, [resource.id]))[resource.id] == pytest.approx(0.519)
    # An out-of-range SoC is ignored; with no power either, nothing is recorded.
    await persister.handle_message(
        ProtocolMessage(topic="modbus/custom", payload={"cha_state": 250.0}, source="modbus")
    )
    assert (await latest_soc(db_session, [resource.id]))[resource.id] == pytest.approx(0.519)
    await db_session.refresh(resource)
    assert resource.current_power == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_cancelled_poll_never_leaves_the_sqlite_write_lock_held(tmp_path):
    """Stopping the ingestion loop mid-write must not leak a locked connection.

    Cancelling SQLAlchemy mid-statement used to invalidate the connection
    without a rollback; SQLite then kept the write lock until the connection
    was garbage collected, and every other writer failed with "database is
    locked". The persister now lets the write finish (commit or roll back,
    close) before the cancellation proceeds.
    """
    import asyncio
    import contextlib
    import sqlite3

    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from vpp.db.base import Base
    from vpp.db.models import ResourceModel

    path = tmp_path / "lock.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        factory = async_sessionmaker(engine, expire_on_commit=False)
        async with factory() as s:
            s.add(ResourceModel(id="r1", name="r1", resource_type="solar", rated_power=10.0))
            await s.commit()
        persister = ModbusResourcePersister("r1", factory)
        # Cancel at many points of the write (before, during and after the
        # UPDATE/INSERT and the COMMIT); another writer is never blocked.
        for step in range(40):
            task = asyncio.ensure_future(
                persister.handle_message(
                    ProtocolMessage(topic="t", payload={"ac_power": 1000.0}, source="modbus")
                )
            )
            for _ in range(step):
                await asyncio.sleep(0.00005)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            other = sqlite3.connect(path, timeout=0.2)
            try:
                other.execute("UPDATE resources SET online = 1")
                other.commit()
            finally:
                other.close()
    finally:
        await engine.dispose()
