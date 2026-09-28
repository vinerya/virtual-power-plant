"""Tests for M5 MQTT battery telemetry ingestion.

Before this feature, nothing in production ever wrote to the
``battery_states`` table, so the degradation updater's DB-backed telemetry
fetch always saw an empty history.
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import datetime

import pytest

from vpp.db.engine import get_session_factory
from vpp.db.repositories import BatteryStateRepository, ResourceRepository
from vpp.events import Event, EventType, get_event_bus, reset_event_bus
from vpp.protocols.base import ProtocolMessage
from vpp.protocols.mqtt import MQTTAdapter
from vpp.protocols.telemetry_ingestion import (
    MQTTTelemetryIngestor,
    parse_telemetry_topic,
)

# ---------------------------------------------------------------------------
# Topic parsing
# ---------------------------------------------------------------------------


def test_parse_telemetry_topic_valid():
    parsed = parse_telemetry_topic("vpp/site1/battery/B001/soc")
    assert parsed is not None
    assert parsed.site_id == "site1"
    assert parsed.resource_type == "battery"
    assert parsed.resource_id == "B001"
    assert parsed.metric == "soc"


@pytest.mark.parametrize(
    "topic",
    [
        "not-vpp/site1/battery/B001/soc",  # wrong prefix
        "vpp/site1/battery/B001",  # too few segments
        "vpp/site1/battery/B001/soc/extra",  # too many segments
        "vpp//battery/B001/soc",  # empty segment
        "",
    ],
)
def test_parse_telemetry_topic_invalid(topic):
    assert parse_telemetry_topic(topic) is None


# ---------------------------------------------------------------------------
# MQTTAdapter.receive_forever
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_receive_forever_drains_queue_in_order():
    adapter = MQTTAdapter()
    msg1 = ProtocolMessage(topic="vpp/site1/battery/B001/soc", payload={"soc": 0.5})
    msg2 = ProtocolMessage(topic="vpp/site1/battery/B001/soc", payload={"soc": 0.6})
    await adapter._message_queue.put(msg1)
    await adapter._message_queue.put(msg2)

    gen = adapter.receive_forever()
    first = await gen.__anext__()
    second = await gen.__anext__()
    assert first is msg1
    assert second is msg2
    await gen.aclose()


@pytest.mark.asyncio
async def test_receive_forever_blocks_until_message_arrives():
    adapter = MQTTAdapter()
    gen = adapter.receive_forever()

    # Don't cancel an in-flight __anext__() -- doing so leaves the async
    # generator closed, so a later call raises StopAsyncIteration instead
    # of resuming. Instead: start it, confirm it's still pending, then let
    # it complete naturally once a message is queued.
    task = asyncio.create_task(gen.__anext__())
    await asyncio.sleep(0.05)
    assert not task.done()

    msg = ProtocolMessage(topic="vpp/site1/battery/B001/soc", payload={"soc": 0.5})
    await adapter._message_queue.put(msg)
    received = await asyncio.wait_for(task, timeout=1.0)
    assert received is msg
    await gen.aclose()


# ---------------------------------------------------------------------------
# MQTTTelemetryIngestor.ingest_message
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_message_persists_battery_state(db_session, app):
    reset_event_bus()
    battery = await ResourceRepository.create(
        db_session,
        name=f"mqtt-battery-{datetime.now().timestamp()}",
        resource_type="battery",
        rated_power=100.0,
    )
    await db_session.commit()

    adapter = MQTTAdapter()
    ingestor = MQTTTelemetryIngestor(adapter, get_session_factory())
    message = ProtocolMessage(
        topic=f"vpp/site1/battery/{battery.id}/soc",
        payload={"soc": 0.72, "voltage": 48.2, "temperature": 26.5},
    )

    wrote = await ingestor.ingest_message(message)
    assert wrote is True

    rows = await BatteryStateRepository.get_latest(db_session, battery.id)
    assert len(rows) == 1
    assert rows[0].soc == pytest.approx(0.72)
    assert rows[0].voltage == pytest.approx(48.2)
    assert rows[0].temperature == pytest.approx(26.5)
    # Defaults apply for fields absent from the payload.
    assert rows[0].current == 0.0
    assert rows[0].power == 0.0


@pytest.mark.asyncio
async def test_ingest_message_publishes_resource_updated_event(db_session, app):
    reset_event_bus()
    bus = get_event_bus()

    battery = await ResourceRepository.create(
        db_session,
        name=f"mqtt-battery-evt-{datetime.now().timestamp()}",
        resource_type="battery",
        rated_power=100.0,
    )
    await db_session.commit()

    received: list[Event] = []

    async def _handler(event: Event) -> None:
        received.append(event)

    bus.subscribe(_handler)

    adapter = MQTTAdapter()
    ingestor = MQTTTelemetryIngestor(adapter, get_session_factory())
    await ingestor.ingest_message(
        ProtocolMessage(
            topic=f"vpp/site1/battery/{battery.id}/soc",
            payload={"soc": 0.5},
        )
    )

    assert len(received) == 1
    assert received[0].event_type == EventType.RESOURCE_UPDATED
    assert received[0].data["resource_id"] == battery.id
    assert received[0].data["soc"] == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_ingest_message_skips_non_battery_resource(db_session, app):
    solar = await ResourceRepository.create(
        db_session,
        name=f"mqtt-solar-{datetime.now().timestamp()}",
        resource_type="solar",
        rated_power=10.0,
    )
    await db_session.commit()

    adapter = MQTTAdapter()
    ingestor = MQTTTelemetryIngestor(adapter, get_session_factory())
    wrote = await ingestor.ingest_message(
        ProtocolMessage(topic=f"vpp/site1/solar/{solar.id}/power", payload={"soc": 0.9})
    )
    assert wrote is False


@pytest.mark.asyncio
async def test_ingest_message_skips_unknown_resource(app):
    adapter = MQTTAdapter()
    ingestor = MQTTTelemetryIngestor(adapter, get_session_factory())
    wrote = await ingestor.ingest_message(
        ProtocolMessage(topic="vpp/site1/battery/does-not-exist/soc", payload={"soc": 0.5})
    )
    assert wrote is False


@pytest.mark.asyncio
async def test_ingest_message_skips_missing_soc(db_session, app):
    battery = await ResourceRepository.create(
        db_session,
        name=f"mqtt-nosoc-{datetime.now().timestamp()}",
        resource_type="battery",
        rated_power=100.0,
    )
    await db_session.commit()

    adapter = MQTTAdapter()
    ingestor = MQTTTelemetryIngestor(adapter, get_session_factory())
    wrote = await ingestor.ingest_message(
        ProtocolMessage(
            topic=f"vpp/site1/battery/{battery.id}/fault",
            payload={"code": "OVER_TEMP"},
        )
    )
    assert wrote is False


@pytest.mark.asyncio
async def test_ingest_message_skips_malformed_topic(app):
    adapter = MQTTAdapter()
    ingestor = MQTTTelemetryIngestor(adapter, get_session_factory())
    wrote = await ingestor.ingest_message(
        ProtocolMessage(topic="not-a-vpp-topic", payload={"soc": 0.5})
    )
    assert wrote is False


@pytest.mark.asyncio
async def test_ingest_message_handles_non_numeric_soc(db_session, app):
    battery = await ResourceRepository.create(
        db_session,
        name=f"mqtt-badsoc-{datetime.now().timestamp()}",
        resource_type="battery",
        rated_power=100.0,
    )
    await db_session.commit()

    adapter = MQTTAdapter()
    ingestor = MQTTTelemetryIngestor(adapter, get_session_factory())
    wrote = await ingestor.ingest_message(
        ProtocolMessage(
            topic=f"vpp/site1/battery/{battery.id}/soc",
            payload={"soc": "not-a-number"},
        )
    )
    assert wrote is False


# ---------------------------------------------------------------------------
# MQTTTelemetryIngestor.run
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_ingests_queued_messages_until_cancelled(db_session, app):
    reset_event_bus()
    battery = await ResourceRepository.create(
        db_session,
        name=f"mqtt-run-{datetime.now().timestamp()}",
        resource_type="battery",
        rated_power=100.0,
    )
    await db_session.commit()

    adapter = MQTTAdapter()
    ingestor = MQTTTelemetryIngestor(adapter, get_session_factory())

    task = asyncio.create_task(ingestor.run())
    try:
        await adapter._message_queue.put(
            ProtocolMessage(topic=f"vpp/site1/battery/{battery.id}/soc", payload={"soc": 0.3})
        )
        await adapter._message_queue.put(
            ProtocolMessage(topic=f"vpp/site1/battery/{battery.id}/soc", payload={"soc": 0.4})
        )
        # Give the loop a moment to drain both messages.
        for _ in range(50):
            rows = await BatteryStateRepository.get_latest(db_session, battery.id, limit=10)
            if len(rows) >= 2:
                break
            await asyncio.sleep(0.02)
        else:
            pytest.fail("ingestor.run() did not persist queued messages in time")
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    rows = await BatteryStateRepository.get_latest(db_session, battery.id, limit=10)
    assert len(rows) == 2


@pytest.mark.asyncio
async def test_run_skips_bad_message_and_continues(db_session, app):
    reset_event_bus()
    battery = await ResourceRepository.create(
        db_session,
        name=f"mqtt-run-bad-{datetime.now().timestamp()}",
        resource_type="battery",
        rated_power=100.0,
    )
    await db_session.commit()

    adapter = MQTTAdapter()
    ingestor = MQTTTelemetryIngestor(adapter, get_session_factory())

    task = asyncio.create_task(ingestor.run())
    try:
        await adapter._message_queue.put(
            ProtocolMessage(topic="garbage-topic", payload={"soc": 1.0})
        )
        await adapter._message_queue.put(
            ProtocolMessage(topic=f"vpp/site1/battery/{battery.id}/soc", payload={"soc": 0.8})
        )
        for _ in range(50):
            rows = await BatteryStateRepository.get_latest(db_session, battery.id, limit=10)
            if rows:
                break
            await asyncio.sleep(0.02)
        else:
            pytest.fail("ingestor.run() stalled after a malformed message")
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    rows = await BatteryStateRepository.get_latest(db_session, battery.id, limit=10)
    assert len(rows) == 1
    assert rows[0].soc == pytest.approx(0.8)
