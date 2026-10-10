"""Tests for M5 MQTT battery telemetry ingestion.

Before this feature, nothing in production ever wrote to the
``battery_states`` table, so the degradation updater's DB-backed telemetry
fetch always saw an empty history.
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import datetime, timezone

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


class _PahoMessage:
    def __init__(self, topic: str, payload: bytes) -> None:
        self.topic = topic
        self.payload = payload
        self.qos = 1


@pytest.mark.asyncio
async def test_message_from_paho_thread_wakes_the_event_loop():
    """Regression (found by `vpp loadtest`): paho delivers on its own thread.

    The callback used to call ``asyncio.Queue.put_nowait`` from that thread,
    which does not wake an idle event loop, so on a quiet server a message
    sat in the queue until some unrelated timer fired (p50 ~80 ms, p95
    ~300 ms publish-to-persist at 1 msg/s, seconds when fully idle).
    """
    import threading
    import time

    adapter = MQTTAdapter()
    adapter._loop = asyncio.get_running_loop()  # set by connect()
    raw = _PahoMessage("vpp/site1/battery/B001/soc", b'{"soc": 55.0}')
    thread = threading.Timer(0.05, adapter._on_message, args=(None, None, raw))
    start = time.monotonic()
    thread.start()
    # Nothing else is scheduled: the loop sleeps until the wait_for timeout
    # unless the producer thread wakes it.
    msg = await asyncio.wait_for(adapter._message_queue.get(), timeout=5.0)
    elapsed = time.monotonic() - start
    thread.join()
    assert msg.payload == {"soc": 55.0}
    assert elapsed < 1.0


@pytest.mark.asyncio
async def test_own_disconnect_is_not_reported_as_unexpected(caplog):
    adapter = MQTTAdapter()

    class _Client:
        def loop_stop(self) -> None:
            pass

        def disconnect(self) -> None:  # paho reports the disconnect via the callback
            adapter._on_disconnect(self, None, None, 0)

    adapter._client = _Client()
    with caplog.at_level("WARNING", logger="vpp.protocols.mqtt"):
        await adapter.disconnect()
    assert "unexpectedly" not in caplog.text
    assert not adapter.is_connected


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


def test_queue_full_drops_are_counted_and_logged_once_per_interval(caplog):
    """Under sustained overload a warning per dropped message flooded the log."""
    adapter = MQTTAdapter()
    for i in range(adapter._message_queue.maxsize):
        adapter._enqueue(ProtocolMessage(topic=f"vpp/s/battery/B{i}/soc", payload={}))
    with caplog.at_level("WARNING", logger="vpp.protocols.mqtt"):
        for i in range(50):
            adapter._enqueue(ProtocolMessage(topic=f"vpp/s/battery/X{i}/soc", payload={}))
    assert adapter.dropped == 50
    assert adapter.metrics.errors == 50
    assert len([r for r in caplog.records if "queue full" in r.getMessage()]) == 1


def test_received_messages_are_counted_in_adapter_metrics():
    """GET /api/v1/protocols reported messages_received=0 for MQTT however
    much telemetry arrived, which hid whether a broker was delivering."""
    adapter = MQTTAdapter()
    assert adapter.metrics.messages_received == 0
    assert adapter.metrics.last_message_at is None
    for i in range(3):
        adapter._enqueue(ProtocolMessage(topic=f"vpp/s/battery/B{i}/state", payload={"soc": 50}))
    assert adapter.metrics.messages_received == 3
    assert adapter.metrics.last_message_at is not None
    assert adapter.receive_nowait(10)[0].topic == "vpp/s/battery/B0/state"


@pytest.mark.asyncio
async def test_ingest_batch_writes_valid_rows_in_one_transaction(db_session, app):
    """Regression for the ingestion throughput hot spot found by `vpp loadtest`:
    a backlog is written with one lookup and one commit, not one per message."""
    from sqlalchemy import event

    reset_event_bus()
    stamp = datetime.now().timestamp()
    base = stamp - 100  # receive times: recent, so the samples are live
    battery = await ResourceRepository.create(
        db_session, name=f"mqtt-batch-{stamp}", resource_type="battery", rated_power=5.0
    )
    solar = await ResourceRepository.create(
        db_session, name=f"mqtt-batch-pv-{stamp}", resource_type="solar", rated_power=5.0
    )
    await db_session.commit()
    received: list[Event] = []

    async def on_event(e: Event) -> None:
        received.append(e)

    get_event_bus().subscribe(on_event, event_types={EventType.RESOURCE_UPDATED})
    messages = [
        ProtocolMessage(
            topic=f"vpp/s/battery/{battery.id}/soc", payload={"soc": 40.0 + i}, timestamp=base + i
        )
        for i in range(50)
    ] + [
        ProtocolMessage(topic=f"vpp/s/battery/{solar.id}/soc", payload={"soc": 1.0}),
        ProtocolMessage(topic="vpp/s/battery/missing/soc", payload={"soc": 1.0}),
        ProtocolMessage(topic=f"vpp/s/battery/{battery.id}/soc", payload={"soc": "bad"}),
    ]
    engine = get_session_factory().kw["bind"]
    statements: list[str] = []

    def count(conn, cursor, statement, *args):
        statements.append(statement.split()[0].upper())

    event.listen(engine.sync_engine, "before_cursor_execute", count)
    try:
        written = await MQTTTelemetryIngestor(MQTTAdapter(), get_session_factory()).ingest_batch(
            messages
        )
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", count)

    assert written == 50
    assert statements.count("SELECT") == 1
    assert statements.count("INSERT") <= 2  # executemany / insertmanyvalues
    rows = await BatteryStateRepository.get_latest(db_session, battery.id, limit=100)
    assert len(rows) == 50
    assert sorted(r.soc for r in rows) == [40.0 + i for i in range(50)]
    # Stamped with the time each message was received, not the commit time.
    newest = max(rows, key=lambda r: r.timestamp)
    assert newest.timestamp.replace(tzinfo=timezone.utc).timestamp() == pytest.approx(
        base + 49, abs=1
    )
    assert [e.data["soc"] for e in received] == [40.0 + i for i in range(50)]


@pytest.mark.asyncio
async def test_run_writes_a_backlog_in_batches(db_session, app):
    reset_event_bus()
    battery = await ResourceRepository.create(
        db_session,
        name=f"mqtt-backlog-{datetime.now().timestamp()}",
        resource_type="battery",
        rated_power=5.0,
    )
    await db_session.commit()
    adapter = MQTTAdapter()
    for i in range(300):
        adapter._message_queue.put_nowait(
            ProtocolMessage(topic=f"vpp/s/battery/{battery.id}/soc", payload={"soc": 10.0 + i})
        )
    ingestor = MQTTTelemetryIngestor(adapter, get_session_factory())
    batches: list[int] = []
    original = ingestor.ingest_batch

    async def spy(messages):
        batches.append(len(messages))
        return await original(messages)

    ingestor.ingest_batch = spy  # type: ignore[method-assign]
    task = asyncio.create_task(ingestor.run(max_batch=100))
    try:
        for _ in range(200):
            if sum(batches) >= 300:
                break
            await asyncio.sleep(0.02)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    assert batches == [100, 100, 100]
    rows = await BatteryStateRepository.get_latest(db_session, battery.id, limit=500)
    assert len(rows) == 300
