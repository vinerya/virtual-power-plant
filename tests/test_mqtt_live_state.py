"""MQTT telemetry: payload timestamps and live ``current_power``.

MQTT used to stamp every sample with its receive time and never touch a
resource's live power, unlike ``POST /api/v1/resources/{id}/telemetry``.
Both paths now share the timestamp guards and live-state rule in
``vpp.portal.telemetry``.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from vpp.db.engine import get_session_factory
from vpp.db.models import ResourceModel
from vpp.db.repositories import BatteryStateRepository, ResourceRepository
from vpp.events import Event, EventType, get_event_bus, reset_event_bus
from vpp.protocols.base import ProtocolMessage
from vpp.protocols.mqtt import MQTTAdapter
from vpp.protocols.telemetry_ingestion import (
    InvalidTimestampError,
    MQTTTelemetryIngestor,
    parse_payload_timestamp,
)

# ---------------------------------------------------------------------------
# parse_payload_timestamp
# ---------------------------------------------------------------------------

_T = datetime(2026, 10, 10, 12, 0, 0, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({}, None),
        ({"timestamp": "2026-10-10T12:00:00Z"}, _T),
        ({"timestamp": "2026-10-10T12:00:00z"}, _T),
        ({"timestamp": "2026-10-10 12:00:00Z"}, _T),
        ({"timestamp": "2026-10-10T14:00:00+02:00"}, _T),
        ({"timestamp": "2026-10-10T07:00:00.250-05:00"}, _T + timedelta(milliseconds=250)),
        ({"timestamp": _T.timestamp()}, _T),
        ({"timestamp": int(_T.timestamp())}, _T),
        ({"timestamp": _T.timestamp() + 0.5}, _T + timedelta(milliseconds=500)),
        ({"timestamp_ms": int(_T.timestamp() * 1000)}, _T),
        ({"timestamp_ms": _T.timestamp() * 1000 + 250}, _T + timedelta(milliseconds=250)),
    ],
)
def test_parse_payload_timestamp_accepts(payload, expected):
    parsed = parse_payload_timestamp(payload)
    assert parsed == expected
    if parsed is not None:
        assert parsed.utcoffset() == timedelta(0)


@pytest.mark.parametrize(
    "payload",
    [
        {"timestamp": "2026-10-10T12:00:00"},  # no zone: ambiguous, not guessed
        {"timestamp": "2026-10-10"},
        {"timestamp": "1791633600"},  # a number in a string
        {"timestamp": "yesterday"},
        {"timestamp": "2026-13-40T12:00:00Z"},
        {"timestamp": True},
        {"timestamp": None},
        {"timestamp": float("nan")},
        {"timestamp_ms": float("inf")},
        {"timestamp_ms": "1791633600000"},
        {"timestamp": 1e300},
        {"timestamp": 1.0, "timestamp_ms": 1000},
    ],
)
def test_parse_payload_timestamp_rejects(payload):
    with pytest.raises(InvalidTimestampError):
        parse_payload_timestamp(payload)


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


async def _battery(db_session, name: str, current_power: float = 0.0) -> ResourceModel:
    battery = await ResourceRepository.create(
        db_session,
        name=f"{name}-{datetime.now().timestamp()}",
        resource_type="battery",
        rated_power=10.0,
    )
    battery.current_power = current_power
    await db_session.commit()
    return battery


async def _current_power(resource_id: str) -> float:
    async with get_session_factory()() as session:
        resource = await session.get(ResourceModel, resource_id)
        assert resource is not None
        return resource.current_power


def _collect_events() -> list[Event]:
    reset_event_bus()
    received: list[Event] = []

    async def on_event(e: Event) -> None:
        received.append(e)

    get_event_bus().subscribe(on_event, event_types={EventType.RESOURCE_UPDATED})
    return received


def _ingestor() -> MQTTTelemetryIngestor:
    return MQTTTelemetryIngestor(MQTTAdapter(), get_session_factory())


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@pytest.mark.asyncio
async def test_power_updates_live_current_power_and_broadcasts(db_session, app):
    events = _collect_events()
    battery = await _battery(db_session, "mqtt-live")

    wrote = await _ingestor().ingest_message(
        ProtocolMessage(
            topic=f"vpp/s/battery/{battery.id}/state", payload={"soc": 61.0, "power": -3.5}
        )
    )

    assert wrote is True
    assert await _current_power(battery.id) == pytest.approx(-3.5)
    assert len(events) == 1
    assert events[0].source == "mqtt.telemetry_ingestion"
    # Same key as the HTTP endpoint's broadcast, plus the MQTT fields.
    assert events[0].data["current_power_kw"] == pytest.approx(-3.5)
    assert events[0].data["soc"] == pytest.approx(61.0)


@pytest.mark.asyncio
async def test_sample_without_power_leaves_live_power_alone(db_session, app):
    events = _collect_events()
    battery = await _battery(db_session, "mqtt-nopower", current_power=4.0)

    await _ingestor().ingest_message(
        ProtocolMessage(topic=f"vpp/s/battery/{battery.id}/state", payload={"soc": 50.0})
    )

    assert await _current_power(battery.id) == pytest.approx(4.0)
    assert len(events) == 1
    assert "current_power_kw" not in events[0].data


@pytest.mark.asyncio
async def test_payload_timestamp_is_stored(db_session, app):
    battery = await _battery(db_session, "mqtt-ts")
    sample_time = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(minutes=2)

    await _ingestor().ingest_message(
        ProtocolMessage(
            topic=f"vpp/s/battery/{battery.id}/state",
            payload={"soc": 50.0, "power": 1.0, "timestamp": _iso(sample_time)},
            # Received much later than sampled (e.g. a gateway's buffer).
            timestamp=datetime.now(timezone.utc).timestamp(),
        )
    )

    rows = await BatteryStateRepository.get_latest(db_session, battery.id)
    assert len(rows) == 1
    assert rows[0].timestamp.replace(tzinfo=timezone.utc) == sample_time
    # Two minutes old is still live (the HTTP limit is 15 minutes).
    assert await _current_power(battery.id) == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_old_sample_is_history_only(db_session, app):
    """Like an HTTP backfill: stored, but no live update and no broadcast."""
    events = _collect_events()
    battery = await _battery(db_session, "mqtt-old", current_power=2.0)
    old = datetime.now(timezone.utc) - timedelta(minutes=16)

    wrote = await _ingestor().ingest_message(
        ProtocolMessage(
            topic=f"vpp/s/battery/{battery.id}/state",
            payload={"soc": 40.0, "power": 9.0, "timestamp_ms": int(old.timestamp() * 1000)},
        )
    )

    assert wrote is True
    rows = await BatteryStateRepository.get_latest(db_session, battery.id)
    assert len(rows) == 1
    assert rows[0].power == pytest.approx(9.0)
    assert await _current_power(battery.id) == pytest.approx(2.0)
    assert events == []


@pytest.mark.asyncio
async def test_future_sample_is_dropped(db_session, app, caplog):
    events = _collect_events()
    battery = await _battery(db_session, "mqtt-future", current_power=2.0)
    future = datetime.now(timezone.utc) + timedelta(minutes=10)

    with caplog.at_level("WARNING", logger="vpp.protocols.telemetry_ingestion"):
        wrote = await _ingestor().ingest_message(
            ProtocolMessage(
                topic=f"vpp/s/battery/{battery.id}/state",
                payload={"soc": 40.0, "power": 9.0, "timestamp": future.timestamp()},
            )
        )

    assert wrote is False
    assert "in the future" in caplog.text
    assert await BatteryStateRepository.get_latest(db_session, battery.id) == []
    assert await _current_power(battery.id) == pytest.approx(2.0)
    assert events == []


@pytest.mark.asyncio
async def test_small_clock_skew_is_accepted(db_session, app):
    battery = await _battery(db_session, "mqtt-skew")
    ahead = datetime.now(timezone.utc) + timedelta(minutes=4)

    wrote = await _ingestor().ingest_message(
        ProtocolMessage(
            topic=f"vpp/s/battery/{battery.id}/state",
            payload={"soc": 40.0, "power": 1.5, "timestamp": _iso(ahead)},
        )
    )

    assert wrote is True
    assert await _current_power(battery.id) == pytest.approx(1.5)


@pytest.mark.asyncio
async def test_invalid_timestamp_drops_only_that_message(db_session, app, caplog):
    battery = await _battery(db_session, "mqtt-badts")
    topic = f"vpp/s/battery/{battery.id}/state"

    with caplog.at_level("WARNING", logger="vpp.protocols.telemetry_ingestion"):
        written = await _ingestor().ingest_batch(
            [
                ProtocolMessage(topic=topic, payload={"soc": 1.0, "timestamp": "2026-10-10"}),
                ProtocolMessage(topic=topic, payload={"soc": 2.0}),
            ]
        )

    assert written == 1
    assert "not an ISO 8601 date-time" in caplog.text
    rows = await BatteryStateRepository.get_latest(db_session, battery.id)
    assert [r.soc for r in rows] == [2.0]


@pytest.mark.asyncio
async def test_newest_sample_in_a_batch_sets_live_power(db_session, app):
    """Out-of-order arrival: the newest sample wins, not the last message."""
    events = _collect_events()
    a = await _battery(db_session, "mqtt-batch-a")
    b = await _battery(db_session, "mqtt-batch-b", current_power=7.0)
    now = datetime.now(timezone.utc)

    def msg(rid: str, power: float, age_s: float) -> ProtocolMessage:
        stamp = _iso(now - timedelta(seconds=age_s))
        return ProtocolMessage(
            topic=f"vpp/s/battery/{rid}/state",
            payload={"soc": 50.0, "power": power, "timestamp": stamp},
        )

    written = await _ingestor().ingest_batch(
        [
            msg(a.id, 1.0, 30),
            msg(a.id, 3.0, 10),  # newest for a
            msg(a.id, 2.0, 20),  # arrives last, but older
            msg(b.id, 5.0, 3600),  # b: only an old sample -> history only
        ]
    )

    assert written == 4
    assert await _current_power(a.id) == pytest.approx(3.0)
    assert await _current_power(b.id) == pytest.approx(7.0)
    # Live samples are broadcast in arrival order; only the one that set the
    # live value carries current_power_kw. The old sample of b is not broadcast.
    assert [(e.data["resource_id"], e.data["power"]) for e in events] == [
        (a.id, 1.0),
        (a.id, 3.0),
        (a.id, 2.0),
    ]
    assert [e.data.get("current_power_kw") for e in events] == [None, 3.0, None]


@pytest.mark.asyncio
async def test_non_finite_values_are_rejected(db_session, app):
    battery = await _battery(db_session, "mqtt-nan")
    wrote = await _ingestor().ingest_message(
        ProtocolMessage(
            topic=f"vpp/s/battery/{battery.id}/state",
            payload={"soc": 50.0, "power": float("inf")},
        )
    )
    assert wrote is False
    assert await _current_power(battery.id) == 0.0
