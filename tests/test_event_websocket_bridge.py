"""EventBus publishes must reach connected WebSocket clients.

Before this fix, EventBus (vpp/events/bus.py) and the WebSocket
ConnectionManager (vpp/api/websocket.py) were two disconnected pub/sub
systems: events were published but nothing ever forwarded them to
connected clients.
"""

from __future__ import annotations

import json

import pytest

from vpp.api import app as app_module
from vpp.api.websocket import ConnectionManager, manager, subscribe_event_bus_to_websocket
from vpp.db import engine as db_engine
from vpp.events import Event, EventType, get_event_bus, reset_event_bus
from vpp.events.bus import EventBus
from vpp.settings import Settings


class _FakeWebSocket:
    """Minimal stand-in for a starlette WebSocket."""

    def __init__(self) -> None:
        self.sent: list[str] = []

    async def accept(self) -> None:
        pass

    async def send_text(self, data: str) -> None:
        self.sent.append(data)


@pytest.mark.asyncio
async def test_published_event_reaches_subscribed_websocket():
    bus = EventBus()
    mgr = ConnectionManager()
    ws = _FakeWebSocket()
    await mgr.connect(ws)
    await mgr.subscribe(ws, "resource_updates")

    subscribe_event_bus_to_websocket(bus, mgr)
    await bus.publish(Event(event_type=EventType.RESOURCE_UPDATED, data={"resource_id": "batt-1"}))

    assert len(ws.sent) == 1
    payload = json.loads(ws.sent[0])
    assert payload["channel"] == "resource_updates"
    assert payload["data"]["data"] == {"resource_id": "batt-1"}


@pytest.mark.asyncio
async def test_event_only_reaches_clients_on_its_mapped_channel():
    bus = EventBus()
    mgr = ConnectionManager()
    ws = _FakeWebSocket()
    await mgr.connect(ws)
    await mgr.subscribe(ws, "resource_updates")

    subscribe_event_bus_to_websocket(bus, mgr)
    # A trade event maps to "market_data", not "resource_updates".
    await bus.publish(Event(event_type=EventType.TRADE_EXECUTED, data={"trade_id": "t-1"}))

    assert ws.sent == []


@pytest.mark.asyncio
async def test_wildcard_subscriber_receives_every_event():
    bus = EventBus()
    mgr = ConnectionManager()
    ws = _FakeWebSocket()
    await mgr.connect(ws)
    await mgr.subscribe(ws, "*")

    subscribe_event_bus_to_websocket(bus, mgr)
    await bus.publish(Event(event_type=EventType.TARIFF_UPDATED, data={"tariff_id": "t-1"}))

    assert len(ws.sent) == 1


@pytest.fixture
def preserve_db_globals():
    """See tests/test_app_lifespan.py — avoids tearing down the shared
    session-scoped test database when a real app lifespan runs."""
    saved_engine = db_engine._engine
    saved_factory = db_engine._session_factory
    yield
    db_engine._engine = saved_engine
    db_engine._session_factory = saved_factory


@pytest.mark.asyncio
async def test_real_app_lifespan_wires_event_bus_to_websocket(monkeypatch, preserve_db_globals):
    """The running app (not just the standalone helper) must forward
    EventBus publishes to connected WebSocket clients."""
    reset_event_bus()

    test_settings = Settings(
        database_url="sqlite+aiosqlite:///./test_lifespan_events.db",
        degradation_updater_enabled=False,
    )
    monkeypatch.setattr(app_module, "get_settings", lambda: test_settings)

    fastapi_app = app_module.create_app(rate_limit_enabled=False)
    ws = _FakeWebSocket()

    async with fastapi_app.router.lifespan_context(fastapi_app):
        await manager.connect(ws)
        await manager.subscribe(ws, "resource_updates")
        try:
            await get_event_bus().publish(
                Event(event_type=EventType.RESOURCE_UPDATED, data={"resource_id": "batt-2"})
            )
            assert len(ws.sent) == 1
            payload = json.loads(ws.sent[0])
            assert payload["channel"] == "resource_updates"
        finally:
            await manager.disconnect(ws)
