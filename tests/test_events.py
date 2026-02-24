"""Tests for the enhanced event bus."""

import pytest

from vpp.events.bus import EventBus, Event, EventType


class TestEventBus:
    @pytest.mark.asyncio
    async def test_subscribe_and_publish(self):
        bus = EventBus()
        received = []

        async def handler(event: Event):
            received.append(event)

        bus.subscribe(handler, event_types={EventType.RESOURCE_ADDED})
        count = await bus.publish(Event(event_type=EventType.RESOURCE_ADDED, data={"name": "bat1"}))

        assert count == 1
        assert len(received) == 1
        assert received[0].data["name"] == "bat1"

    @pytest.mark.asyncio
    async def test_type_filtering(self):
        bus = EventBus()
        received = []

        async def handler(event: Event):
            received.append(event)

        bus.subscribe(handler, event_types={EventType.RESOURCE_ADDED})
        await bus.publish(Event(event_type=EventType.ORDER_SUBMITTED))
        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_custom_filter(self):
        bus = EventBus()
        received = []

        async def handler(event: Event):
            received.append(event)

        bus.subscribe(
            handler,
            event_types={EventType.ALERT_TRIGGERED},
            filter_fn=lambda e: e.severity == "critical",
        )

        await bus.publish(Event(event_type=EventType.ALERT_TRIGGERED, severity="info"))
        assert len(received) == 0

        await bus.publish(Event(event_type=EventType.ALERT_TRIGGERED, severity="critical"))
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_wildcard_subscriber(self):
        bus = EventBus()
        received = []

        async def handler(event: Event):
            received.append(event)

        bus.subscribe(handler)  # No type filter = all events
        await bus.publish(Event(event_type=EventType.RESOURCE_ADDED))
        await bus.publish(Event(event_type=EventType.ORDER_FILLED))
        assert len(received) == 2

    @pytest.mark.asyncio
    async def test_unsubscribe(self):
        bus = EventBus()
        received = []

        async def handler(event: Event):
            received.append(event)

        sub_id = bus.subscribe(handler)
        await bus.publish(Event(event_type=EventType.RESOURCE_ADDED))
        assert len(received) == 1

        bus.unsubscribe(sub_id)
        await bus.publish(Event(event_type=EventType.RESOURCE_ADDED))
        assert len(received) == 1  # no new events

    @pytest.mark.asyncio
    async def test_history(self):
        bus = EventBus(max_history=5)
        for i in range(10):
            await bus.publish(Event(event_type=EventType.RESOURCE_UPDATED, data={"i": i}))

        history = bus.get_history()
        assert len(history) == 5  # capped at max_history

    @pytest.mark.asyncio
    async def test_history_by_type(self):
        bus = EventBus()
        await bus.publish(Event(event_type=EventType.RESOURCE_ADDED))
        await bus.publish(Event(event_type=EventType.ORDER_SUBMITTED))
        await bus.publish(Event(event_type=EventType.RESOURCE_ADDED))

        history = bus.get_history(event_type=EventType.RESOURCE_ADDED)
        assert len(history) == 2

    @pytest.mark.asyncio
    async def test_subscriber_error_handled(self):
        bus = EventBus()

        async def bad_handler(event: Event):
            raise RuntimeError("oops")

        async def good_handler(event: Event):
            pass  # should still be called

        bus.subscribe(bad_handler)
        bus.subscribe(good_handler)

        # Should not raise
        count = await bus.publish(Event(event_type=EventType.SYSTEM_ERROR))
        assert count == 2

    def test_event_to_dict(self):
        e = Event(event_type=EventType.RESOURCE_ADDED, data={"x": 1}, source="test")
        d = e.to_dict()
        assert d["event_type"] == "resource_added"
        assert d["data"]["x"] == 1

    @pytest.mark.asyncio
    async def test_publish_count(self):
        bus = EventBus()
        assert bus.publish_count == 0
        await bus.publish(Event(event_type=EventType.RESOURCE_ADDED))
        await bus.publish(Event(event_type=EventType.RESOURCE_ADDED))
        assert bus.publish_count == 2
