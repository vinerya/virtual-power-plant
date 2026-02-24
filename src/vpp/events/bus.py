"""Enhanced event bus with typed subscriptions, filters, and async dispatch.

Builds on the existing VPP event system with:
- Typed event subscriptions with optional filter functions
- Async publish with concurrent subscriber notification
- WebSocket auto-broadcast integration point
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    # Resources
    RESOURCE_ADDED = "resource_added"
    RESOURCE_REMOVED = "resource_removed"
    RESOURCE_UPDATED = "resource_updated"
    RESOURCE_FAULT = "resource_fault"

    # Optimization
    OPTIMIZATION_STARTED = "optimization_started"
    OPTIMIZATION_COMPLETED = "optimization_completed"
    OPTIMIZATION_FAILED = "optimization_failed"
    DISPATCH_EXECUTED = "dispatch_executed"

    # Trading
    ORDER_SUBMITTED = "order_submitted"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    TRADE_EXECUTED = "trade_executed"
    MARKET_DATA = "market_data"

    # Protocols
    PROTOCOL_CONNECTED = "protocol_connected"
    PROTOCOL_DISCONNECTED = "protocol_disconnected"
    PROTOCOL_ERROR = "protocol_error"
    DR_EVENT_RECEIVED = "dr_event_received"
    DR_RESPONSE_SENT = "dr_response_sent"

    # V2G
    EV_CONNECTED = "ev_connected"
    EV_DISCONNECTED = "ev_disconnected"
    V2G_DISPATCH = "v2g_dispatch"
    V2G_SCHEDULE_CREATED = "v2g_schedule_created"

    # Grid
    ISLAND_DETECTED = "island_detected"
    ISLAND_ENTERED = "island_entered"
    GRID_RECONNECTED = "grid_reconnected"
    LOAD_SHED = "load_shed"

    # System
    ALERT_TRIGGERED = "alert_triggered"
    CONFIG_CHANGED = "config_changed"
    SYSTEM_ERROR = "system_error"


@dataclass
class Event:
    """An event emitted by the VPP platform."""

    event_type: EventType
    data: dict[str, Any] = field(default_factory=dict)
    source: str = ""
    severity: str = "info"  # info, warning, error, critical
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "data": self.data,
            "source": self.source,
            "severity": self.severity,
            "timestamp": self.timestamp,
        }


# Subscriber callback type
EventCallback = Callable[[Event], Awaitable[None]]
EventFilter = Callable[[Event], bool]


@dataclass
class Subscription:
    """A registered event subscription."""

    callback: EventCallback
    event_types: set[EventType] | None = None  # None = all events
    filter_fn: EventFilter | None = None
    subscription_id: str = field(default_factory=lambda: str(uuid.uuid4()))


class EventBus:
    """Async event bus with typed subscriptions and filtering.

    Usage::

        bus = EventBus()
        async def on_alert(event: Event):
            print(f"Alert: {event.data}")

        bus.subscribe(on_alert, event_types={EventType.ALERT_TRIGGERED})
        await bus.publish(Event(event_type=EventType.ALERT_TRIGGERED, data={"msg": "hi"}))
    """

    def __init__(self, max_history: int = 1000) -> None:
        self._subscriptions: list[Subscription] = []
        self._history: list[Event] = []
        self._max_history = max_history
        self._publish_count = 0

    def subscribe(
        self,
        callback: EventCallback,
        event_types: set[EventType] | None = None,
        filter_fn: EventFilter | None = None,
    ) -> str:
        """Subscribe to events.  Returns subscription ID."""
        sub = Subscription(
            callback=callback,
            event_types=event_types,
            filter_fn=filter_fn,
        )
        self._subscriptions.append(sub)
        return sub.subscription_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """Remove a subscription.  Returns True if found."""
        before = len(self._subscriptions)
        self._subscriptions = [
            s for s in self._subscriptions if s.subscription_id != subscription_id
        ]
        return len(self._subscriptions) < before

    async def publish(self, event: Event) -> int:
        """Publish an event to all matching subscribers.

        Returns the number of subscribers that were notified.
        """
        self._publish_count += 1
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        notified = 0
        tasks = []

        for sub in self._subscriptions:
            # Type filter
            if sub.event_types is not None and event.event_type not in sub.event_types:
                continue
            # Custom filter
            if sub.filter_fn is not None and not sub.filter_fn(event):
                continue

            tasks.append(self._safe_call(sub.callback, event))
            notified += 1

        if tasks:
            await asyncio.gather(*tasks)

        return notified

    def get_history(
        self,
        event_type: EventType | None = None,
        limit: int = 50,
        since: float | None = None,
    ) -> list[Event]:
        """Query event history."""
        events = self._history
        if event_type is not None:
            events = [e for e in events if e.event_type == event_type]
        if since is not None:
            events = [e for e in events if e.timestamp >= since]
        return events[-limit:]

    @property
    def subscriber_count(self) -> int:
        return len(self._subscriptions)

    @property
    def publish_count(self) -> int:
        return self._publish_count

    @staticmethod
    async def _safe_call(callback: EventCallback, event: Event) -> None:
        try:
            await callback(event)
        except Exception:
            logger.exception(
                "Event subscriber error for %s", event.event_type.value,
            )
