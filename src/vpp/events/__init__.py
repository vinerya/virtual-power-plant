"""Enhanced event bus with typed subscriptions and async dispatch."""

from vpp.events.bus import EventBus, Event, EventType

_global_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    """Return the process-global :class:`EventBus`, creating it on demand."""
    global _global_bus
    if _global_bus is None:
        _global_bus = EventBus()
    return _global_bus


def reset_event_bus() -> None:
    """Replace the process-global bus with a fresh instance (for tests)."""
    global _global_bus
    _global_bus = EventBus()


__all__ = ["EventBus", "Event", "EventType", "get_event_bus", "reset_event_bus"]
