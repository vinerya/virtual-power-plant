"""OpenADR 2.0b adapter — Virtual Top Node (VTN) and Virtual End Node (VEN).

Implements demand-response event handling for grid operators: the VTN role
publishes DR signals and the VEN role receives them and schedules responses
through the VPP optimisation engine.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable

from vpp.protocols.base import (
    ProtocolAdapter,
    ProtocolMessage,
    ProtocolStatus,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OpenADR value objects
# ---------------------------------------------------------------------------

class DREventStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class DRSignalType(str, Enum):
    SIMPLE = "SIMPLE"               # 0/1/2/3 levels
    ELECTRICITY_PRICE = "ELECTRICITY_PRICE"
    LOAD_DISPATCH = "LOAD_DISPATCH"  # absolute kW target
    LOAD_CONTROL = "LOAD_CONTROL"    # delta kW
    LOAD_PERCENTAGE = "LOAD_PERCENTAGE"


@dataclass
class DREvent:
    """A demand-response event from or for the grid operator."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    signal_type: DRSignalType = DRSignalType.SIMPLE
    signal_level: float = 0.0
    start_time: float = 0.0
    duration_seconds: int = 3600
    status: DREventStatus = DREventStatus.PENDING
    market_context: str = "default"
    resource_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def end_time(self) -> float:
        return self.start_time + self.duration_seconds

    @property
    def is_active(self) -> bool:
        now = time.time()
        return self.start_time <= now < self.end_time and self.status == DREventStatus.ACTIVE

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "signal_type": self.signal_type.value,
            "signal_level": self.signal_level,
            "start_time": self.start_time,
            "duration_seconds": self.duration_seconds,
            "status": self.status.value,
            "market_context": self.market_context,
            "resource_ids": self.resource_ids,
        }


@dataclass
class DRResponse:
    """VEN response to a DR event."""

    event_id: str
    opt_type: str = "optIn"  # optIn | optOut
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "opt_type": self.opt_type,
            "created_at": self.created_at,
        }


DREventHandler = Callable[[DREvent], Awaitable[DRResponse]]


class OpenADRAdapter(ProtocolAdapter):
    """OpenADR 2.0b adapter (VTN + VEN roles).

    Configuration keys:
        role ("vtn" | "ven"), vtn_url, ven_id, poll_interval_s,
        market_context, auto_opt_in
    """

    def __init__(self) -> None:
        super().__init__("openadr", "2.0b")
        self._events: dict[str, DREvent] = {}
        self._responses: dict[str, DRResponse] = {}
        self._event_handlers: list[DREventHandler] = []
        self._poll_task: asyncio.Task | None = None
        self._message_queue: asyncio.Queue[ProtocolMessage] = asyncio.Queue(maxsize=500)

    # -- Lifecycle -----------------------------------------------------------

    async def connect(self) -> None:
        self._status = ProtocolStatus.CONNECTING

        role = self._config.get("role", "ven")
        if role not in ("vtn", "ven"):
            raise ValueError(f"Invalid OpenADR role: {role}")

        # In a full implementation we would start an HTTP server (VTN) or
        # begin polling a VTN endpoint (VEN).  Here we set up the data
        # structures and mark as connected.
        self._status = ProtocolStatus.CONNECTED
        self._metrics.connected_since = time.time()

        poll_interval = float(self._config.get("poll_interval_s", 30))
        if role == "ven" and poll_interval > 0:
            self._poll_task = asyncio.create_task(self._ven_poll_loop(poll_interval))

        logger.info("OpenADR adapter connected (role=%s)", role)

    async def disconnect(self) -> None:
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        self._status = ProtocolStatus.DISCONNECTED
        self._metrics.connected_since = None
        logger.info("OpenADR adapter disconnected")

    async def send(self, message: ProtocolMessage) -> None:
        """Send a DR event (VTN) or DR response (VEN)."""
        if not self.is_connected:
            raise ConnectionError("OpenADR not connected")

        if message.topic.startswith("openadr/event"):
            event = DREvent(**message.payload)
            self._events[event.event_id] = event
            logger.info("Published DR event %s (signal=%s)", event.event_id, event.signal_type.value)
        elif message.topic.startswith("openadr/response"):
            response = DRResponse(**message.payload)
            self._responses[response.event_id] = response
            logger.info("Sent DR response for event %s: %s", response.event_id, response.opt_type)

        self._metrics.messages_sent += 1
        self._metrics.last_message_at = time.time()

    async def receive(self) -> ProtocolMessage | None:
        try:
            return self._message_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    # -- DR event management -------------------------------------------------

    def register_event_handler(self, handler: DREventHandler) -> None:
        """Register an async callback for incoming DR events."""
        self._event_handlers.append(handler)

    async def publish_event(self, event: DREvent) -> None:
        """Publish a DR event (VTN role)."""
        self._events[event.event_id] = event
        msg = ProtocolMessage(
            topic=f"openadr/event/{event.event_id}",
            payload=event.to_dict(),
            source="openadr",
        )
        await self._dispatch(msg)
        self._metrics.messages_sent += 1

    async def handle_incoming_event(self, event: DREvent) -> DRResponse:
        """Process a received DR event (VEN role).

        Runs registered handlers; defaults to auto opt-in if configured.
        """
        self._events[event.event_id] = event

        # Notify subscribers
        msg = ProtocolMessage(
            topic=f"openadr/event/{event.event_id}",
            payload=event.to_dict(),
            source="openadr",
        )
        await self._dispatch(msg)
        try:
            self._message_queue.put_nowait(msg)
        except asyncio.QueueFull:
            self._metrics.errors += 1

        # Run handlers
        for handler in self._event_handlers:
            try:
                response = await handler(event)
                self._responses[event.event_id] = response
                return response
            except Exception:
                logger.exception("DR event handler failed for %s", event.event_id)
                self._metrics.errors += 1

        # Default: auto opt-in
        auto = self._config.get("auto_opt_in", True)
        response = DRResponse(
            event_id=event.event_id,
            opt_type="optIn" if auto else "optOut",
        )
        self._responses[event.event_id] = response
        return response

    def get_active_events(self) -> list[DREvent]:
        """Return currently active DR events."""
        return [e for e in self._events.values() if e.is_active]

    def get_event(self, event_id: str) -> DREvent | None:
        return self._events.get(event_id)

    # -- VEN polling ---------------------------------------------------------

    async def _ven_poll_loop(self, interval: float) -> None:
        """VEN: periodically request events from VTN (stub)."""
        while self.is_connected:
            try:
                # In production this would make an HTTP request to the VTN.
                # Here we just check for active events and update statuses.
                now = time.time()
                for event in self._events.values():
                    if event.status == DREventStatus.PENDING and event.start_time <= now:
                        event.status = DREventStatus.ACTIVE
                    elif event.status == DREventStatus.ACTIVE and now >= event.end_time:
                        event.status = DREventStatus.COMPLETED
            except Exception:
                logger.exception("OpenADR VEN poll error")
                self._metrics.errors += 1
            await asyncio.sleep(interval)
