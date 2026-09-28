"""Protocol adapter abstract base and registry.

Follows the same plugin pattern as ``OptimizationPlugin`` — every protocol
adapter implements a small ABC surface and is discovered at runtime through
``ProtocolRegistry``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------

class ProtocolStatus(str, Enum):
    """Lifecycle status of a protocol adapter.

    ``CONNECTED`` means the adapter holds a *real* link to an external
    endpoint (broker, device, VTN, utility server) or, for server-side
    adapters, is actually accepting connections from real peers.

    ``SIMULATED`` means the adapter is running without any configured real
    endpoint: its in-memory state machine works (handy for demos, tests and
    offline development) but nothing leaves the process. Adapters must never
    report ``CONNECTED`` in that situation.
    """

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    SIMULATED = "simulated"
    ERROR = "error"
    RECONNECTING = "reconnecting"


class ProtocolMode(str, Enum):
    """Whether an adapter talks to a real endpoint or simulates one."""

    LIVE = "live"
    SIMULATED = "simulated"


@dataclass
class ProtocolMessage:
    """A message received or sent over a protocol adapter."""

    topic: str
    payload: dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    source: str = ""
    message_id: str = ""
    qos: int = 0

    def __repr__(self) -> str:
        return f"ProtocolMessage(topic={self.topic!r}, source={self.source!r})"


@dataclass
class ProtocolMetrics:
    """Runtime metrics for a protocol adapter."""

    messages_sent: int = 0
    messages_received: int = 0
    errors: int = 0
    last_message_at: float | None = None
    connected_since: float | None = None
    reconnect_count: int = 0

    @property
    def uptime_seconds(self) -> float:
        if self.connected_since is None:
            return 0.0
        return time.time() - self.connected_since


# Callback type: async fn(msg) -> None
MessageCallback = Callable[[ProtocolMessage], Awaitable[None]]


# ---------------------------------------------------------------------------
# Abstract adapter
# ---------------------------------------------------------------------------

class ProtocolAdapter(ABC):
    """Abstract base class for all protocol adapters.

    Each adapter manages its own connection lifecycle and message handling.
    Subclasses must implement ``connect``, ``disconnect``, ``send``, and
    ``receive``.
    """

    def __init__(self, name: str, version: str = "1.0") -> None:
        self.name = name
        self.version = version
        self._status = ProtocolStatus.DISCONNECTED
        self._metrics = ProtocolMetrics()
        self._subscribers: dict[str, list[MessageCallback]] = {}
        self._config: dict[str, Any] = {}

    # -- Properties ----------------------------------------------------------

    @property
    def status(self) -> ProtocolStatus:
        return self._status

    @property
    def metrics(self) -> ProtocolMetrics:
        return self._metrics

    @property
    def is_connected(self) -> bool:
        """True only when linked to a real external endpoint."""
        return self._status == ProtocolStatus.CONNECTED

    @property
    def is_simulated(self) -> bool:
        """True when running in simulated mode (no real endpoint)."""
        return self._status == ProtocolStatus.SIMULATED

    @property
    def is_operational(self) -> bool:
        """True when the adapter accepts operations (live *or* simulated)."""
        return self._status in (ProtocolStatus.CONNECTED, ProtocolStatus.SIMULATED)

    @property
    def mode(self) -> ProtocolMode:
        """Live vs simulated mode.

        Subclasses that can run against a real endpoint override this to
        reflect their configuration; the default derives it from status.
        """
        return ProtocolMode.SIMULATED if self.is_simulated else ProtocolMode.LIVE

    def _enter_simulated_mode(self, reason: str) -> None:
        """Mark the adapter as operational-but-simulated and log why."""
        self._status = ProtocolStatus.SIMULATED
        self._metrics.connected_since = time.time()
        logger.warning(
            "%s adapter running in SIMULATED mode (%s); no traffic leaves this process",
            self.name,
            reason,
        )

    # -- Configuration -------------------------------------------------------

    def configure(self, **kwargs: Any) -> None:
        """Store adapter-specific configuration."""
        self._config.update(kwargs)

    # -- Lifecycle -----------------------------------------------------------

    @abstractmethod
    async def connect(self) -> None:
        """Establish a connection to the protocol endpoint."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Close the connection gracefully."""

    @abstractmethod
    async def send(self, message: ProtocolMessage) -> None:
        """Send a message over the protocol."""

    @abstractmethod
    async def receive(self) -> ProtocolMessage | None:
        """Receive a single message (non-blocking, may return *None*)."""

    # -- Pub / Sub helpers ---------------------------------------------------

    def subscribe(self, topic: str, callback: MessageCallback) -> None:
        """Register a callback for messages on *topic*."""
        self._subscribers.setdefault(topic, []).append(callback)

    def unsubscribe(self, topic: str, callback: MessageCallback) -> None:
        cbs = self._subscribers.get(topic, [])
        if callback in cbs:
            cbs.remove(callback)

    async def _dispatch(self, message: ProtocolMessage) -> None:
        """Dispatch an inbound message to matching subscribers."""
        self._metrics.messages_received += 1
        self._metrics.last_message_at = time.time()

        # Exact-match + wildcard subscribers
        for topic_pattern in (message.topic, "*"):
            for cb in self._subscribers.get(topic_pattern, []):
                try:
                    await cb(message)
                except Exception:
                    logger.exception(
                        "Subscriber error on topic=%s adapter=%s",
                        message.topic,
                        self.name,
                    )
                    self._metrics.errors += 1

    # -- Reconnection helper -------------------------------------------------

    async def _reconnect_loop(
        self,
        max_retries: int = 10,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
    ) -> None:
        """Exponential-backoff reconnect loop (utility for subclasses)."""
        self._status = ProtocolStatus.RECONNECTING
        for attempt in range(1, max_retries + 1):
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            logger.info(
                "Reconnecting %s (attempt %d/%d, delay %.1fs)",
                self.name,
                attempt,
                max_retries,
                delay,
            )
            await asyncio.sleep(delay)
            try:
                await self.connect()
                self._metrics.reconnect_count += 1
                return
            except Exception:
                logger.warning("Reconnect attempt %d failed for %s", attempt, self.name)
        self._status = ProtocolStatus.ERROR
        logger.error("All reconnect attempts exhausted for %s", self.name)

    # -- Repr ----------------------------------------------------------------

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r} status={self._status.value}>"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ProtocolRegistry:
    """Discover and manage protocol adapter instances."""

    def __init__(self) -> None:
        self._adapters: dict[str, ProtocolAdapter] = {}

    def register(self, adapter: ProtocolAdapter) -> None:
        if adapter.name in self._adapters:
            raise ValueError(f"Adapter '{adapter.name}' already registered")
        self._adapters[adapter.name] = adapter
        logger.info("Registered protocol adapter: %s v%s", adapter.name, adapter.version)

    def unregister(self, name: str) -> None:
        self._adapters.pop(name, None)

    def get(self, name: str) -> ProtocolAdapter | None:
        return self._adapters.get(name)

    def list_adapters(self) -> list[ProtocolAdapter]:
        return list(self._adapters.values())

    async def connect_all(self) -> None:
        """Connect every registered adapter."""
        for adapter in self._adapters.values():
            try:
                await adapter.connect()
            except Exception:
                logger.exception("Failed to connect adapter %s", adapter.name)

    async def disconnect_all(self) -> None:
        """Disconnect every registered adapter."""
        for adapter in self._adapters.values():
            try:
                await adapter.disconnect()
            except Exception:
                logger.exception("Failed to disconnect adapter %s", adapter.name)

    def __len__(self) -> int:
        return len(self._adapters)

    def __contains__(self, name: str) -> bool:
        return name in self._adapters
