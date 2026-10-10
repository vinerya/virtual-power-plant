"""MQTT protocol adapter for IoT telemetry.

Topic structure: ``vpp/{site_id}/{resource_type}/{resource_id}/{metric}``

Supports QoS 0/1/2, TLS (server verification and client certificates), and
automatic reconnection.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import secrets
import socket
import time
from typing import TYPE_CHECKING, Any

from vpp.protocols.base import (
    ProtocolAdapter,
    ProtocolMessage,
    ProtocolStatus,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)

#: Minimum seconds between "queue full" warnings.
DROP_LOG_INTERVAL_S = 10.0

#: Random part of :func:`default_client_id`, drawn once per process.
_PROCESS_TOKEN = secrets.token_hex(3)


def default_client_id(prefix: str = "vpp") -> str:
    """A client id unique to this process and stable for its lifetime.

    ``{prefix}-{hostname}-{pid}-{random}``: the broker disconnects an
    existing session when another client connects with the same id, so a
    fixed id made two VPP installations (or two API processes during a
    lease hand-over) on one broker kick each other off in a loop. The id is
    the same for every reconnect of this process; the ingestion client uses
    a clean session, so nothing depends on it surviving a restart. The
    random part keeps containers that share a hostname pattern and pid 1
    apart. Set ``VPP_MQTT_CLIENT_ID`` for a fixed id (e.g. for broker ACLs).
    """
    host = re.sub(r"[^A-Za-z0-9-]", "-", socket.gethostname())[:20].strip("-") or "host"
    return f"{prefix}-{host}-{os.getpid()}-{_PROCESS_TOKEN}"


class MQTTAdapter(ProtocolAdapter):
    """Paho-MQTT async wrapper implementing the VPP ``ProtocolAdapter`` ABC.

    Configuration keys (passed via ``configure``):
        broker_host, broker_port, username, password, client_id (default:
        :func:`default_client_id`), keepalive, topic_prefix, and for TLS:
        use_tls, ca_certs (CA bundle; default: the system trust store),
        certfile/keyfile (client certificate), tls_insecure (skip
        certificate and hostname verification -- testing only)
    """

    def __init__(self) -> None:
        super().__init__("mqtt", "1.0")
        self._client: Any | None = None
        self._listen_task: asyncio.Task | None = None
        self._message_queue: asyncio.Queue[ProtocolMessage] = asyncio.Queue(maxsize=1000)
        # Paho calls back from its own network thread; messages are handed to
        # the queue on this loop (asyncio.Queue is not thread-safe).
        self._loop: asyncio.AbstractEventLoop | None = None
        self.dropped = 0  # messages dropped because the queue was full
        self._drop_logged_at = 0.0
        self._drops_since_log = 0
        self._refused: str | None = None  # CONNACK failure reason, if refused

    # -- Lifecycle -----------------------------------------------------------

    async def connect(self) -> None:
        try:
            import paho.mqtt.client as mqtt
        except ImportError as exc:
            raise RuntimeError(
                "paho-mqtt is required for MQTT support. "
                "Install with: pip install 'virtual-power-plant[protocols]'"
            ) from exc

        self._status = ProtocolStatus.CONNECTING
        self._refused = None
        self._loop = asyncio.get_running_loop()

        broker = self._config.get("broker_host", "localhost")
        port = int(self._config.get("broker_port", 1883))
        keepalive = int(self._config.get("keepalive", 60))
        client_id = self._config.get("client_id") or default_client_id()

        self._client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=client_id,
        )

        # Auth
        username = self._config.get("username")
        password = self._config.get("password")
        if username:
            self._client.username_pw_set(username, password)

        # TLS
        if self._config.get("use_tls"):
            self._configure_tls(self._client)

        # Callbacks
        self._client.on_message = self._on_message
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._client.connect, broker, port, keepalive)
        self._client.loop_start()

        # Wait for CONNACK
        for _ in range(50):
            if self._status == ProtocolStatus.CONNECTED:
                break
            if self._refused is not None:
                break
            await asyncio.sleep(0.1)
        else:
            self._status = ProtocolStatus.ERROR
            raise ConnectionError(f"MQTT CONNACK timeout connecting to {broker}:{port}")
        if self._refused is not None:
            # Stop paho retrying with credentials the broker just rejected;
            # the caller (the ingestion loop) retries on its own schedule.
            self._client.loop_stop()
            self._client = None
            self._status = ProtocolStatus.ERROR
            raise ConnectionError(
                f"MQTT broker {broker}:{port} refused the connection: {self._refused}"
            )

        self._metrics.connected_since = time.time()
        logger.info(
            "MQTT connected to %s:%d as %r%s",
            broker,
            port,
            client_id,
            " over TLS" if self._config.get("use_tls") else "",
        )

    def _configure_tls(self, client: Any) -> None:
        import ssl

        insecure = bool(self._config.get("tls_insecure"))
        client.tls_set(
            ca_certs=self._config.get("ca_certs") or None,
            certfile=self._config.get("certfile") or None,
            keyfile=self._config.get("keyfile") or None,
            cert_reqs=ssl.CERT_NONE if insecure else ssl.CERT_REQUIRED,
        )
        if insecure:
            client.tls_insecure_set(True)
            logger.warning(
                "MQTT TLS certificate and hostname verification are DISABLED "
                "(VPP_MQTT_TLS_INSECURE); use this for testing only"
            )

    async def disconnect(self) -> None:
        # Set first: paho reports our own disconnect through _on_disconnect.
        self._status = ProtocolStatus.DISCONNECTED
        if self._client is not None:
            self._client.loop_stop()
            self._client.disconnect()
            self._client = None
        if self._listen_task and not self._listen_task.done():
            self._listen_task.cancel()
        self._status = ProtocolStatus.DISCONNECTED
        self._metrics.connected_since = None
        logger.info("MQTT disconnected")

    async def send(self, message: ProtocolMessage) -> None:
        client = self._client
        if not self.is_connected or client is None:
            raise ConnectionError("MQTT adapter is not connected")

        payload = json.dumps(message.payload)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: client.publish(message.topic, payload, qos=message.qos),
        )
        self._metrics.messages_sent += 1
        self._metrics.last_message_at = time.time()

    async def receive(self) -> ProtocolMessage | None:
        try:
            return self._message_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def receive_forever(self) -> AsyncIterator[ProtocolMessage]:
        """Yield inbound messages as they arrive.

        Unlike :meth:`receive` (single message, non-blocking), this blocks
        on the internal queue between messages -- the right primitive for a
        long-running ingestion loop, which would otherwise have to busy-poll
        ``receive()``.
        """
        while True:
            yield await self._message_queue.get()

    def receive_nowait(self, limit: int) -> list[ProtocolMessage]:
        """Up to ``limit`` messages that are already queued (never waits)."""
        out: list[ProtocolMessage] = []
        while len(out) < limit:
            try:
                out.append(self._message_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return out

    # -- MQTT-specific -------------------------------------------------------

    async def subscribe_topic(self, topic: str, qos: int = 1) -> None:
        """Subscribe to an MQTT topic on the broker."""
        client = self._client
        if client is None:
            raise ConnectionError("Not connected")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: client.subscribe(topic, qos))
        logger.info("MQTT subscribed to %s (QoS %d)", topic, qos)

    async def publish(
        self,
        topic: str,
        payload: dict[str, Any],
        qos: int = 1,
        retain: bool = False,
    ) -> None:
        """Convenience wrapper around ``send``."""
        msg = ProtocolMessage(topic=topic, payload=payload, qos=qos, source="mqtt")
        await self.send(msg)

    # -- Paho callbacks (run in paho thread) ---------------------------------

    def _on_connect(
        self, client: Any, userdata: Any, flags: Any, rc: Any, properties: Any = None
    ) -> None:
        if getattr(rc, "is_failure", False) or (isinstance(rc, int) and rc != 0):
            # e.g. "Not authorized" / "Bad user name or password": this is
            # not a connection, so don't report one (or subscribe).
            self._refused = str(rc)
            self._status = ProtocolStatus.ERROR
            logger.warning("MQTT broker refused the connection: %s", rc)
            return
        self._refused = None
        self._status = ProtocolStatus.CONNECTED
        prefix = self._config.get("topic_prefix", "vpp/#")
        client.subscribe(prefix, qos=1)

    def _on_disconnect(
        self, client: Any, userdata: Any, flags: Any, rc: Any, properties: Any = None
    ) -> None:
        if self._status != ProtocolStatus.DISCONNECTED:
            self._status = ProtocolStatus.RECONNECTING
            logger.warning("MQTT disconnected unexpectedly (rc=%s)", rc)

    def _on_message(self, client: Any, userdata: Any, mqtt_msg: Any) -> None:
        try:
            payload = json.loads(mqtt_msg.payload.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload = {"raw": mqtt_msg.payload.decode("utf-8", errors="replace")}

        msg = ProtocolMessage(
            topic=mqtt_msg.topic,
            payload=payload,
            source="mqtt",
            qos=mqtt_msg.qos,
        )
        loop = self._loop
        if loop is None or loop.is_closed():
            self._enqueue(msg)  # no loop to hand over to (tests calling this directly)
            return
        # put_nowait from this (paho) thread would not wake the event loop:
        # the message would sit in the queue until something else did.
        # RuntimeError: the loop closed meanwhile (shutdown).
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(self._enqueue, msg)

    def _enqueue(self, msg: ProtocolMessage) -> None:
        # Every message the broker delivered, including any dropped below,
        # so GET /api/v1/protocols shows whether telemetry is arriving.
        self._metrics.messages_received += 1
        self._metrics.last_message_at = time.time()
        try:
            self._message_queue.put_nowait(msg)
        except asyncio.QueueFull:
            self._metrics.errors += 1
            self.dropped += 1
            self._drops_since_log += 1
            from vpp.metrics import record_protocol_error

            record_protocol_error("mqtt")
            # At most one line per interval: under sustained overload a line
            # per message would flood the log (and cost the CPU it lacks).
            now = time.monotonic()
            if now - self._drop_logged_at >= DROP_LOG_INTERVAL_S:
                logger.warning(
                    "MQTT message queue full (ingestion not keeping up); dropped %d "
                    "message(s) since the last report, e.g. on %s",
                    self._drops_since_log,
                    msg.topic,
                )
                self._drop_logged_at = now
                self._drops_since_log = 0
