"""MQTT protocol adapter for IoT telemetry.

Topic structure: ``vpp/{site_id}/{resource_type}/{resource_id}/{metric}``

Supports QoS 0/1/2, TLS, and automatic reconnection.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from vpp.protocols.base import (
    ProtocolAdapter,
    ProtocolMessage,
    ProtocolStatus,
)

logger = logging.getLogger(__name__)


class MQTTAdapter(ProtocolAdapter):
    """Paho-MQTT async wrapper implementing the VPP ``ProtocolAdapter`` ABC.

    Configuration keys (passed via ``configure``):
        broker_host, broker_port, username, password, client_id, use_tls,
        ca_certs, keepalive, topic_prefix
    """

    def __init__(self) -> None:
        super().__init__("mqtt", "1.0")
        self._client: Any | None = None
        self._listen_task: asyncio.Task | None = None
        self._message_queue: asyncio.Queue[ProtocolMessage] = asyncio.Queue(maxsize=1000)

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

        broker = self._config.get("broker_host", "localhost")
        port = int(self._config.get("broker_port", 1883))
        keepalive = int(self._config.get("keepalive", 60))
        client_id = self._config.get("client_id", "vpp-mqtt-adapter")

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
            self._client.tls_set(ca_certs=self._config.get("ca_certs"))

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
            await asyncio.sleep(0.1)
        else:
            self._status = ProtocolStatus.ERROR
            raise ConnectionError(f"MQTT CONNACK timeout connecting to {broker}:{port}")

        self._metrics.connected_since = time.time()
        logger.info("MQTT connected to %s:%d", broker, port)

    async def disconnect(self) -> None:
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
        if not self.is_connected or self._client is None:
            raise ConnectionError("MQTT adapter is not connected")

        payload = json.dumps(message.payload)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: self._client.publish(message.topic, payload, qos=message.qos),
        )
        self._metrics.messages_sent += 1
        self._metrics.last_message_at = time.time()

    async def receive(self) -> ProtocolMessage | None:
        try:
            return self._message_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    # -- MQTT-specific -------------------------------------------------------

    async def subscribe_topic(self, topic: str, qos: int = 1) -> None:
        """Subscribe to an MQTT topic on the broker."""
        if self._client is None:
            raise ConnectionError("Not connected")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: self._client.subscribe(topic, qos))
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

    def _on_connect(self, client: Any, userdata: Any, flags: Any, rc: Any, properties: Any = None) -> None:
        self._status = ProtocolStatus.CONNECTED
        prefix = self._config.get("topic_prefix", "vpp/#")
        client.subscribe(prefix, qos=1)

    def _on_disconnect(self, client: Any, userdata: Any, flags: Any, rc: Any, properties: Any = None) -> None:
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
        try:
            self._message_queue.put_nowait(msg)
        except asyncio.QueueFull:
            self._metrics.errors += 1
            logger.warning("MQTT message queue full, dropping message on %s", mqtt_msg.topic)
