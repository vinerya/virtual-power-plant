"""WebSocket endpoint for real-time event streaming."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages active WebSocket connections and channel subscriptions."""

    def __init__(self) -> None:
        self._connections: dict[WebSocket, set[str]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._connections[ws] = set()

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            self._connections.pop(ws, None)

    async def subscribe(self, ws: WebSocket, channel: str) -> None:
        async with self._lock:
            if ws in self._connections:
                self._connections[ws].add(channel)

    async def unsubscribe(self, ws: WebSocket, channel: str) -> None:
        async with self._lock:
            if ws in self._connections:
                self._connections[ws].discard(channel)

    async def broadcast(self, channel: str, data: dict[str, Any]) -> None:
        """Send a message to all subscribers of *channel*."""
        message = json.dumps({
            "channel": channel,
            "data": data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        async with self._lock:
            targets = [
                ws for ws, channels in self._connections.items()
                if channel in channels or "*" in channels
            ]
        for ws in targets:
            try:
                await ws.send_text(message)
            except Exception:
                await self.disconnect(ws)

    @property
    def active_count(self) -> int:
        return len(self._connections)


# Singleton
manager = ConnectionManager()

VALID_CHANNELS = {
    "resource_updates",
    "optimization_events",
    "market_data",
    "alerts",
    "*",
}


async def websocket_endpoint(ws: WebSocket) -> None:
    """Handle a WebSocket connection.

    Protocol (JSON messages):
        Client → Server:
            {"action": "subscribe", "channel": "resource_updates"}
            {"action": "unsubscribe", "channel": "resource_updates"}
        Server → Client:
            {"channel": "...", "data": {...}, "timestamp": "..."}
    """
    await manager.connect(ws)
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_text(json.dumps({"error": "Invalid JSON"}))
                continue

            action = msg.get("action")
            channel = msg.get("channel", "")

            if action == "subscribe" and channel in VALID_CHANNELS:
                await manager.subscribe(ws, channel)
                await ws.send_text(json.dumps({"ack": f"subscribed:{channel}"}))
            elif action == "unsubscribe" and channel in VALID_CHANNELS:
                await manager.unsubscribe(ws, channel)
                await ws.send_text(json.dumps({"ack": f"unsubscribed:{channel}"}))
            else:
                await ws.send_text(json.dumps({"error": f"Unknown action or channel: {action}/{channel}"}))
    except WebSocketDisconnect:
        pass
    finally:
        await manager.disconnect(ws)
