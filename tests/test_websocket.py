"""Tests for the WebSocket endpoint."""

import pytest

from vpp.api.websocket import ConnectionManager


@pytest.mark.asyncio
async def test_manager_connection_tracking():
    mgr = ConnectionManager()
    assert mgr.active_count == 0
