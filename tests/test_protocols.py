"""Tests for protocol adapter base, registry, and adapters."""

import asyncio
import pytest

from vpp.protocols.base import (
    ProtocolAdapter,
    ProtocolMessage,
    ProtocolRegistry,
    ProtocolStatus,
)
from vpp.protocols.openadr import OpenADRAdapter, DREvent, DRSignalType
from vpp.protocols.ocpp import OCPPAdapter, ChargePoint, ChargePointStatus


# ---------------------------------------------------------------------------
# In-memory test adapter
# ---------------------------------------------------------------------------

class MemoryAdapter(ProtocolAdapter):
    """Simple in-memory adapter for testing."""

    def __init__(self) -> None:
        super().__init__("memory", "1.0")
        self.sent: list[ProtocolMessage] = []
        self.inbox: list[ProtocolMessage] = []

    async def connect(self) -> None:
        self._status = ProtocolStatus.CONNECTED

    async def disconnect(self) -> None:
        self._status = ProtocolStatus.DISCONNECTED

    async def send(self, message: ProtocolMessage) -> None:
        self.sent.append(message)
        self._metrics.messages_sent += 1

    async def receive(self) -> ProtocolMessage | None:
        return self.inbox.pop(0) if self.inbox else None


# ---------------------------------------------------------------------------
# Base + Registry tests
# ---------------------------------------------------------------------------

class TestProtocolBase:
    @pytest.mark.asyncio
    async def test_adapter_lifecycle(self):
        adapter = MemoryAdapter()
        assert adapter.status == ProtocolStatus.DISCONNECTED
        assert not adapter.is_connected

        await adapter.connect()
        assert adapter.status == ProtocolStatus.CONNECTED
        assert adapter.is_connected

        await adapter.disconnect()
        assert adapter.status == ProtocolStatus.DISCONNECTED

    @pytest.mark.asyncio
    async def test_send_receive(self):
        adapter = MemoryAdapter()
        await adapter.connect()

        msg = ProtocolMessage(topic="test/topic", payload={"value": 42})
        await adapter.send(msg)
        assert len(adapter.sent) == 1
        assert adapter.metrics.messages_sent == 1

    @pytest.mark.asyncio
    async def test_subscribe_dispatch(self):
        adapter = MemoryAdapter()
        received = []

        async def handler(msg: ProtocolMessage) -> None:
            received.append(msg)

        adapter.subscribe("test/topic", handler)
        msg = ProtocolMessage(topic="test/topic", payload={"data": "hello"})
        await adapter._dispatch(msg)

        assert len(received) == 1
        assert received[0].payload["data"] == "hello"
        assert adapter.metrics.messages_received == 1

    @pytest.mark.asyncio
    async def test_wildcard_subscriber(self):
        adapter = MemoryAdapter()
        received = []

        async def handler(msg: ProtocolMessage) -> None:
            received.append(msg)

        adapter.subscribe("*", handler)
        await adapter._dispatch(ProtocolMessage(topic="any/topic", payload={}))
        assert len(received) == 1

    def test_configure(self):
        adapter = MemoryAdapter()
        adapter.configure(host="localhost", port=1883)
        assert adapter._config["host"] == "localhost"
        assert adapter._config["port"] == 1883


class TestProtocolRegistry:
    def test_register_and_list(self):
        registry = ProtocolRegistry()
        a1 = MemoryAdapter()
        registry.register(a1)
        assert len(registry) == 1
        assert "memory" in registry
        assert registry.get("memory") is a1

    def test_duplicate_registration_raises(self):
        registry = ProtocolRegistry()
        registry.register(MemoryAdapter())
        with pytest.raises(ValueError, match="already registered"):
            registry.register(MemoryAdapter())

    def test_unregister(self):
        registry = ProtocolRegistry()
        registry.register(MemoryAdapter())
        registry.unregister("memory")
        assert len(registry) == 0

    @pytest.mark.asyncio
    async def test_connect_disconnect_all(self):
        registry = ProtocolRegistry()
        a1 = MemoryAdapter()
        a1.name = "mem1"
        a2 = MemoryAdapter()
        a2.name = "mem2"
        registry.register(a1)
        registry.register(a2)

        await registry.connect_all()
        assert a1.is_connected
        assert a2.is_connected

        await registry.disconnect_all()
        assert not a1.is_connected
        assert not a2.is_connected


# ---------------------------------------------------------------------------
# OpenADR tests
# ---------------------------------------------------------------------------

class TestOpenADR:
    @pytest.mark.asyncio
    async def test_lifecycle(self):
        adapter = OpenADRAdapter()
        adapter.configure(role="ven", poll_interval_s=0)
        await adapter.connect()
        assert adapter.is_connected
        await adapter.disconnect()
        assert not adapter.is_connected

    @pytest.mark.asyncio
    async def test_event_handling(self):
        adapter = OpenADRAdapter()
        adapter.configure(role="ven", poll_interval_s=0, auto_opt_in=True)
        await adapter.connect()

        event = DREvent(
            signal_type=DRSignalType.LOAD_DISPATCH,
            signal_level=50.0,
            duration_seconds=1800,
        )

        response = await adapter.handle_incoming_event(event)
        assert response.opt_type == "optIn"
        assert event.event_id in adapter._events

        await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_event_handler_callback(self):
        adapter = OpenADRAdapter()
        adapter.configure(role="ven", poll_interval_s=0)
        await adapter.connect()

        from vpp.protocols.openadr import DRResponse

        async def custom_handler(event: DREvent) -> DRResponse:
            return DRResponse(event_id=event.event_id, opt_type="optOut")

        adapter.register_event_handler(custom_handler)

        event = DREvent(signal_type=DRSignalType.SIMPLE, signal_level=2)
        response = await adapter.handle_incoming_event(event)
        assert response.opt_type == "optOut"

        await adapter.disconnect()


# ---------------------------------------------------------------------------
# OCPP tests
# ---------------------------------------------------------------------------

class TestOCPP:
    @pytest.mark.asyncio
    async def test_lifecycle(self):
        adapter = OCPPAdapter()
        await adapter.connect()
        assert adapter.is_connected
        await adapter.disconnect()
        assert not adapter.is_connected

    @pytest.mark.asyncio
    async def test_charge_point_management(self):
        adapter = OCPPAdapter()
        await adapter.connect()

        cp = ChargePoint(
            charge_point_id="CP-001",
            max_power_kw=22.0,
            v2g_capable=True,
        )
        adapter.register_charge_point(cp)

        assert len(adapter.list_charge_points()) == 1
        assert adapter.get_charge_point("CP-001") is cp

        await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_remote_start_stop(self):
        adapter = OCPPAdapter()
        await adapter.connect()

        cp = ChargePoint(charge_point_id="CP-002", status=ChargePointStatus.AVAILABLE)
        adapter.register_charge_point(cp)

        ok = await adapter.remote_start("CP-002")
        assert ok
        assert cp.status == ChargePointStatus.CHARGING
        assert cp.active_transaction_id is not None

        ok = await adapter.remote_stop("CP-002")
        assert ok
        assert cp.status == ChargePointStatus.AVAILABLE
        assert cp.active_transaction_id is None

        await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_remote_start_unavailable(self):
        adapter = OCPPAdapter()
        await adapter.connect()

        cp = ChargePoint(charge_point_id="CP-003", status=ChargePointStatus.FAULTED)
        adapter.register_charge_point(cp)

        ok = await adapter.remote_start("CP-003")
        assert not ok
        assert cp.status == ChargePointStatus.FAULTED

        await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_fleet_queries(self):
        adapter = OCPPAdapter()
        await adapter.connect()

        cp1 = ChargePoint(charge_point_id="CP-A", current_power_kw=10.0, v2g_capable=True, status=ChargePointStatus.CHARGING, max_power_kw=22.0)
        cp2 = ChargePoint(charge_point_id="CP-B", current_power_kw=5.0, v2g_capable=False, status=ChargePointStatus.CHARGING, max_power_kw=11.0)
        adapter.register_charge_point(cp1)
        adapter.register_charge_point(cp2)

        assert adapter.total_charging_power() == 15.0
        assert adapter.available_v2g_capacity() == 22.0

        await adapter.disconnect()
