"""Settings-driven OCPP / OpenADR / IEEE 2030.5 startup, and API status honesty."""

from __future__ import annotations

import asyncio

import pytest

from vpp.api import app as app_module
from vpp.db import engine as db_engine
from vpp.protocols.base import ProtocolAdapter, ProtocolMessage, ProtocolRegistry, ProtocolStatus
from vpp.protocols.bootstrap import (
    build_ieee2030_5_adapter,
    build_openadr_adapter,
    start_protocol_adapters,
    stop_protocol_adapters,
    supervise_adapter,
)
from vpp.protocols.ocpp import OCPPAdapter
from vpp.protocols.openadr import OpenADRAdapter
from vpp.settings import Settings


@pytest.fixture
def preserve_db_globals():
    saved_engine = db_engine._engine
    saved_factory = db_engine._session_factory
    yield
    db_engine._engine = saved_engine
    db_engine._session_factory = saved_factory


@pytest.fixture
def isolated_protocol_registry(monkeypatch):
    from vpp.api.routes import protocols as protocols_module

    registry = ProtocolRegistry()
    monkeypatch.setattr(protocols_module, "_registry", registry)
    return registry


def test_protocol_settings_disabled_by_default():
    s = Settings()
    assert s.ocpp_enabled is False
    assert s.openadr_enabled is False
    assert s.ieee2030_5_enabled is False


def test_builders_map_settings():
    s = Settings(
        openadr_vtn_url="https://vtn/OpenADR2/Simple/2.0b",
        openadr_poll_interval_s=7,
        openadr_cert_path="/c.pem",
        ieee2030_5_server_url="https://s",
        ieee2030_5_lfdi="ABC",
    )
    oadr = build_openadr_adapter(s)
    assert oadr._config["vtn_url"] == "https://vtn/OpenADR2/Simple/2.0b"
    assert oadr._config["poll_interval_s"] == 7
    assert oadr._config["cert_path"] == "/c.pem"
    sep = build_ieee2030_5_adapter(s)
    assert sep._config["server_url"] == "https://s"
    assert "poll_interval_s" not in sep._config


@pytest.mark.asyncio
async def test_supervisor_retries_connect_with_backoff(monkeypatch):
    import vpp.protocols.bootstrap as bootstrap

    class Flaky(ProtocolAdapter):
        def __init__(self):
            super().__init__("flaky")
            self.attempts = 0

        async def connect(self):
            self.attempts += 1
            if self.attempts < 3:
                raise ConnectionError("down")
            self._status = ProtocolStatus.CONNECTED

        async def disconnect(self):
            self._status = ProtocolStatus.DISCONNECTED

        async def send(self, message: ProtocolMessage) -> None:
            pass

        async def receive(self):
            return None

    sleeps: list[float] = []
    real_sleep = asyncio.sleep

    async def fake_sleep(delay):
        sleeps.append(delay)
        await real_sleep(0)

    monkeypatch.setattr(bootstrap.asyncio, "sleep", fake_sleep)
    registry = ProtocolRegistry()
    adapter = Flaky()
    task = asyncio.create_task(supervise_adapter(adapter, registry, base_delay_s=2))
    for _ in range(100):
        if adapter.is_connected:
            break
        await real_sleep(0)
    assert adapter.is_connected
    assert sleeps == [2.0, 4.0]
    assert registry.get("flaky") is adapter
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert "flaky" not in registry
    assert adapter.status == ProtocolStatus.DISCONNECTED


@pytest.mark.asyncio
async def test_start_only_enabled_adapters():
    registry = ProtocolRegistry()
    tasks = start_protocol_adapters(
        Settings(ocpp_enabled=True, openadr_enabled=True), registry
    )
    try:
        for _ in range(50):
            if len(registry) == 2 and all(a.is_operational for a in registry.list_adapters()):
                break
            await asyncio.sleep(0.01)
        ocpp = registry.get("ocpp")
        oadr = registry.get("openadr")
        assert isinstance(ocpp, OCPPAdapter) and ocpp.status == ProtocolStatus.CONNECTED
        # Enabled but without a VTN URL -> honest SIMULATED status
        assert isinstance(oadr, OpenADRAdapter) and oadr.status == ProtocolStatus.SIMULATED
        assert registry.get("ieee2030_5") is None
    finally:
        await stop_protocol_adapters(tasks)
    assert len(registry) == 0


@pytest.mark.asyncio
async def test_lifespan_starts_and_stops_ocpp(
    monkeypatch, preserve_db_globals, isolated_protocol_registry
):
    test_settings = Settings(
        database_url="sqlite+aiosqlite:///./test_lifespan_ocpp.db",
        degradation_updater_enabled=False,
        ocpp_enabled=True,
    )
    monkeypatch.setattr(app_module, "get_settings", lambda: test_settings)
    fastapi_app = app_module.create_app()
    async with fastapi_app.router.lifespan_context(fastapi_app):
        for _ in range(50):
            if "ocpp" in isolated_protocol_registry:
                break
            await asyncio.sleep(0.01)
        assert isolated_protocol_registry.get("ocpp").is_connected
        assert len(fastapi_app.state.protocol_tasks) == 1
    assert "ocpp" not in isolated_protocol_registry


@pytest.mark.asyncio
async def test_api_reports_simulated_status(client, auth_headers, isolated_protocol_registry):
    ocpp = OCPPAdapter()
    isolated_protocol_registry.register(ocpp)

    resp = await client.post("/api/v1/protocols/ocpp/connect", headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "simulated"
    assert "simulated" in body["message"].lower()

    resp = await client.get("/api/v1/protocols/", headers=auth_headers)
    assert resp.status_code == 200
    [info] = resp.json()
    assert info["name"] == "ocpp"
    assert info["status"] == "simulated"
    assert info["mode"] == "simulated"
    assert info["simulated"] is True

    resp = await client.get("/api/v1/protocols/ocpp/metrics", headers=auth_headers)
    assert resp.json()["mode"] == "simulated"

    await ocpp.disconnect()


def test_ocpp_route_mounted_on_app(isolated_protocol_registry):
    """The real app serves /ocpp/{id}; with no live Central System registered
    the handshake is refused with 1013 (try again later), not 1000 (no route)."""
    from starlette.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect

    client = TestClient(app_module.create_app(rate_limit_enabled=False))
    with pytest.raises(WebSocketDisconnect) as info, client.websocket_connect(
        "/ocpp/CP1", subprotocols=["ocpp1.6"]
    ):
        pass
    assert info.value.code == 1013
