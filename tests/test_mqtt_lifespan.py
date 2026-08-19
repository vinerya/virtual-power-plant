"""Tests for MQTT telemetry ingestion wired into the FastAPI lifespan (M5).

Mirrors tests/test_app_lifespan.py's pattern for the degradation updater.
"""

from __future__ import annotations

import asyncio

import pytest

from vpp.api import app as app_module
from vpp.db import engine as db_engine
from vpp.protocols.base import ProtocolRegistry
from vpp.settings import Settings


@pytest.fixture
def preserve_db_globals():
    """See tests/test_app_lifespan.py -- avoids tearing down the shared
    session-scoped test database when a real app lifespan runs."""
    saved_engine = db_engine._engine
    saved_factory = db_engine._session_factory
    yield
    db_engine._engine = saved_engine
    db_engine._session_factory = saved_factory


@pytest.fixture
def isolated_protocol_registry(monkeypatch):
    """Swap in a scratch ProtocolRegistry so tests don't pollute (or
    collide with) the module-level singleton used by other test modules."""
    from vpp.api.routes import protocols as protocols_module

    registry = ProtocolRegistry()
    monkeypatch.setattr(protocols_module, "_registry", registry)
    return registry


@pytest.mark.asyncio
async def test_mqtt_ingestion_disabled_by_default(monkeypatch, preserve_db_globals):
    """mqtt_ingestion_enabled defaults to False -- no task is created."""
    called = False

    async def fake_loop(settings, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(app_module, "_mqtt_ingestion_loop", fake_loop)

    test_settings = Settings(
        database_url="sqlite+aiosqlite:///./test_lifespan_mqtt_default.db",
        degradation_updater_enabled=False,
    )
    monkeypatch.setattr(app_module, "get_settings", lambda: test_settings)

    fastapi_app = app_module.create_app()
    async with fastapi_app.router.lifespan_context(fastapi_app):
        assert fastapi_app.state.mqtt_ingestion_task is None

    assert called is False


@pytest.mark.asyncio
async def test_mqtt_ingestion_starts_when_enabled(monkeypatch, preserve_db_globals):
    started = asyncio.Event()

    async def fake_loop(settings, **kwargs):
        started.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            raise

    monkeypatch.setattr(app_module, "_mqtt_ingestion_loop", fake_loop)

    test_settings = Settings(
        database_url="sqlite+aiosqlite:///./test_lifespan_mqtt_enabled.db",
        degradation_updater_enabled=False,
        mqtt_ingestion_enabled=True,
    )
    monkeypatch.setattr(app_module, "get_settings", lambda: test_settings)

    fastapi_app = app_module.create_app()
    async with fastapi_app.router.lifespan_context(fastapi_app):
        await asyncio.wait_for(started.wait(), timeout=2.0)
        task = fastapi_app.state.mqtt_ingestion_task
        assert task is not None
        assert not task.done()

    assert task.cancelled() or task.done()


@pytest.mark.asyncio
async def test_mqtt_ingestion_loop_registers_and_unregisters_adapter(
    monkeypatch, preserve_db_globals, isolated_protocol_registry
):
    """The real _mqtt_ingestion_loop (not a fake) must register the MQTT
    adapter into the shared registry on start and remove it on cancel --
    even though connect() will fail (no broker), so GET /api/v1/protocols
    doesn't silently stay empty just because the ingestion task exists."""
    from vpp.protocols.mqtt import MQTTAdapter

    async def failing_connect(self):
        raise ConnectionError("no broker in this test")

    monkeypatch.setattr(MQTTAdapter, "connect", failing_connect)

    test_settings = Settings(
        database_url="sqlite+aiosqlite:///./test_lifespan_mqtt_registry.db",
        degradation_updater_enabled=False,
        mqtt_ingestion_enabled=True,
    )
    monkeypatch.setattr(app_module, "get_settings", lambda: test_settings)

    fastapi_app = app_module.create_app()
    async with fastapi_app.router.lifespan_context(fastapi_app):
        # Give the loop a moment to register before its first (failing)
        # connect attempt.
        for _ in range(50):
            if len(isolated_protocol_registry) >= 1:
                break
            await asyncio.sleep(0.02)
        assert "mqtt" in isolated_protocol_registry
        task = fastapi_app.state.mqtt_ingestion_task
        assert task is not None

    assert "mqtt" not in isolated_protocol_registry
