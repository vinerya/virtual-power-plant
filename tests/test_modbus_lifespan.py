"""Tests for Modbus telemetry ingestion wired into the FastAPI lifespan.

Mirrors tests/test_mqtt_lifespan.py's pattern.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from vpp.api import app as app_module
from vpp.db import engine as db_engine
from vpp.db.engine import init_db
from vpp.db.repositories import ResourceRepository
from vpp.protocols.base import ProtocolRegistry
from vpp.settings import Settings


def _fresh_sqlite_url(filename: str) -> str:
    """Delete any leftover file from a prior run before returning its URL.

    Unlike the other lifespan tests, these create rows with a UNIQUE
    constraint (resources.name) -- Base.metadata.create_all() is
    idempotent and doesn't wipe existing data, so a stale file from a
    previous run would collide on re-insert.
    """
    try:
        os.remove(filename)
    except FileNotFoundError:
        pass
    return f"sqlite+aiosqlite:///./{filename}"


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
async def test_modbus_ingestion_disabled_by_default(monkeypatch, preserve_db_globals):
    """modbus_ingestion_enabled defaults to False -- no task is created."""
    called = False

    async def fake_loop(settings, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(app_module, "_modbus_ingestion_loop", fake_loop)

    test_settings = Settings(
        database_url="sqlite+aiosqlite:///./test_lifespan_modbus_default.db",
        degradation_updater_enabled=False,
    )
    monkeypatch.setattr(app_module, "get_settings", lambda: test_settings)

    fastapi_app = app_module.create_app()
    async with fastapi_app.router.lifespan_context(fastapi_app):
        assert fastapi_app.state.modbus_ingestion_task is None

    assert called is False


@pytest.mark.asyncio
async def test_modbus_ingestion_starts_when_enabled(monkeypatch, preserve_db_globals):
    started = asyncio.Event()

    async def fake_loop(settings, **kwargs):
        started.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            raise

    monkeypatch.setattr(app_module, "_modbus_ingestion_loop", fake_loop)

    test_settings = Settings(
        database_url="sqlite+aiosqlite:///./test_lifespan_modbus_enabled.db",
        degradation_updater_enabled=False,
        modbus_ingestion_enabled=True,
    )
    monkeypatch.setattr(app_module, "get_settings", lambda: test_settings)

    fastapi_app = app_module.create_app()
    async with fastapi_app.router.lifespan_context(fastapi_app):
        await asyncio.wait_for(started.wait(), timeout=2.0)
        task = fastapi_app.state.modbus_ingestion_task
        assert task is not None
        assert not task.done()

    assert task.cancelled() or task.done()


@pytest.mark.asyncio
async def test_modbus_ingestion_skips_resources_without_modbus_metadata(
    monkeypatch, preserve_db_globals
):
    """A resource with no `modbus` key in metadata must not spawn a device
    task -- _modbus_ingestion_loop should return with nothing to await."""
    db_url = _fresh_sqlite_url("test_lifespan_modbus_no_config.db")
    await init_db(db_url)
    session_factory = db_engine._session_factory
    async with session_factory() as session:
        await ResourceRepository.create(
            session,
            name="plain-battery",
            resource_type="battery",
            rated_power=5.0,
        )
        await session.commit()

    test_settings = Settings(
        database_url=db_url,
        degradation_updater_enabled=False,
        modbus_ingestion_enabled=True,
    )
    monkeypatch.setattr(app_module, "get_settings", lambda: test_settings)

    fastapi_app = app_module.create_app()
    async with fastapi_app.router.lifespan_context(fastapi_app):
        task = fastapi_app.state.modbus_ingestion_task
        assert task is not None
        # The real loop should finish quickly (no devices to connect) rather
        # than hang -- give it a moment then confirm it's done.
        await asyncio.wait_for(asyncio.shield(task), timeout=2.0)
        assert task.done()


@pytest.mark.asyncio
async def test_modbus_ingestion_loop_registers_and_unregisters_per_device_adapter(
    monkeypatch, preserve_db_globals, isolated_protocol_registry
):
    """The real _modbus_ingestion_loop (not a fake) must discover a
    modbus-configured resource, register its adapter under a name unique
    to that resource, and remove it on cancel -- even though connect()
    will fail (no real device)."""
    from vpp.protocols.modbus import ModbusAdapter

    async def failing_connect(self):
        raise ConnectionError("no device in this test")

    monkeypatch.setattr(ModbusAdapter, "connect", failing_connect)

    db_url = _fresh_sqlite_url("test_lifespan_modbus_registry.db")
    await init_db(db_url)
    session_factory = db_engine._session_factory
    async with session_factory() as session:
        resource = await ResourceRepository.create(
            session,
            name="modbus-inverter",
            resource_type="solar",
            rated_power=5.0,
            metadata={
                "modbus": {
                    "host": "127.0.0.1",
                    "port": 5020,
                    "device_profile": "generic_meter",
                }
            },
        )
        await session.commit()
        resource_id = resource.id

    test_settings = Settings(
        database_url=db_url,
        degradation_updater_enabled=False,
        modbus_ingestion_enabled=True,
    )
    monkeypatch.setattr(app_module, "get_settings", lambda: test_settings)

    fastapi_app = app_module.create_app()
    expected_name = f"modbus:{resource_id}"
    async with fastapi_app.router.lifespan_context(fastapi_app):
        for _ in range(50):
            if expected_name in isolated_protocol_registry:
                break
            await asyncio.sleep(0.02)
        assert expected_name in isolated_protocol_registry
        task = fastapi_app.state.modbus_ingestion_task
        assert task is not None

    assert expected_name not in isolated_protocol_registry
