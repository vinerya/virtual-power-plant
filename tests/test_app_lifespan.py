"""Tests for the FastAPI lifespan handler introduced in M4."""

from __future__ import annotations

import asyncio

import pytest

from vpp.api import app as app_module
from vpp.db import engine as db_engine
from vpp.settings import Settings


@pytest.fixture
def preserve_db_globals():
    """Save and restore the global async engine/session_factory.

    The lifespan handler calls ``close_db()`` on exit, which would otherwise
    tear down the test-suite's session-scoped database.  This fixture
    snapshots the globals before each lifespan test and restores them
    afterwards so the rest of the suite continues to work.
    """
    saved_engine = db_engine._engine
    saved_factory = db_engine._session_factory
    yield
    db_engine._engine = saved_engine
    db_engine._session_factory = saved_factory


@pytest.mark.asyncio
async def test_lifespan_starts_updater(monkeypatch, preserve_db_globals):
    """If degradation_updater_enabled is True, a background task is created."""
    started = asyncio.Event()

    async def fake_loop(interval_minutes: int) -> None:
        started.set()
        try:
            # block until cancelled
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            raise

    monkeypatch.setattr(app_module, "_degradation_periodic_loop", fake_loop)

    test_settings = Settings(
        database_url="sqlite+aiosqlite:///./test_lifespan_a.db",
        degradation_updater_enabled=True,
    )
    monkeypatch.setattr(app_module, "get_settings", lambda: test_settings)

    fastapi_app = app_module.create_app()
    async with fastapi_app.router.lifespan_context(fastapi_app):
        await asyncio.wait_for(started.wait(), timeout=2.0)
        task = fastapi_app.state.degradation_task
        assert task is not None
        assert not task.done()


@pytest.mark.asyncio
async def test_lifespan_cancels_on_shutdown(monkeypatch, preserve_db_globals):
    """The background task must be cancelled cleanly on shutdown."""
    started = asyncio.Event()

    async def fake_loop(interval_minutes: int) -> None:
        started.set()
        await asyncio.sleep(3600)

    monkeypatch.setattr(app_module, "_degradation_periodic_loop", fake_loop)

    test_settings = Settings(
        database_url="sqlite+aiosqlite:///./test_lifespan_b.db",
        degradation_updater_enabled=True,
    )
    monkeypatch.setattr(app_module, "get_settings", lambda: test_settings)

    fastapi_app = app_module.create_app()
    async with fastapi_app.router.lifespan_context(fastapi_app):
        await asyncio.wait_for(started.wait(), timeout=2.0)
        task = fastapi_app.state.degradation_task
        assert task is not None

    assert task.cancelled() or task.done()


@pytest.mark.asyncio
async def test_disable_via_settings(monkeypatch, preserve_db_globals):
    """When the feature flag is False, no background task is created."""
    called = False

    async def fake_loop(interval_minutes: int) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(app_module, "_degradation_periodic_loop", fake_loop)

    test_settings = Settings(
        database_url="sqlite+aiosqlite:///./test_lifespan_c.db",
        degradation_updater_enabled=False,
    )
    monkeypatch.setattr(app_module, "get_settings", lambda: test_settings)

    fastapi_app = app_module.create_app()
    async with fastapi_app.router.lifespan_context(fastapi_app):
        assert fastapi_app.state.degradation_task is None

    assert called is False
