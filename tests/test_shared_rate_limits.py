"""Rate limiter and login throttle shared by several API workers (``shared_rate_limits``).

Each "worker" below is its own :class:`SharedLimitStore` on its own engine
pointing at one database, as separate processes would be. The tests run on a
throwaway SQLite file by default; set ``VPP_CLUSTER_TEST_DB_URL`` to an
*async* PostgreSQL URL to run them there (the table is recreated).
"""

from __future__ import annotations

import logging
import os

import pytest
from _v2g_helpers import tmp_session_factory
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from vpp.auth.middleware import RateLimitMiddleware
from vpp.auth.shared_limits import (
    DATABASE,
    MEMORY,
    SharedLimitStore,
    resolve_backend,
    storage_key,
)
from vpp.auth.throttle import LoginThrottle, SharedLoginThrottle
from vpp.db.models import SharedRateLimitModel
from vpp.settings import Settings, get_settings

DB_URL_ENV = "VPP_CLUSTER_TEST_DB_URL"


class Clock:
    def __init__(self, now: float = 1_800_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def make_factory(tmp_path):
    """Returns a function giving a *new* engine/session factory on the shared DB."""
    url = os.environ.get(DB_URL_ENV)
    if url:
        from sqlalchemy import create_engine

        from vpp.db.base import Base

        sync = create_engine(url.replace("+asyncpg", "+psycopg2"))
        tables = [SharedRateLimitModel.__table__]
        Base.metadata.drop_all(sync, tables=tables)
        Base.metadata.create_all(sync, tables=tables)
        sync.dispose()
    else:
        path = tmp_path / "limits.db"
        tmp_session_factory(path)  # creates the schema
        url = f"sqlite+aiosqlite:///{path}"

    def _make() -> async_sessionmaker:
        engine = create_async_engine(url, poolclass=NullPool)
        return async_sessionmaker(engine, expire_on_commit=False)

    return _make


@pytest.fixture
def broken_factory(tmp_path):
    """A database without the table: every statement raises OperationalError."""
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'empty.db'}", poolclass=NullPool
    )
    return async_sessionmaker(engine, expire_on_commit=False)


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("workers", "choice", "expected"),
    [
        (1, "auto", MEMORY),
        (4, "auto", DATABASE),
        (4, "memory", MEMORY),
        (1, "database", DATABASE),
        (2, " Database ", DATABASE),
    ],
)
def test_backend_selection(workers, choice, expected):
    settings = Settings(api_workers=workers, rate_limit_backend=choice)
    assert resolve_backend(settings) == expected


def test_backend_default_is_auto_and_validated():
    assert Settings().rate_limit_backend == "auto"
    with pytest.raises(ValidationError):
        Settings(rate_limit_backend="redis")


def _rate_limit_kwargs(app: FastAPI) -> dict:
    (mw,) = [m for m in app.user_middleware if m.cls is RateLimitMiddleware]
    return dict(mw.kwargs)


@pytest.mark.parametrize(("workers", "expected"), [(1, MEMORY), (3, DATABASE)])
def test_create_app_picks_backend_from_worker_count(monkeypatch, workers, expected):
    from vpp.api.app import create_app

    monkeypatch.setattr(get_settings(), "api_workers", workers)
    app = create_app(rate_limit_enabled=True, rate_limit_requests_per_minute=5)
    assert _rate_limit_kwargs(app)["backend"] == expected


def test_long_keys_are_hashed():
    assert storage_key("login", "bob") == "login:bob"
    long_key = storage_key("login", "x" * 400)
    assert len(long_key) <= 255 and long_key.startswith("login:sha256:")


def test_topology_log_mentions_backend(caplog):
    from vpp.cluster.topology import log_topology

    with caplog.at_level(logging.INFO, logger="vpp.cluster.topology"):
        log_topology(Settings(api_workers=2), {})
    assert "shared by all workers" in caplog.text
    assert "counts HTTP requests and failed logins separately" not in caplog.text
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="vpp.cluster.topology"):
        log_topology(Settings(api_workers=2, rate_limit_backend="memory"), {})
    assert "counts HTTP requests and failed logins separately" in caplog.text


# ---------------------------------------------------------------------------
# HTTP rate limiter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_two_stores_share_request_counts(make_factory):
    clock = Clock()
    a = SharedLimitStore(make_factory(), clock=clock)
    b = SharedLimitStore(make_factory(), clock=clock)
    key = storage_key("http", "198.51.100.1")
    results = [await (a if i % 2 else b).hit(key, 3) for i in range(6)]
    assert results == [True, True, True, False, False, False]
    # Another client is unaffected.
    assert await a.hit(storage_key("http", "198.51.100.2"), 3)


@pytest.mark.asyncio
async def test_sliding_window_and_refused_requests_are_not_counted(make_factory):
    clock = Clock(1_800_000_000.0 - 1_800_000_000.0 % 60)  # start of a window
    store = SharedLimitStore(make_factory(), clock=clock)
    key = storage_key("http", "198.51.100.1")
    assert [await store.hit(key, 4) for _ in range(6)] == [True] * 4 + [False] * 2
    # Half-way through the next window half of the previous one still counts:
    # 4 * 0.5 + hits <= 4 -> two more requests fit.
    clock.now += 90
    assert [await store.hit(key, 4) for _ in range(3)] == [True, True, False]
    # A window later only the (2) hits of the previous window count, scaled.
    clock.now += 60
    assert await store.hit(key, 4)
    # Two windows of silence: a fresh start.
    clock.now += 180
    assert [await store.hit(key, 4) for _ in range(5)] == [True] * 4 + [False]


@pytest.mark.asyncio
async def test_expired_rows_are_purged(make_factory):
    clock = Clock()
    factory = make_factory()
    store = SharedLimitStore(factory, clock=clock)
    await store.hit(storage_key("http", "198.51.100.1"), 3)
    await store.login_failure(storage_key("login", "bob"), 5, 60.0)
    assert await store.purge() == 0
    clock.now += 1_000
    assert await store.purge() == 2
    async with factory() as session:
        assert (await session.execute(select(SharedRateLimitModel))).first() is None


def _limited_app(*, backend: str, store: SharedLimitStore | None, rpm: int) -> FastAPI:
    app = FastAPI()

    @app.get("/ping")
    async def ping() -> dict[str, bool]:
        return {"ok": True}

    app.add_middleware(RateLimitMiddleware, requests_per_minute=rpm, backend=backend, store=store)
    return app


async def _get(app: FastAPI, n: int, peer: str = "203.0.113.7") -> list[int]:
    transport = ASGITransport(app=app, client=(peer, 40000))
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        return [(await client.get("/ping")).status_code for _ in range(n)]


@pytest.mark.asyncio
async def test_limit_holds_across_workers(make_factory):
    """Two apps (workers) with their own stores on one DB share the client's budget."""
    clock = Clock()
    w1 = _limited_app(backend=DATABASE, store=SharedLimitStore(make_factory(), clock=clock), rpm=4)
    w2 = _limited_app(backend=DATABASE, store=SharedLimitStore(make_factory(), clock=clock), rpm=4)
    assert await _get(w1, 2) == [200, 200]
    assert await _get(w2, 2) == [200, 200]
    assert await _get(w1, 1) == [429]
    assert await _get(w2, 1) == [429]
    assert await _get(w2, 1, peer="203.0.113.8") == [200]


@pytest.mark.asyncio
async def test_memory_backend_is_per_process(make_factory):
    """The single-worker default: no DB involved, each app has its own buckets."""
    w1 = _limited_app(backend=MEMORY, store=None, rpm=2)
    w2 = _limited_app(backend=MEMORY, store=None, rpm=2)
    assert await _get(w1, 3) == [200, 200, 429]
    assert await _get(w2, 3) == [200, 200, 429]


@pytest.mark.asyncio
async def test_rate_limiter_fails_open_on_db_error(broken_factory, caplog):
    app = _limited_app(backend=DATABASE, store=SharedLimitStore(broken_factory), rpm=1)
    with caplog.at_level(logging.WARNING, logger="vpp.auth.middleware"):
        assert await _get(app, 3) == [200, 200, 200]
    warnings = [r for r in caplog.records if "Shared rate limiter unavailable" in r.message]
    assert len(warnings) == 1  # logged once, not per request


@pytest.mark.asyncio
async def test_rate_limiter_fails_open_without_database(caplog, monkeypatch):
    """DB not initialised (e.g. before startup): allowed, with a warning."""
    import vpp.db.engine as engine_mod

    monkeypatch.setattr(engine_mod, "_session_factory", None)
    app = _limited_app(backend=DATABASE, store=None, rpm=1)
    with caplog.at_level(logging.WARNING, logger="vpp.auth.middleware"):
        assert await _get(app, 2) == [200, 200]
    assert "Shared rate limiter unavailable" in caplog.text


# ---------------------------------------------------------------------------
# Login throttle
# ---------------------------------------------------------------------------


@pytest.fixture
def login_settings(monkeypatch):
    monkeypatch.setattr(get_settings(), "login_max_failures", 3)
    monkeypatch.setattr(get_settings(), "login_lockout_seconds", 60)


def _worker(factory, clock: Clock) -> SharedLoginThrottle:
    return SharedLoginThrottle(
        LoginThrottle(clock=clock),
        SharedLimitStore(factory, clock=clock),
        backend=DATABASE,
    )


@pytest.mark.asyncio
async def test_shared_login_throttle_semantics(make_factory, login_settings):
    """Same scenario as the in-memory unit test, spread over two workers."""
    clock = Clock()
    w1, w2 = _worker(make_factory(), clock), _worker(make_factory(), clock)
    await w1.record_failure("Bob")
    await w2.record_failure("bob")
    assert await w1.retry_after("bob") is None
    await w2.record_failure("BOB ")
    assert await w1.retry_after("bob") == 60
    assert await w2.retry_after("Bob") == 60
    clock.now += 30
    assert await w1.retry_after("bob") == 30
    clock.now += 31
    assert await w2.retry_after("bob") is None
    await w1.record_failure("bob")
    await w2.record_success("bob")
    await w1.record_failure("bob")
    await w2.record_failure("bob")
    assert await w1.retry_after("bob") is None  # success cleared the earlier failure
    # Nothing was kept in memory: the database held all of it.
    assert w1.local.retry_after("bob") is None and not w1.local._entries


@pytest.mark.asyncio
async def test_shared_login_failures_expire_after_window(make_factory, login_settings):
    clock = Clock()
    w1, w2 = _worker(make_factory(), clock), _worker(make_factory(), clock)
    await w1.record_failure("alice")
    await w2.record_failure("alice")
    clock.now += 61  # older than the window: forgotten
    await w1.record_failure("alice")
    await w2.record_failure("alice")
    assert await w1.retry_after("alice") is None
    await w1.record_failure("alice")
    assert await w2.retry_after("alice") == 60
    # Other usernames are unaffected.
    assert await w2.retry_after("carol") is None


@pytest.mark.asyncio
async def test_shared_login_throttle_disabled_with_zero(make_factory, monkeypatch):
    monkeypatch.setattr(get_settings(), "login_max_failures", 0)
    factory = make_factory()
    w = _worker(factory, Clock())
    for _ in range(10):
        await w.record_failure("bob")
    assert await w.retry_after("bob") is None
    async with factory() as session:
        assert (await session.execute(select(SharedRateLimitModel))).first() is None


@pytest.mark.asyncio
async def test_login_throttle_falls_back_to_memory_on_db_error(
    broken_factory, login_settings, caplog
):
    clock = Clock()
    w = _worker(broken_factory, clock)
    with caplog.at_level(logging.WARNING, logger="vpp.auth.throttle"):
        for _ in range(3):
            await w.record_failure("bob")
        assert await w.retry_after("bob") == 60  # still throttled, in memory
        await w.record_success("bob")
        assert await w.retry_after("bob") is None
    assert "Shared login throttle unavailable" in caplog.text


@pytest.mark.asyncio
async def test_lockout_recorded_during_outage_is_honoured_after(
    make_factory, broken_factory, login_settings
):
    clock = Clock()
    local = LoginThrottle(clock=clock)
    down = SharedLoginThrottle(
        local, SharedLimitStore(broken_factory, clock=clock), backend=DATABASE
    )
    for _ in range(3):
        await down.record_failure("bob")
    up = SharedLoginThrottle(
        local, SharedLimitStore(make_factory(), clock=clock), backend=DATABASE
    )
    assert await up.retry_after("bob") == 60


@pytest.mark.asyncio
async def test_memory_backend_does_not_touch_the_database(broken_factory, login_settings, caplog):
    clock = Clock()
    w = SharedLoginThrottle(
        LoginThrottle(clock=clock), SharedLimitStore(broken_factory, clock=clock), backend=MEMORY
    )
    with caplog.at_level(logging.WARNING, logger="vpp.auth.throttle"):
        for _ in range(3):
            await w.record_failure("bob")
        assert await w.retry_after("bob") == 60
    assert "unavailable" not in caplog.text


@pytest.mark.asyncio
async def test_login_route_uses_the_shared_store(client, monkeypatch, login_settings):
    """End to end: with the database backend the lockout lives in shared_rate_limits."""
    from vpp.auth import throttle
    from vpp.db.engine import get_session_factory

    monkeypatch.setattr(get_settings(), "rate_limit_backend", "database")
    username = "ghost-shared-limits"
    codes = [
        (
            await client.post(
                "/api/v1/auth/token", data={"username": username, "password": "x" * 12}
            )
        ).status_code
        for _ in range(4)
    ]
    assert codes == [401, 401, 401, 429]
    assert throttle.login_throttle.retry_after(username) is None  # not in memory
    key = storage_key("login", username)
    async with get_session_factory()() as session:
        row = await session.get(SharedRateLimitModel, key)
        assert row is not None and row.locked_until > 0
    await throttle.shared_login_throttle.store.clear(key)
