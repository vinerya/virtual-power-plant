"""Async database engine and session management."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from .base import Base

if TYPE_CHECKING:
    from alembic.config import Config

logger = logging.getLogger(__name__)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None

_T = TypeVar("_T")


def create_engine_from_settings(database_url: str, echo: bool = False) -> AsyncEngine:
    """Create an async engine from a database URL.

    Supports ``sqlite+aiosqlite`` (dev) and ``postgresql+asyncpg`` (prod).
    """
    connect_args: dict = {}
    if "sqlite" in database_url:
        connect_args["check_same_thread"] = False

    engine = create_async_engine(
        database_url,
        echo=echo,
        connect_args=connect_args,
        pool_pre_ping=True,
    )
    from vpp import tracing

    tracing.instrument_sqlalchemy(engine)  # no-op unless tracing is on
    return engine


async def init_db(
    database_url: str,
    echo: bool = False,
    use_alembic: bool = False,
) -> None:
    """Initialise the global engine, session factory, and create tables.

    By default schemas are bootstrapped via ``Base.metadata.create_all``
    (backwards compatible with pre-M4 dev workflows).  Set
    ``use_alembic=True`` (or ``VPP_USE_ALEMBIC=1`` in the environment) to
    run ``alembic upgrade head`` against the configured database instead.

    Several API workers start at once: on PostgreSQL the bootstrap runs under
    an advisory lock so they do not race each other's ``CREATE TABLE`` /
    migrations; on SQLite a lost ``create_all`` race is retried.
    """
    global _engine, _session_factory

    _engine = create_engine_from_settings(database_url, echo=echo)
    _session_factory = async_sessionmaker(_engine, expire_on_commit=False)

    if use_alembic:
        async with _schema_lock(_engine):
            await _run_alembic_upgrade(database_url)
    else:
        for attempt in range(3):
            try:
                async with _schema_lock(_engine), _engine.begin() as conn:
                    await conn.run_sync(Base.metadata.create_all)
                break
            except OperationalError:
                # SQLite: another process created a table between our check
                # and our CREATE; checkfirst skips it on the next attempt.
                if attempt == 2 or _engine.dialect.name != "sqlite":
                    raise


# Arbitrary constant: pg_advisory_lock key for schema bootstrap.
_SCHEMA_LOCK_KEY = 0x565050_0001


@asynccontextmanager
async def _schema_lock(engine: AsyncEngine) -> AsyncIterator[None]:
    """Hold a cluster-wide lock on PostgreSQL; no-op elsewhere."""
    if engine.dialect.name != "postgresql":
        yield
        return
    async with engine.connect() as conn:
        await conn.execute(text("SELECT pg_advisory_lock(:k)"), {"k": _SCHEMA_LOCK_KEY})
        try:
            yield
        finally:
            await conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": _SCHEMA_LOCK_KEY})


def migrations_dir() -> Path:
    """Return the directory holding the packaged alembic environment."""
    return Path(__file__).resolve().parent.parent / "migrations"


def alembic_config(database_url: str | None = None) -> Config:
    """Build an alembic :class:`~alembic.config.Config` for the packaged migrations.

    No ``alembic.ini`` is needed: ``script_location`` points at
    ``vpp/migrations`` inside the installed package, so this works from any
    working directory and from a wheel install.  When *database_url* is given
    it is passed to ``env.py`` as ``-x url=...`` (async drivers are rewritten
    to sync DBAPIs there).
    """
    from alembic.config import Config

    cfg = Config()
    cfg.set_main_option("script_location", str(migrations_dir()))
    if database_url is not None:
        cfg.cmd_opts = argparse.Namespace(x=[f"url={database_url}"])
    return cfg


def run_alembic_upgrade(database_url: str, revision: str = "head") -> None:
    """Run ``alembic upgrade <revision>`` synchronously against *database_url*."""
    from alembic import command

    command.upgrade(alembic_config(database_url), revision)


async def _run_alembic_upgrade(database_url: str) -> None:
    """Run ``alembic upgrade head`` off the event loop."""
    import asyncio

    await asyncio.get_running_loop().run_in_executor(None, run_alembic_upgrade, database_url)


#: ``session.info`` key holding rows queued with :func:`defer_row`.
_DEFERRED_KEY = "vpp.deferred_rows"


def defer_row(session: AsyncSession, row: object, *, on_error: bool = False) -> None:
    """Queue ``row`` to be inserted in its own transaction when the request ends.

    Used for the audit log (:mod:`vpp.audit`). The row is written *after* the
    request's own transaction has committed (or, with ``on_error=True``, also
    after it has been rolled back -- e.g. a failed login that answers 401).
    A failure to insert it is logged and swallowed, so it never turns a
    successful request into an error or undoes the request's own changes.
    Only sessions from :func:`get_db` honour the queue.
    """
    session.info.setdefault(_DEFERRED_KEY, []).append((row, on_error))


async def _write_deferred(session: AsyncSession, *, failed: bool) -> None:
    queued = session.info.pop(_DEFERRED_KEY, [])
    rows = [row for row, on_error in queued if on_error or not failed]
    if not rows:
        return
    try:
        session.add_all(rows)
        await session.commit()
    except Exception:
        logger.warning("Could not write %d deferred row(s) (audit log)", len(rows), exc_info=True)
        with contextlib.suppress(Exception):
            await session.rollback()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields an async database session.

    Commits when the request succeeds, rolls back when it raises; rows
    queued with :func:`defer_row` are written afterwards.
    """
    if _session_factory is None:
        raise RuntimeError("Database not initialised. Call init_db() during application startup.")

    async with _session_factory() as session:
        try:
            yield session
            # Commit and the post-commit audit write run shielded, so a
            # cancelled request cannot interrupt them mid-statement.
            await run_shielded(lambda: _commit_and_write_deferred(session))
        except Exception:
            await run_shielded(lambda: _rollback_and_write_deferred(session))
            raise


async def _commit_and_write_deferred(session: AsyncSession) -> None:
    await session.commit()
    await _write_deferred(session, failed=False)


async def _rollback_and_write_deferred(session: AsyncSession) -> None:
    await session.rollback()
    await _write_deferred(session, failed=True)


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the global session factory.  Raises if DB not initialised."""
    if _session_factory is None:
        raise RuntimeError("Database not initialised. Call init_db() during application startup.")
    return _session_factory


# ---------------------------------------------------------------------------
# Cancellation-safe units of DB work (background tasks)
# ---------------------------------------------------------------------------


async def run_shielded(work: Callable[[], Awaitable[_T]]) -> _T:
    """Run one unit of DB work to completion even if the caller is cancelled.

    Cancelling a task while it awaits a DB call interrupts SQLAlchemy
    mid-statement: the connection is invalidated without a rollback and, on
    SQLite, keeps the database write lock until it is garbage collected, so
    every other writer fails with "database is locked" (on PostgreSQL it
    leaves a connection "idle in transaction" holding row locks).

    ``work()`` therefore runs in its own task. When the caller is cancelled,
    it waits (ignoring further cancellations) for that task to finish -- its
    session commits or rolls back and closes -- and then re-raises the
    ``CancelledError``. Use it around each *single* transaction (or a short,
    bounded sequence of them), never around a long-running loop, so a
    cancelled background task still stops promptly.
    """
    task = asyncio.ensure_future(work())
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue  # cancelled again (or the work itself was): keep waiting
            except Exception:
                break
        if not task.cancelled() and task.exception() is not None:
            logger.warning(
                "DB work finished with an error after its caller was cancelled",
                exc_info=task.exception(),
            )
        raise


async def run_in_session(
    session_factory: Callable[[], AsyncSession],
    work: Callable[[AsyncSession], Awaitable[_T]],
    *,
    commit: bool = True,
) -> _T:
    """Run ``work(session)`` in a fresh session as one cancellation-safe transaction.

    Commits when ``work`` returns (unless ``commit=False``), rolls back when
    it raises, and always closes the session -- all before a cancellation of
    the caller proceeds (see :func:`run_shielded`).
    """

    async def unit() -> _T:
        async with session_factory() as session:
            try:
                result = await work(session)
                if commit:
                    await session.commit()
                return result
            except BaseException:
                with contextlib.suppress(Exception):
                    await session.rollback()
                raise

    return await run_shielded(unit)


async def close_db() -> None:
    """Dispose of the engine connection pool."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
