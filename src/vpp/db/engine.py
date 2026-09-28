"""Async database engine and session management."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Optional

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from .base import Base

_engine: Optional[AsyncEngine] = None
_session_factory: Optional[async_sessionmaker[AsyncSession]] = None


def create_engine_from_settings(database_url: str, echo: bool = False) -> AsyncEngine:
    """Create an async engine from a database URL.

    Supports ``sqlite+aiosqlite`` (dev) and ``postgresql+asyncpg`` (prod).
    """
    connect_args: dict = {}
    if "sqlite" in database_url:
        connect_args["check_same_thread"] = False

    return create_async_engine(
        database_url,
        echo=echo,
        connect_args=connect_args,
        pool_pre_ping=True,
    )


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
    """
    global _engine, _session_factory

    _engine = create_engine_from_settings(database_url, echo=echo)
    _session_factory = async_sessionmaker(_engine, expire_on_commit=False)

    if use_alembic:
        await _run_alembic_upgrade(database_url)
    else:
        async with _engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)


def run_alembic_upgrade(database_url: str, revision: str = "head") -> None:
    """Run ``alembic upgrade <revision>`` synchronously against *database_url*.

    Async driver suffixes are rewritten to sync DBAPIs by ``alembic/env.py``.
    Requires a source checkout (``alembic.ini`` at the repository root).
    """
    from pathlib import Path

    from alembic import command
    from alembic.config import Config

    # Locate alembic.ini at the repo root.
    ini = Path(__file__).resolve().parents[3] / "alembic.ini"
    if not ini.is_file():
        raise FileNotFoundError(
            f"alembic.ini not found at {ini}; migrations require a source checkout"
        )
    cfg = Config(str(ini))
    # Pass the runtime URL via -x so env.py's _resolve_url picks it up.
    cfg.cmd_opts = type("X", (), {"x": [f"url={database_url}"]})()
    command.upgrade(cfg, revision)


async def _run_alembic_upgrade(database_url: str) -> None:
    """Run ``alembic upgrade head`` off the event loop."""
    import asyncio

    await asyncio.get_running_loop().run_in_executor(
        None, run_alembic_upgrade, database_url
    )


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields an async database session."""
    if _session_factory is None:
        raise RuntimeError(
            "Database not initialised. Call init_db() during application startup."
        )

    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the global session factory.  Raises if DB not initialised."""
    if _session_factory is None:
        raise RuntimeError(
            "Database not initialised. Call init_db() during application startup."
        )
    return _session_factory


async def close_db() -> None:
    """Dispose of the engine connection pool."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
