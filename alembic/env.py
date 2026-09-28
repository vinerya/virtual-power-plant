"""Alembic migration environment for the Virtual Power Plant platform.

Design notes
============

The application runs on an *async* SQLAlchemy engine (``aiosqlite`` /
``asyncpg``).  Alembic's online mode supports async, but the simplest and
most portable approach is to:

1. Read the configured database URL from :class:`vpp.settings.Settings`.
2. Rewrite any async driver (``+aiosqlite``, ``+asyncpg``) to its sync
   equivalent (plain ``sqlite``, ``postgresql+psycopg2``) so that
   ``engine_from_config`` can run ordinary ``op.*`` operations.
3. Use ``Base.metadata`` from :mod:`vpp.db.models` as the autogenerate
   target so that future ``alembic revision --autogenerate`` calls Just
   Work.

This keeps migrations decoupled from the runtime async stack while
honouring the same configuration source.
"""

from __future__ import annotations

import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# Make ``src`` importable when alembic is invoked from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vpp.db.base import Base  # noqa: E402
from vpp.db import models as _models  # noqa: F401,E402  (register tables on Base.metadata)
from vpp.settings import get_settings  # noqa: E402

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)


def _resolve_url() -> str:
    """Return a *sync* SQLAlchemy URL suitable for alembic.

    Order of precedence:
        1. ``-x url=...`` passed on the alembic CLI.
        2. ``sqlalchemy.url`` set in ``alembic.ini``.
        3. ``Settings.database_url`` (which is normally async).

    Async driver suffixes are stripped so that alembic can use plain DBAPI.
    """
    x_args = context.get_x_argument(as_dictionary=True)
    if "url" in x_args:
        url = x_args["url"]
    else:
        url = config.get_main_option("sqlalchemy.url") or ""
        if not url:
            url = get_settings().database_url

    # Strip async drivers so alembic uses sync DBAPI.
    if url.startswith("sqlite+aiosqlite"):
        url = url.replace("sqlite+aiosqlite", "sqlite", 1)
    elif url.startswith("postgresql+asyncpg"):
        url = url.replace("postgresql+asyncpg", "postgresql+psycopg2", 1)
    return url


target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode -- emit SQL to stdout."""
    url = _resolve_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=url.startswith("sqlite"),
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations using a sync engine derived from settings."""
    url = _resolve_url()
    cfg = config.get_section(config.config_ini_section, {}) or {}
    cfg["sqlalchemy.url"] = url

    connectable = engine_from_config(
        cfg,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=url.startswith("sqlite"),
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
