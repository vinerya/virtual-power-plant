"""Guard against drift between the ORM models and the alembic migrations.

Production deployments run ``alembic upgrade head`` (``VPP_USE_ALEMBIC=1``)
while tests and dev use ``Base.metadata.create_all``.  If a model is added or
changed without a matching migration, the two diverge silently and the bug
only surfaces in production.  This test fails whenever
``alembic.autogenerate.compare_metadata`` finds any difference, and prints
the operations a new revision would need.

Server defaults are intentionally *not* compared: several migrations add
``server_default`` values for columns whose ORM definition uses a Python-side
``default``; that is a deliberate, harmless difference.

By default the drift check runs against a throwaway SQLite file.  Set
``VPP_MIGRATION_TEST_DB_URL`` to a *sync* SQLAlchemy URL (e.g.
``postgresql+psycopg2://vpp:vpp@localhost:5432/vpp_migrations``) to run it
against a real server instead; CI does this against PostgreSQL.  The database
is upgraded to ``head`` (a no-op if it already is) and left there.
"""

from __future__ import annotations

import os
import pprint
from pathlib import Path

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, inspect

from vpp.db import models as _models  # noqa: F401  (register tables on Base.metadata)
from vpp.db.base import Base

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALEMBIC_INI = PROJECT_ROOT / "alembic.ini"
DB_URL_ENV = "VPP_MIGRATION_TEST_DB_URL"


def _sqlite_url(db_path: Path) -> str:
    return f"sqlite:///{db_path}"


def _make_config(db_url: str) -> Config:
    cfg = Config(str(ALEMBIC_INI))
    cfg.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    # ConfigParser interpolation: a literal "%" (URL-encoded passwords) must be doubled.
    cfg.set_main_option("sqlalchemy.url", db_url.replace("%", "%%"))
    return cfg


def test_single_migration_head(tmp_path: Path) -> None:
    """Concurrent branches must be merged; multiple heads break ``upgrade head``."""
    script = ScriptDirectory.from_config(_make_config(_sqlite_url(tmp_path / "unused.db")))
    heads = script.get_heads()
    assert len(heads) == 1, f"multiple alembic heads: {heads}"


def test_migrations_match_models(tmp_path: Path) -> None:
    """``alembic upgrade head`` yields exactly the schema in ``Base.metadata``."""
    db_url = os.environ.get(DB_URL_ENV) or _sqlite_url(tmp_path / "drift.db")
    command.upgrade(_make_config(db_url), "head")

    engine = create_engine(db_url)
    try:
        with engine.connect() as conn:
            ctx = MigrationContext.configure(conn, opts={"compare_type": True})
            diffs = compare_metadata(ctx, Base.metadata)
    finally:
        engine.dispose()

    assert diffs == [], (
        "ORM models and alembic migrations have drifted. Add a migration under "
        "alembic/versions/ covering:\n" + pprint.pformat(diffs)
    )


def test_tariffs_table_round_trips(tmp_path: Path) -> None:
    """Revision 0004 creates ``tariffs`` and its downgrade removes it cleanly."""
    db_url = _sqlite_url(tmp_path / "tariffs.db")
    cfg = _make_config(db_url)
    command.upgrade(cfg, "0004_add_tariffs")

    engine = create_engine(db_url)
    try:
        inspector = inspect(engine)
        assert "tariffs" in inspector.get_table_names()
        index_names = {ix["name"] for ix in inspector.get_indexes("tariffs")}
        assert {
            "ix_tariffs_name",
            "ix_tariffs_utility",
            "ix_tariffs_urdb_label",
        } <= index_names

        command.downgrade(cfg, "0003_add_nominal_energy")
        inspector = inspect(engine)
        assert "tariffs" not in inspector.get_table_names()
        created_at = {c["name"]: c for c in inspector.get_columns("resources")}["created_at"]
        assert created_at["nullable"] is True
    finally:
        engine.dispose()
