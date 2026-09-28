"""Tests for the alembic migration sequence introduced in M4."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALEMBIC_INI = PROJECT_ROOT / "alembic.ini"


def _make_config(db_path: Path) -> Config:
    cfg = Config(str(ALEMBIC_INI))
    cfg.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    return cfg


@pytest.fixture
def fresh_db(tmp_path: Path) -> Path:
    db = tmp_path / "alembic_test.db"
    yield db
    if db.exists():
        os.remove(db)


def test_baseline_migration_runs_clean(fresh_db: Path):
    """A pristine sqlite DB upgrades to head with all expected tables."""
    cfg = _make_config(fresh_db)
    command.upgrade(cfg, "head")

    engine = create_engine(f"sqlite:///{fresh_db}")
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())

    expected = {
        "alembic_version",
        "resources",
        "battery_states",
        "battery_soh_samples",
        "optimization_runs",
        "orders",
        "trades",
        "users",
        "api_keys",
        "event_log",
        "tariffs",
    }
    assert expected.issubset(tables), f"missing: {expected - tables}"


def test_degradation_columns_present_after_upgrade(fresh_db: Path):
    """After upgrade, resources has the M3+M4 degradation columns."""
    cfg = _make_config(fresh_db)
    command.upgrade(cfg, "head")

    engine = create_engine(f"sqlite:///{fresh_db}")
    inspector = inspect(engine)
    cols = {c["name"] for c in inspector.get_columns("resources")}

    for required in (
        "state_of_health",
        "cumulative_throughput_kwh",
        "last_degradation_update",
        "chemistry",
        "nominal_energy_kwh",
    ):
        assert required in cols, f"resources missing {required!r}"


def test_downgrade_baseline(fresh_db: Path):
    """Downgrade back to base leaves only the alembic_version table."""
    cfg = _make_config(fresh_db)
    command.upgrade(cfg, "head")
    command.downgrade(cfg, "base")

    engine = create_engine(f"sqlite:///{fresh_db}")
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    # Either alembic_version remains (with no rows) or no app tables remain.
    assert tables.issubset({"alembic_version"})


def test_upgrade_from_pre_m3(fresh_db: Path):
    """A pre-M3 DB stamped at 0001 can upgrade and gain the new columns."""
    cfg = _make_config(fresh_db)

    # Simulate a pre-M3 deployment: only the baseline schema exists.
    command.upgrade(cfg, "0001_baseline")

    engine = create_engine(f"sqlite:///{fresh_db}")

    # Insert a row that pre-dates the degradation columns.
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO resources (id, name, resource_type, rated_power) "
                "VALUES ('legacy-1', 'legacy-battery', 'battery', 50.0)"
            )
        )

    # Sanity: column does not exist yet.
    inspector = inspect(engine)
    cols_before = {c["name"] for c in inspector.get_columns("resources")}
    assert "state_of_health" not in cols_before
    assert "nominal_energy_kwh" not in cols_before

    # Upgrade to head and confirm columns appear, data preserved.
    command.upgrade(cfg, "head")

    inspector = inspect(engine)
    cols_after = {c["name"] for c in inspector.get_columns("resources")}
    assert "state_of_health" in cols_after
    assert "nominal_energy_kwh" in cols_after

    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT name, rated_power, state_of_health, nominal_energy_kwh "
                "FROM resources WHERE id='legacy-1'"
            )
        ).one()
    assert row.name == "legacy-battery"
    assert row.rated_power == 50.0
    assert row.state_of_health == 1.0
    assert row.nominal_energy_kwh is None
