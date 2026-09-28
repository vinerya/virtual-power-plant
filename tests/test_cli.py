"""Tests for the `vpp` CLI (src/vpp/cli/main.py)."""

from __future__ import annotations

from click.testing import CliRunner

from vpp.cli.main import cli


def test_dispatch_command_does_not_crash():
    """`vpp dispatch <target>` must not raise. Regression guard: a dead,
    shadowed `vpp/config.py` module used to sit alongside the `vpp/config/`
    package with an incompatible VPPConfig(name: str) constructor — it was
    never actually reachable (packages always win over same-named modules),
    but its presence could mislead someone into "fixing" the wrong file."""
    runner = CliRunner()
    result = runner.invoke(cli, ["dispatch", "10.0"])

    assert result.exit_code == 0, result.output
    assert result.exception is None


def test_migrate_command_upgrades_to_head(tmp_path, monkeypatch):
    """`vpp migrate` runs `alembic upgrade head` (it used to be a no-op stub)."""
    from sqlalchemy import create_engine, inspect

    from vpp.settings import get_settings

    db_path = tmp_path / "cli_migrate.db"
    monkeypatch.setenv("VPP_DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
    get_settings.cache_clear()
    try:
        result = CliRunner().invoke(cli, ["migrate"])
    finally:
        get_settings.cache_clear()

    assert result.exit_code == 0, result.output
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        tables = set(inspect(engine).get_table_names())
    finally:
        engine.dispose()
    assert {"alembic_version", "resources", "tariffs"} <= tables
