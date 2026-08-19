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
