"""Tests for demo applications — verify they run without errors."""


class TestDemos:
    """Each demo should run to completion without raising exceptions."""

    def test_residential_demo(self, capsys):
        from vpp.demos.residential_demo import run

        run()
        captured = capsys.readouterr()
        assert "RESIDENTIAL VPP DEMO" in captured.out
        assert "Demo complete" in captured.out
        assert "Peak reduction" in captured.out

    def test_ev_fleet_demo(self, capsys):
        from vpp.demos.ev_fleet_demo import run

        run()
        captured = capsys.readouterr()
        assert "EV FLEET V2G DEMO" in captured.out
        assert "Demo complete" in captured.out
        assert "Smart V2G" in captured.out

    def test_microgrid_demo(self, capsys):
        from vpp.demos.microgrid_demo import run

        run()
        captured = capsys.readouterr()
        assert "MICROGRID ISLANDING DEMO" in captured.out
        assert "Demo complete" in captured.out
        assert "GRID FAULT DETECTED" in captured.out
        assert "Reconnecting" in captured.out

    def test_trading_demo(self, capsys):
        from vpp.demos.trading_demo import run

        run()
        captured = capsys.readouterr()
        assert "TRADING BOT DEMO" in captured.out
        assert "Demo complete" in captured.out
        assert "Sharpe ratio" in captured.out

    def test_protocols_demo(self, capsys):
        from vpp.demos.protocols_demo import run

        run()
        captured = capsys.readouterr()
        assert "MULTI-PROTOCOL DEMO" in captured.out
        assert "Demo complete" in captured.out
        assert "OpenADR" in captured.out
        assert "OCPP" in captured.out

    def test_protocols_demo_without_current_event_loop(self, capsys):
        """Regression: the demo used asyncio.get_event_loop(), which raises
        "There is no current event loop" when an earlier test (e.g. any
        pytest-asyncio test or asyncio.run call) has cleared the loop.
        Reproduce that state deterministically instead of relying on order."""
        import asyncio

        from vpp.demos.protocols_demo import run

        asyncio.set_event_loop(None)
        run()
        captured = capsys.readouterr()
        assert "Response: optIn (auto)" in captured.out
        assert "Demo complete" in captured.out

    def test_dashboard_demo(self, capsys):
        from vpp.demos.dashboard_demo import run

        run()
        captured = capsys.readouterr()
        assert "INTERACTIVE DASHBOARD DEMO" in captured.out
        assert "Demo complete" in captured.out
        assert "SESSION SUMMARY" in captured.out


def test_legacy_top_level_imports_alias_the_package_modules():
    """``demos`` / ``benchmarks`` at the repo root are aliases for the package."""
    import sys
    from pathlib import Path

    root = str(Path(__file__).resolve().parents[1])
    sys.path.insert(0, root)
    try:
        import benchmarks.scenarios as legacy_scenarios
        import vpp.benchmarks.scenarios as scenarios
        from benchmarks import ScenarioRegistry
        from demos.residential_demo import run as legacy_run
        from vpp.demos.residential_demo import run
    finally:
        sys.path.remove(root)

    assert legacy_scenarios is scenarios
    assert ScenarioRegistry is scenarios.ScenarioRegistry
    assert legacy_run is run


def test_cli_demo_and_benchmark_do_not_need_the_repo_root(tmp_path, monkeypatch):
    """`vpp demo` / `vpp benchmark` load from the package, not the CWD."""
    from click.testing import CliRunner

    from vpp.cli.main import cli

    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["demo", "residential"])
    assert result.exit_code == 0, result.output
    assert "Demo complete" in result.output

    result = runner.invoke(cli, ["benchmark", "list"])
    assert result.exit_code == 0, result.output
    assert "=== Scenarios ===" in result.output
