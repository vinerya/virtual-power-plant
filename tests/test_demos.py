"""Tests for demo applications — verify they run without errors."""

import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestDemos:
    """Each demo should run to completion without raising exceptions."""

    def test_residential_demo(self, capsys):
        from demos.residential_demo import run
        run()
        captured = capsys.readouterr()
        assert "RESIDENTIAL VPP DEMO" in captured.out
        assert "Demo complete" in captured.out
        assert "Peak reduction" in captured.out

    def test_ev_fleet_demo(self, capsys):
        from demos.ev_fleet_demo import run
        run()
        captured = capsys.readouterr()
        assert "EV FLEET V2G DEMO" in captured.out
        assert "Demo complete" in captured.out
        assert "Smart V2G" in captured.out

    def test_microgrid_demo(self, capsys):
        from demos.microgrid_demo import run
        run()
        captured = capsys.readouterr()
        assert "MICROGRID ISLANDING DEMO" in captured.out
        assert "Demo complete" in captured.out
        assert "GRID FAULT DETECTED" in captured.out
        assert "Reconnecting" in captured.out

    def test_trading_demo(self, capsys):
        from demos.trading_demo import run
        run()
        captured = capsys.readouterr()
        assert "TRADING BOT DEMO" in captured.out
        assert "Demo complete" in captured.out
        assert "Sharpe ratio" in captured.out

    def test_protocols_demo(self, capsys):
        from demos.protocols_demo import run
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

        from demos.protocols_demo import run

        asyncio.set_event_loop(None)
        run()
        captured = capsys.readouterr()
        assert "Response: optIn (auto)" in captured.out
        assert "Demo complete" in captured.out

    def test_dashboard_demo(self, capsys):
        from demos.dashboard_demo import run
        run()
        captured = capsys.readouterr()
        assert "INTERACTIVE DASHBOARD DEMO" in captured.out
        assert "Demo complete" in captured.out
        assert "SESSION SUMMARY" in captured.out
