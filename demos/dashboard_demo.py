"""Interactive Dashboard Demo — live terminal UI.

Demonstrates a real-time terminal dashboard with:
- Resource status panel
- Optimisation results
- Trading P&L
- Alert feed
- Battery SOC gauges
"""

from __future__ import annotations

import time
import numpy as np


def _bar(value: float, max_val: float = 100.0, width: int = 30, char: str = "#") -> str:
    """Create a simple ASCII progress bar."""
    filled = int(width * min(value, max_val) / max_val)
    empty = width - filled
    pct = value / max_val * 100 if max_val > 0 else 0
    return f"[{char * filled}{'.' * empty}] {pct:5.1f}%"


def _spark(values: list[float], width: int = 20) -> str:
    """Create a sparkline from values."""
    if not values:
        return " " * width
    blocks = " _.-~*"
    mn, mx = min(values), max(values)
    rng = mx - mn if mx > mn else 1
    recent = values[-width:] if len(values) > width else values
    line = ""
    for v in recent:
        idx = int((v - mn) / rng * (len(blocks) - 1))
        line += blocks[idx]
    return line.ljust(width)


def run() -> None:
    """Run the interactive dashboard demo."""
    print("=" * 70)
    print("  INTERACTIVE DASHBOARD DEMO")
    print("=" * 70)

    np.random.seed(42)

    # Simulate 10 time steps of live data
    n_steps = 10
    solar_history: list[float] = []
    wind_history: list[float] = []
    load_history: list[float] = []
    pnl_history: list[float] = []
    freq_history: list[float] = []

    for step in range(n_steps):
        hour = 8.0 + step * 0.5  # 8:00 - 12:30

        # Simulate live values
        solar_kw = max(0, 45 * np.sin((hour - 6) * np.pi / 12) + np.random.randn() * 3)
        wind_kw = max(0, 20 + np.random.randn() * 8)
        load_kw = 60 + 15 * np.sin((hour - 14) * np.pi / 10) + np.random.randn() * 5
        batt_soc = 0.45 + step * 0.04 + np.random.randn() * 0.02
        batt_soc = np.clip(batt_soc, 0.1, 0.95)
        batt_kw = (solar_kw + wind_kw) - load_kw
        batt_kw = np.clip(batt_kw, -25, 25)
        grid_kw = load_kw - solar_kw - wind_kw - batt_kw
        freq = 60.0 + np.random.randn() * 0.015
        voltage = 1.0 + np.random.randn() * 0.008
        trading_pnl = sum(pnl_history) + np.random.randn() * 50 + 20
        ev_connected = int(35 + np.random.randint(-5, 5))
        ev_charging = int(ev_connected * 0.6)
        ev_v2g = int(ev_connected * 0.15)
        alerts_active = max(0, int(2 + np.random.randn()))

        solar_history.append(solar_kw)
        wind_history.append(wind_kw)
        load_history.append(load_kw)
        pnl_history.append(trading_pnl)
        freq_history.append(freq)

        # Render dashboard frame
        print(f"\n{'=' * 70}")
        print(f"  VPP LIVE DASHBOARD    {hour:05.1f}h    Step {step+1}/{n_steps}")
        print(f"{'=' * 70}")

        # Resource Panel
        print(f"\n  RESOURCES")
        print(f"  {'Solar':15} {solar_kw:>7.1f} kW  {_bar(solar_kw, 60, 25, '#')}")
        print(f"  {'Wind':15} {wind_kw:>7.1f} kW  {_bar(wind_kw, 40, 25, '~')}")
        print(f"  {'Load':15} {load_kw:>7.1f} kW  {_bar(load_kw, 100, 25, '=')}")
        print(f"  {'Battery':15} {batt_kw:>+7.1f} kW  SOC: {_bar(batt_soc * 100, 100, 20, '|')}")
        print(f"  {'Grid':15} {grid_kw:>+7.1f} kW  {'IMPORTING' if grid_kw > 0 else 'EXPORTING'}")

        # Grid quality
        print(f"\n  GRID QUALITY")
        print(f"  Frequency : {freq:.3f} Hz  {'OK' if 59.95 < freq < 60.05 else 'WARNING'}")
        print(f"  Voltage   : {voltage:.3f} pu  {'OK' if 0.95 < voltage < 1.05 else 'WARNING'}")
        print(f"  Freq trend: {_spark(freq_history, 30)}")

        # EV Fleet
        print(f"\n  EV FLEET")
        print(f"  Connected: {ev_connected}   Charging: {ev_charging}   V2G: {ev_v2g}")
        print(f"  Fleet load: {ev_charging * 7.4:.0f} kW   V2G discharge: {ev_v2g * 5:.0f} kW")

        # Trading
        print(f"\n  TRADING")
        print(f"  Session P&L : ${trading_pnl:>+9.2f}")
        print(f"  P&L trend   : {_spark(pnl_history, 30)}")

        # Alerts
        if alerts_active > 0:
            alert_types = [
                ("WARNING", "Battery SOC below 30%"),
                ("INFO", "DR event scheduled for 14:00"),
                ("ERROR", "Modbus connection timeout"),
            ]
            print(f"\n  ALERTS ({alerts_active} active)")
            for sev, msg in alert_types[:alerts_active]:
                indicator = {"WARNING": "!", "ERROR": "X", "INFO": "i"}.get(sev, " ")
                print(f"  [{indicator}] {sev:>8}: {msg}")

        # Optimisation
        opt_status = "OPTIMAL" if step > 2 else "RUNNING"
        opt_savings = 12.50 + step * 3.20 + np.random.randn() * 2
        print(f"\n  OPTIMISATION")
        print(f"  Status    : {opt_status}")
        print(f"  Savings   : ${opt_savings:.2f} (this hour)")
        print(f"  Peak shave: {_bar(min(35 + step * 5, 85), 100, 20, '>')}")

        # Brief pause to simulate real-time updates
        # (in a real dashboard, this would be continuous)

    # Final summary
    print(f"\n{'=' * 70}")
    print(f"  SESSION SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Duration      : {n_steps * 0.5:.1f} hours")
    print(f"  Avg solar     : {np.mean(solar_history):.1f} kW")
    print(f"  Avg wind      : {np.mean(wind_history):.1f} kW")
    print(f"  Avg load      : {np.mean(load_history):.1f} kW")
    print(f"  Peak load     : {np.max(load_history):.1f} kW")
    print(f"  Final P&L     : ${pnl_history[-1]:+.2f}")
    print(f"  Freq stability: {np.std(freq_history)*1000:.1f} mHz std dev")

    print("\n[Demo complete]")
    print("\nNote: In production, use `vpp serve` + Grafana for real-time web dashboards.")


if __name__ == "__main__":
    run()
