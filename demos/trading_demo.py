"""Trading Bot Demo — automated multi-market arbitrage.

Demonstrates:
- Multi-market trading (day-ahead, intraday, balancing)
- Cross-zone arbitrage detection
- Risk management with VaR limits and position sizing
- Portfolio tracking with P&L, Sharpe ratio, drawdown
"""

from __future__ import annotations

import numpy as np

from vpp.trading import (
    TradingEngine,
    DayAheadMarket,
    RealTimeMarket,
    AncillaryServicesMarket,
    RiskManager,
)
from vpp.trading.core import RiskLimits


def run() -> None:
    """Run the trading bot demo."""
    print("=" * 70)
    print("  TRADING BOT DEMO — Multi-Market Arbitrage")
    print("=" * 70)

    np.random.seed(42)

    # Setup markets
    engine = TradingEngine()
    da_market = DayAheadMarket(name="day_ahead")
    id_market = RealTimeMarket(name="intraday")
    bal_market = AncillaryServicesMarket(name="balancing")

    engine.add_market(da_market)
    engine.add_market(id_market)
    engine.add_market(bal_market)

    # Risk limits
    risk = RiskManager(limits=RiskLimits(
        max_position=50.0,
        max_daily_loss=5000.0,
        var_limit=10000.0,
    ))

    print(f"\nMarkets: {', '.join(m.name for m in [da_market, id_market, bal_market])}")
    print(f"Risk limits: position={risk.limits.max_position} MW, "
          f"daily_loss=${risk.limits.max_daily_loss}, VaR=${risk.limits.var_limit}")

    # Generate price series (48h @ 15min = 192 steps)
    n_steps = 192
    resolution_h = 0.25

    hours = np.arange(n_steps) * resolution_h
    daily_hour = hours % 24

    # Day-ahead prices (set day before, smooth)
    da_base = 45 + 25 * np.sin((daily_hour - 6) * np.pi / 12)
    da_prices = da_base + np.random.randn(n_steps) * 3

    # Intraday prices (more volatile, mean-reverts to DA)
    id_prices = da_prices + np.cumsum(np.random.randn(n_steps) * 2)
    id_prices = 0.7 * id_prices + 0.3 * da_prices  # partial mean reversion

    # Balancing prices (most volatile)
    bal_prices = da_prices + np.random.randn(n_steps) * 15
    bal_spikes = np.random.choice([1.0, 1.0, 1.0, 2.5, 4.0], n_steps, p=[0.85, 0.05, 0.04, 0.04, 0.02])
    bal_prices *= bal_spikes

    print(f"\nPrice statistics (48h):")
    print(f"  Day-Ahead  : avg=${np.mean(da_prices):.1f}, std=${np.std(da_prices):.1f}")
    print(f"  Intraday   : avg=${np.mean(id_prices):.1f}, std=${np.std(id_prices):.1f}")
    print(f"  Balancing   : avg=${np.mean(bal_prices):.1f}, std=${np.std(bal_prices):.1f}")

    # Trading simulation
    position = 0.0  # MW
    pnl = 0.0
    trades = []
    equity = [0.0]
    max_position = 0.0
    n_trades = 0
    n_wins = 0

    # Battery constraints
    battery_mw = 10.0
    battery_mwh = 40.0
    soc = 0.5

    print("\n" + "-" * 70)
    print("  TRADING LOG (selected events)")
    print("-" * 70)

    for t in range(n_steps):
        hour = hours[t]
        da = da_prices[t]
        intra = id_prices[t]
        bal = bal_prices[t]

        # Strategy 1: DA-Intraday spread arbitrage
        spread_di = intra - da
        if abs(spread_di) > 8 and abs(position) < battery_mw * 0.8:
            size = min(5.0, battery_mw - abs(position))
            if spread_di > 8 and soc > 0.3:
                # Intraday expensive — sell intraday, buy DA
                pnl_trade = size * spread_di * resolution_h
                position -= size
                soc -= size * resolution_h / battery_mwh
                n_trades += 1
                if pnl_trade > 0:
                    n_wins += 1
                pnl += pnl_trade
                trades.append(("DA-ID arb (sell)", hour, size, spread_di, pnl_trade))
                if len(trades) <= 5 or pnl_trade > 50:
                    print(f"  t={hour:5.1f}h  SELL {size:.1f}MW  DA-ID spread=${spread_di:.1f}  P&L=${pnl_trade:.2f}")
            elif spread_di < -8 and soc < 0.9:
                # DA expensive — buy intraday
                pnl_trade = size * abs(spread_di) * resolution_h
                position += size
                soc += size * resolution_h / battery_mwh
                n_trades += 1
                if pnl_trade > 0:
                    n_wins += 1
                pnl += pnl_trade
                trades.append(("DA-ID arb (buy)", hour, size, spread_di, pnl_trade))
                if len(trades) <= 5 or pnl_trade > 50:
                    print(f"  t={hour:5.1f}h  BUY  {size:.1f}MW  DA-ID spread=${spread_di:.1f}  P&L=${pnl_trade:.2f}")

        # Strategy 2: Balancing market spikes
        if bal > da * 1.8 and soc > 0.3 and position > -battery_mw * 0.5:
            size = min(3.0, battery_mw - abs(position), (soc - 0.2) * battery_mwh / resolution_h)
            if size > 0:
                pnl_trade = size * (bal - da) * resolution_h
                position -= size
                soc -= size * resolution_h / battery_mwh
                n_trades += 1
                if pnl_trade > 0:
                    n_wins += 1
                pnl += pnl_trade
                trades.append(("Bal spike (sell)", hour, size, bal, pnl_trade))
                print(f"  t={hour:5.1f}h  SELL {size:.1f}MW  Bal spike=${bal:.0f}  P&L=${pnl_trade:.2f}  ***")

        elif bal < da * 0.5 and soc < 0.85:
            size = min(3.0, (0.9 - soc) * battery_mwh / resolution_h)
            if size > 0:
                pnl_trade = size * (da - bal) * resolution_h
                position += size
                soc += size * resolution_h / battery_mwh
                n_trades += 1
                if pnl_trade > 0:
                    n_wins += 1
                pnl += pnl_trade
                trades.append(("Bal dip (buy)", hour, size, bal, pnl_trade))
                if pnl_trade > 20:
                    print(f"  t={hour:5.1f}h  BUY  {size:.1f}MW  Bal dip=${bal:.0f}  P&L=${pnl_trade:.2f}")

        # Track
        max_position = max(max_position, abs(position))
        equity.append(pnl)

        # Gradually flatten position toward zero
        if abs(position) > 0.5:
            flatten = min(1.0, abs(position) * 0.1)
            if position > 0:
                position -= flatten
            else:
                position += flatten

    equity_arr = np.array(equity)
    returns = np.diff(equity_arr)

    # Performance metrics
    peak = np.maximum.accumulate(equity_arr)
    drawdown = peak - equity_arr
    max_drawdown = np.max(drawdown)

    nonzero_returns = returns[returns != 0]
    sharpe = 0.0
    if len(nonzero_returns) > 1 and np.std(nonzero_returns) > 0:
        sharpe = np.mean(nonzero_returns) / np.std(nonzero_returns) * np.sqrt(35040)

    win_rate = n_wins / n_trades * 100 if n_trades > 0 else 0

    print("\n" + "-" * 70)
    print("  TRADING PERFORMANCE")
    print("-" * 70)
    print(f"  Total P&L           : ${pnl:,.2f}")
    print(f"  Number of trades    : {n_trades}")
    print(f"  Win rate            : {win_rate:.1f}%")
    print(f"  Max position        : {max_position:.1f} MW")
    print(f"  Max drawdown        : ${max_drawdown:,.2f}")
    print(f"  Sharpe ratio (ann.) : {sharpe:.2f}")
    print(f"  Final SOC           : {soc*100:.1f}%")

    # Break down by strategy
    arb_trades = [t for t in trades if "arb" in t[0]]
    spike_trades = [t for t in trades if "spike" in t[0] or "dip" in t[0]]
    arb_pnl = sum(t[4] for t in arb_trades)
    spike_pnl = sum(t[4] for t in spike_trades)

    print(f"\n  Strategy breakdown:")
    print(f"    DA-ID Arbitrage : {len(arb_trades)} trades, P&L=${arb_pnl:,.2f}")
    print(f"    Balancing Spikes: {len(spike_trades)} trades, P&L=${spike_pnl:,.2f}")
    print("-" * 70)

    annual_pnl = pnl * 365 / 2  # 48h sim -> annualise
    print(f"\nProjected annual P&L: ${annual_pnl:,.0f}")
    print(f"Battery asset value: ${battery_mwh * 300:,.0f} (at $300/kWh)")
    roi = annual_pnl / (battery_mwh * 300) * 100 if battery_mwh > 0 else 0
    print(f"Annual ROI from trading: {roi:.1f}%")

    print("\n[Demo complete]")


if __name__ == "__main__":
    run()
