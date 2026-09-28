"""Bar-by-bar strategy backtesting over supplied or synthetic price series.

Execution model (deliberately simple and stated up front):

* Each bar exposes ``last_price`` with a symmetric bid/ask of
  ``last * (1 -/+ half_spread)``.
* Market-order signals fill in full at the ask (buy) / bid (sell).
* Limit-order signals fill in full at the limit price if the bar's last
  price is at or through the limit (buy: last <= limit, sell: last >= limit);
  otherwise they are dropped -- nothing rests between bars.
* Every fill pays ``fee_per_unit`` per unit of quantity.
* Positions are marked to each bar's last price. Sharpe is annualised from
  per-bar equity returns using the bar interval; drawdown is peak-to-trough
  on the marked equity curve.

There is no market impact, queue position, or liquidity cap beyond the
strategy's own ``max_position_size``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

import numpy as np

from .data import MarketData, SimulatedDataProvider
from .portfolio import Portfolio, Trade
from .strategies import (
    ArbitrageStrategy,
    MeanReversionStrategy,
    MLTradingStrategy,
    MomentumStrategy,
    TradingStrategy,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


@dataclass(frozen=True)
class StrategySpec:
    """Catalogue entry describing a strategy exposed through the API."""

    name: str
    description: str
    factory: Callable[..., TradingStrategy]
    parameters: dict[str, Any]
    min_markets: int = 1


_COMMON = {"base_quantity": 10.0, "max_position_size": 50.0, "max_daily_trades": 24}

STRATEGY_SPECS: dict[str, StrategySpec] = {
    "arbitrage": StrategySpec(
        name="arbitrage",
        description=(
            "Cross-market spread capture: buys the cheaper and sells the dearer "
            "market when the price gap exceeds 2x transaction cost plus a threshold."
        ),
        factory=ArbitrageStrategy,
        parameters={"price_threshold": 2.0, "transaction_cost": 0.07,
                    "markets": ["day_ahead", "real_time"], **_COMMON},
        min_markets=2,
    ),
    "momentum": StrategySpec(
        name="momentum",
        description=(
            "Trend following: buys when the price rose more than momentum_threshold "
            "over the lookback window, sells when it fell."
        ),
        factory=MomentumStrategy,
        parameters={"lookback_hours": 4, "momentum_threshold": 0.05, **_COMMON},
    ),
    "mean_reversion": StrategySpec(
        name="mean_reversion",
        description=(
            "Z-score reversion: sells when price is deviation_threshold standard "
            "deviations above its lookback mean, buys when below (limit at the mean)."
        ),
        factory=MeanReversionStrategy,
        parameters={"lookback_hours": 24, "deviation_threshold": 2.0, **_COMMON},
    ),
    "ml": StrategySpec(
        name="ml",
        description=(
            "Heuristic placeholder for an ML strategy: no trained model is loaded; "
            "trades real_time when it deviates >5% from day_ahead."
        ),
        factory=MLTradingStrategy,
        parameters={"prediction_threshold": 0.6, **_COMMON},
        min_markets=2,
    ),
}


def build_strategy(name: str, params: dict[str, Any] | None = None) -> TradingStrategy:
    """Instantiate a catalogued strategy, validating parameter names."""
    spec = STRATEGY_SPECS.get(name)
    if spec is None:
        raise KeyError(name)
    params = dict(params or {})
    unknown = sorted(set(params) - set(spec.parameters))
    if unknown:
        raise ValueError(f"Unknown parameter(s) for {name}: {unknown}")
    merged = {**spec.parameters, **params}
    return spec.factory(**merged)


def generate_synthetic_prices(
    markets: Sequence[str],
    *,
    start: datetime | None = None,
    periods: int = 168,
    interval_minutes: float = 60.0,
    seed: int = 42,
    base_prices: dict[str, float] | None = None,
    volatility: float = 0.35,
    mean_reversion: float = 3.0,
    seasonal_amplitude: float = 0.25,
) -> tuple[list[datetime], dict[str, list[float]]]:
    """Seeded synthetic price paths using :class:`SimulatedDataProvider`."""
    start = start or datetime(2025, 1, 1)
    provider = SimulatedDataProvider({
        "markets": list(markets),
        "base_prices": base_prices or {"day_ahead": 45.0, "real_time": 50.0},
        "volatility": volatility,
        "mean_reversion": mean_reversion,
        "seasonal_amplitude": seasonal_amplitude,
        "seed": seed,
    })
    provider.connect()
    timestamps = [start + timedelta(minutes=interval_minutes * i) for i in range(periods)]
    prices: dict[str, list[float]] = {m: [] for m in markets}
    for ts in timestamps:
        for market in markets:
            data = provider.generate(market, ts)
            prices[market].append(round(float(data.last_price), 4))
    return timestamps, prices


@dataclass
class BacktestResult:
    strategy: str
    markets: list[str]
    periods: int
    interval_minutes: float
    initial_cash: float
    final_equity: float
    total_pnl: float
    total_return: float
    realized_pnl: float
    unrealized_pnl: float
    fees: float
    sharpe_ratio: float
    max_drawdown: float
    num_trades: int
    win_rate: float | None
    final_positions: dict[str, float]
    equity_curve: list[dict[str, Any]] = field(default_factory=list)
    trades: list[dict[str, Any]] = field(default_factory=list)


def _max_drawdown(equity: Sequence[float], initial: float) -> float:
    peak = initial
    worst = 0.0
    for value in equity:
        peak = max(peak, value)
        if peak > 0:
            worst = max(worst, (peak - value) / peak)
    return worst


def _sharpe(equity: Sequence[float], initial: float, interval_minutes: float) -> float:
    series = np.array([initial, *equity], dtype=float)
    if len(series) < 3 or np.any(series[:-1] <= 0):
        return 0.0
    returns = np.diff(series) / series[:-1]
    std = float(np.std(returns, ddof=1))
    if std == 0 or not math.isfinite(std):
        return 0.0
    periods_per_year = 365.0 * 24.0 * 60.0 / interval_minutes
    return float(np.mean(returns) / std * math.sqrt(periods_per_year))


def run_backtest(
    strategy: TradingStrategy,
    prices: dict[str, Sequence[float]],
    *,
    timestamps: Sequence[datetime] | None = None,
    interval_minutes: float = 60.0,
    initial_cash: float = 100_000.0,
    fee_per_unit: float = 0.07,
    half_spread: float = 0.001,
    equity_curve_points: int = 500,
) -> BacktestResult:
    """Replay *prices* bar by bar through *strategy*.

    Args:
        prices: per-market price series; all series must share one length.
        timestamps: bar timestamps; generated from ``interval_minutes``
            (starting 2025-01-01) when omitted.
    """
    if not prices:
        raise ValueError("At least one price series is required")
    lengths = {len(series) for series in prices.values()}
    if len(lengths) != 1:
        raise ValueError("All price series must have the same length")
    periods = lengths.pop()
    if periods < 2:
        raise ValueError("At least two bars are required")
    if timestamps is None:
        start = datetime(2025, 1, 1)
        timestamps = [start + timedelta(minutes=interval_minutes * i) for i in range(periods)]
    elif len(timestamps) != periods:
        raise ValueError("timestamps must match the price series length")
    for market, series in prices.items():
        if any((not math.isfinite(p)) or p <= 0 for p in series):
            raise ValueError(f"Prices for {market} must be positive and finite")

    portfolio = Portfolio(initial_cash=initial_cash)
    trade_log: list[dict[str, Any]] = []
    equity: list[float] = []

    for i, ts in enumerate(timestamps):
        bar = {m: float(series[i]) for m, series in prices.items()}
        market_data = {
            m: MarketData(
                market=m, timestamp=ts, last_price=p,
                bid_price=p * (1 - half_spread), ask_price=p * (1 + half_spread),
                source="backtest",
            )
            for m, p in bar.items()
        }
        for signal in strategy.generate_signals(market_data, portfolio):
            market = signal.get("market")
            if market not in bar:
                continue
            side = signal.get("action")
            quantity = float(signal.get("quantity") or 0.0)
            if side not in ("buy", "sell") or quantity <= 0:
                continue
            last = bar[market]
            if signal.get("order_type") == "market":
                fill_price = market_data[market].ask_price if side == "buy" else market_data[market].bid_price
            else:
                limit = float(signal.get("price") or 0.0)
                marketable = last <= limit if side == "buy" else last >= limit
                if not marketable:
                    continue
                fill_price = limit
            trade = Trade(
                order_id=f"bt-{i}", market=market, side=side, quantity=quantity,
                price=fill_price, timestamp=ts, fees=quantity * fee_per_unit,
                strategy=strategy.name,
            )
            portfolio.add_trade(trade)
            strategy.trades_executed += 1
            trade_log.append({
                "timestamp": ts, "market": market, "side": side,
                "quantity": quantity, "price": fill_price,
                "fees": trade.fees, "realized_pnl": trade.realized_pnl,
            })
        portfolio.update_equity_curve(bar, timestamp=ts)
        equity.append(portfolio.get_equity(bar))

    last_bar = {m: float(series[-1]) for m, series in prices.items()}
    final_equity = equity[-1]
    closing = [t for t in trade_log if abs(t["realized_pnl"]) > 1e-12]
    win_rate = (
        sum(1 for t in closing if t["realized_pnl"] - t["fees"] > 0) / len(closing)
        if closing else None
    )
    step = max(1, len(equity) // max(1, equity_curve_points))
    curve = [
        {"timestamp": timestamps[i], "equity": equity[i]}
        for i in range(0, len(equity), step)
    ]
    if curve[-1]["timestamp"] != timestamps[-1]:
        curve.append({"timestamp": timestamps[-1], "equity": equity[-1]})

    strategy.total_pnl = final_equity - initial_cash
    return BacktestResult(
        strategy=strategy.name,
        markets=list(prices),
        periods=periods,
        interval_minutes=interval_minutes,
        initial_cash=initial_cash,
        final_equity=final_equity,
        total_pnl=final_equity - initial_cash,
        total_return=(final_equity - initial_cash) / initial_cash if initial_cash else 0.0,
        realized_pnl=portfolio.calculate_realized_pnl(),
        unrealized_pnl=portfolio.calculate_unrealized_pnl(last_bar),
        fees=portfolio.total_fees,
        sharpe_ratio=_sharpe(equity, initial_cash, interval_minutes),
        max_drawdown=_max_drawdown(equity, initial_cash),
        num_trades=len(trade_log),
        win_rate=win_rate,
        final_positions={m: p.quantity for m, p in portfolio.positions.items() if p.quantity},
        equity_curve=curve,
        trades=trade_log,
    )
