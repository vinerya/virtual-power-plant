"""Tests for the simulated exchange and the backtesting engine."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from vpp.trading.backtest import (
    STRATEGY_SPECS,
    build_strategy,
    generate_synthetic_prices,
    run_backtest,
)
from vpp.trading.data import SimulatedDataProvider
from vpp.trading.markets import MarketStatus, RealTimeMarket
from vpp.trading.orders import (
    FillOrKillOrder,
    ImmediateOrCancelOrder,
    LimitOrder,
    MarketOrder,
    OrderStatus,
    StopOrder,
)
from vpp.trading.portfolio import PnLCalculator
from vpp.trading.strategies import TradingStrategy

T0 = datetime(2025, 3, 3, 12, 0)


def _exchange(base_volume=20.0, seed=11):
    from vpp.trading.simulation import SimulatedExchange

    market = RealTimeMarket("rt")
    market.status = MarketStatus.OPEN
    market.tick_size = 0.01
    market.lot_size = 0.1
    provider = SimulatedDataProvider(
        {
            "markets": ["rt"],
            "base_prices": {"rt": 50.0},
            "seed": seed,
            "volatility": 0.3,
            "mean_reversion": 3.0,
            "base_volume": base_volume,
        }
    )
    return SimulatedExchange([market], provider, start=T0), market


def _ask_levels(market):
    return sorted(market.order_book.asks.items())


class TestSimulatedExchange:
    def test_tick_builds_a_sane_book(self):
        ex, market = _exchange()
        snap = ex.snapshot("rt")
        assert snap.bid_price < snap.last_price < snap.ask_price
        for price in list(market.order_book.bids) + list(market.order_book.asks):
            assert round(price, 2) == pytest.approx(price)

    def test_market_order_sweeps_asks_and_consumes_liquidity(self):
        ex, market = _exchange()
        levels = _ask_levels(market)
        first_price, first_level = levels[0]
        qty = round(first_level.quantity + 0.5, 1)
        order = MarketOrder("rt", "buy", qty)
        fills = ex.submit(order, T0)
        assert order.status == OrderStatus.FILLED
        assert sum(f.quantity for f in fills) == pytest.approx(qty)
        assert fills[0].price == first_price and fills[0].liquidity == "taker"
        assert fills[-1].price > first_price
        assert fills[0].fee == pytest.approx(
            fills[0].quantity * (market.transaction_fee + market.market_fee)
        )
        assert first_price not in market.order_book.asks  # consumed
        assert ex.snapshot("rt").ask_price > first_price

    def test_market_order_remainder_is_cancelled(self):
        ex, market = _exchange(base_volume=1.0)
        total = market.order_book.get_total_volume("sell")
        order = MarketOrder("rt", "buy", round(total + 10, 1))
        fills = ex.submit(order, T0)
        assert sum(f.quantity for f in fills) == pytest.approx(total)
        assert order.status == OrderStatus.CANCELLED
        assert order.id not in ex.open_orders

    def test_passive_limit_rests_and_can_be_cancelled(self):
        ex, _ = _exchange()
        order = LimitOrder("rt", "buy", 5, 1.00)
        assert ex.submit(order, T0) == []
        assert order.status == OrderStatus.PENDING
        assert order.id in ex.open_orders
        cancelled = ex.cancel(order.id)
        assert cancelled is order and order.status == OrderStatus.CANCELLED
        assert ex.cancel(order.id) is None

    def test_resting_order_fills_when_price_moves(self):
        ex, _ = _exchange()
        last = ex.snapshot("rt").last_price
        order = LimitOrder("rt", "sell", 1, round(last * 2, 2))
        ex.submit(order, T0)
        assert order.status == OrderStatus.PENDING
        ex.provider.base_prices["rt"] = 50.0 * 5  # force a big move
        result = ex.tick(T0 + timedelta(minutes=5))
        assert order.status == OrderStatus.FILLED
        assert result.fills and result.fills[0].liquidity == "maker"
        assert result.fills[0].price >= order.price
        assert order in result.touched_orders
        assert order.id not in ex.open_orders

    def test_fok_and_ioc(self):
        ex, market = _exchange()
        price, level = _ask_levels(market)[0]
        available = level.quantity
        too_big = round(available + 1, 1)
        fok = FillOrKillOrder("rt", "buy", too_big, price)
        assert ex.submit(fok, T0) == []
        assert fok.status == OrderStatus.CANCELLED and fok.filled_quantity == 0
        ioc = ImmediateOrCancelOrder("rt", "buy", too_big, price)
        fills = ex.submit(ioc, T0)
        assert sum(f.quantity for f in fills) == pytest.approx(available)
        assert ioc.status == OrderStatus.CANCELLED
        assert ioc.filled_quantity == pytest.approx(available)

    def test_stop_order_triggers_on_touch(self):
        ex, _ = _exchange()
        ask = ex.snapshot("rt").ask_price
        stop = StopOrder("rt", "buy", 1, stop_price=round(ask * 1.5, 2))
        assert ex.submit(stop, T0) == []
        assert stop.id in ex.open_orders
        ex.provider.base_prices["rt"] = 50.0 * 4
        ex.tick(T0 + timedelta(minutes=5))
        assert stop.status == OrderStatus.FILLED

    def test_day_order_expires_on_date_rollover(self):
        ex, _ = _exchange()
        order = LimitOrder("rt", "buy", 1, 1.00, time_in_force="DAY")
        order.timestamp = T0
        ex.submit(order, T0)
        result = ex.tick(T0 + timedelta(days=1))
        assert order.status == OrderStatus.EXPIRED
        assert [u.order for u in result.expired] == [order]
        assert order.id not in ex.open_orders

    def test_validation_rejections(self):
        from vpp.trading.simulation import OrderRejected

        ex, _ = _exchange()
        with pytest.raises(OrderRejected) as exc:
            ex.submit(MarketOrder("nope", "buy", 1), T0)
        assert exc.value.code == "unknown_market"
        with pytest.raises(OrderRejected) as exc:
            ex.submit(LimitOrder("rt", "buy", 1, 45.123), T0)
        assert exc.value.code == "order_invalid"

    def test_restore_keeps_resting_orders_only(self):
        ex, _ = _exchange()
        resting = LimitOrder("rt", "buy", 1, 1.0)
        ex.restore(resting)
        ex.restore(MarketOrder("rt", "buy", 1))
        assert list(ex.open_orders) == [resting.id]


class _BuyThenSell(TradingStrategy):
    """Deterministic test strategy: buy on bar 0, sell on bar ``exit_bar``."""

    def __init__(self, exit_bar=2, qty=10.0):
        super().__init__("scripted", {})
        self.bar = 0
        self.exit_bar = exit_bar
        self.qty = qty

    def generate_signals(self, market_data, portfolio):
        signals = []
        if self.bar == 0:
            signals.append(
                {"action": "buy", "market": "m", "quantity": self.qty, "order_type": "market"}
            )
        elif self.bar == self.exit_bar:
            signals.append(
                {"action": "sell", "market": "m", "quantity": self.qty, "order_type": "market"}
            )
        self.bar += 1
        return signals


class TestBacktest:
    def test_hand_computed_round_trip(self):
        result = run_backtest(
            _BuyThenSell(),
            {"m": [10.0, 11.0, 12.0]},
            fee_per_unit=0.0,
            half_spread=0.0,
            initial_cash=1000.0,
        )
        assert result.num_trades == 2
        assert result.total_pnl == pytest.approx(20.0)
        assert result.realized_pnl == pytest.approx(20.0)
        assert result.final_equity == pytest.approx(1020.0)
        assert result.max_drawdown == 0.0
        assert result.win_rate == 1.0
        assert result.sharpe_ratio > 0
        assert result.final_positions == {}

    def test_fees_spread_and_drawdown(self):
        result = run_backtest(
            _BuyThenSell(exit_bar=5),
            {"m": [10.0, 5.0, 10.0]},
            fee_per_unit=0.1,
            half_spread=0.0,
            initial_cash=1000.0,
        )
        assert result.num_trades == 1
        assert result.fees == pytest.approx(1.0)
        # Equity: 999 -> 949 -> 999; drawdown from the 1000 starting peak.
        assert result.max_drawdown == pytest.approx((1000 - 949) / 1000)
        assert result.unrealized_pnl == pytest.approx(0.0)
        assert result.final_positions == {"m": 10.0}

        spread = run_backtest(
            _BuyThenSell(),
            {"m": [10.0, 10.0, 10.0]},
            fee_per_unit=0.0,
            half_spread=0.01,
            initial_cash=1000.0,
        )
        assert spread.total_pnl == pytest.approx(-(0.1 + 0.1) * 10)

    def test_input_validation(self):
        with pytest.raises(ValueError):
            run_backtest(_BuyThenSell(), {"a": [1, 2], "b": [1, 2, 3]})
        with pytest.raises(ValueError):
            run_backtest(_BuyThenSell(), {"m": [1.0, -2.0]})
        with pytest.raises(ValueError):
            run_backtest(_BuyThenSell(), {"m": [1.0]})
        with pytest.raises(ValueError):
            run_backtest(_BuyThenSell(), {})

    def test_synthetic_prices_are_seeded(self):
        a = generate_synthetic_prices(["day_ahead", "real_time"], periods=48, seed=5)
        b = generate_synthetic_prices(["day_ahead", "real_time"], periods=48, seed=5)
        c = generate_synthetic_prices(["day_ahead", "real_time"], periods=48, seed=6)
        assert a == b and a[1] != c[1]
        assert len(a[0]) == 48 and all(p > 0 for p in a[1]["day_ahead"])

    @pytest.mark.parametrize("name", sorted(STRATEGY_SPECS))
    def test_every_catalogued_strategy_backtests(self, name):
        ts, prices = generate_synthetic_prices(["day_ahead", "real_time"], periods=96, seed=1)
        result = run_backtest(build_strategy(name), prices, timestamps=ts)
        assert result.strategy in (name, "ml_trading")
        assert result.periods == 96
        assert 0.0 <= result.max_drawdown < 1.0
        assert result.total_pnl == pytest.approx(result.final_equity - result.initial_cash)
        assert result.total_pnl == pytest.approx(
            result.realized_pnl + result.unrealized_pnl - result.fees, abs=1e-6
        )
        for qty in result.final_positions.values():
            assert abs(qty) <= STRATEGY_SPECS[name].parameters["max_position_size"] + 1e-9

    def test_build_strategy_rejects_unknown_params(self):
        with pytest.raises(ValueError):
            build_strategy("momentum", {"nope": 1})
        with pytest.raises(KeyError):
            build_strategy("does_not_exist")
        assert build_strategy("momentum", {"lookback_hours": 6}).lookback_hours == 6


def test_expected_shortfall_includes_var_observation():
    assert PnLCalculator.calculate_expected_shortfall([-0.05] + [0.01] * 19) == pytest.approx(0.05)
