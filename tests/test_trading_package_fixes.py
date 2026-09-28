"""Regression tests for bugs fixed in the ``vpp.trading`` package."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from vpp.trading.core import RiskLimits, RiskManager, TradingEngine
from vpp.trading.data import MarketData, SimulatedDataProvider
from vpp.trading.markets import MarketStatus, RealTimeMarket
from vpp.trading.orders import (
    FillOrKillOrder,
    LimitOrder,
    MarketOrder,
    OrderBook,
    OrderStatus,
    StopLimitOrder,
    validate_order_parameters,
)
from vpp.trading.portfolio import (
    PnLCalculator,
    Portfolio,
    Position,
    Trade,
    parametric_var,
)
from vpp.trading.strategies import (
    ArbitrageStrategy,
    MLTradingStrategy,
    MomentumStrategy,
)


def _trade(market="DA", side="buy", qty=10.0, price=50.0, fees=0.0, ts=None):
    return Trade(
        market=market,
        side=side,
        quantity=qty,
        price=price,
        fees=fees,
        timestamp=ts or datetime.now(),
    )


# ---------------------------------------------------------------------------
# Portfolio / positions
# ---------------------------------------------------------------------------


class TestPortfolioAccounting:
    def test_equity_is_cash_plus_market_value(self):
        """Buying at the market must not change equity (old code halved it)."""
        p = Portfolio(initial_cash=1000.0)
        p.add_trade(_trade(qty=10, price=50))
        assert p.cash == pytest.approx(500.0)
        assert p.get_equity({"DA": 50.0}) == pytest.approx(1000.0)
        assert p.get_equity({"DA": 60.0}) == pytest.approx(1100.0)

    def test_short_equity_marks_correctly(self):
        p = Portfolio(initial_cash=1000.0)
        p.add_trade(_trade(side="sell", qty=10, price=50))
        assert p.get_equity({"DA": 40.0}) == pytest.approx(1100.0)

    def test_partial_short_cover_realizes_pnl_and_keeps_average(self):
        pos = Position(market="DA")
        pos.update_from_trade(_trade(side="sell", qty=10, price=50))
        pos.update_from_trade(_trade(side="buy", qty=4, price=40))
        assert pos.quantity == pytest.approx(-6)
        assert pos.average_price == pytest.approx(50.0)  # was re-averaged to 46.67
        assert pos.realized_pnl == pytest.approx(40.0)

    def test_trade_realized_pnl_is_attributed(self):
        p = Portfolio(initial_cash=10_000.0)
        opening = _trade(qty=10, price=50)
        closing = _trade(side="sell", qty=10, price=55, fees=1.0)
        p.add_trade(opening)
        p.add_trade(closing)
        assert opening.realized_pnl == 0.0
        assert closing.realized_pnl == pytest.approx(50.0)
        assert p.total_fees == pytest.approx(1.0)

    def test_daily_pnl_counts_buy_to_cover_net_of_fees(self):
        p = Portfolio(initial_cash=10_000.0)
        p.add_trade(_trade(side="sell", qty=10, price=50))
        p.add_trade(_trade(side="buy", qty=10, price=60, fees=2.0))
        # Short covered at a loss of 100 plus 2 fees. Old code only summed sells -> 0.
        assert p.calculate_daily_pnl() == pytest.approx(-102.0)

    def test_daily_pnl_filters_by_day(self):
        p = Portfolio(initial_cash=10_000.0)
        yesterday = datetime.now() - timedelta(days=1)
        p.add_trade(_trade(qty=10, price=50, ts=yesterday))
        p.add_trade(_trade(side="sell", qty=10, price=40, ts=yesterday))
        assert p.calculate_daily_pnl() == 0.0
        assert p.calculate_daily_pnl(yesterday) == pytest.approx(-100.0)

    def test_realized_and_unrealized_totals(self):
        p = Portfolio(initial_cash=10_000.0)
        p.add_trade(_trade(qty=10, price=50))
        p.add_trade(_trade(side="sell", qty=4, price=55))
        assert p.calculate_realized_pnl() == pytest.approx(20.0)
        assert p.calculate_unrealized_pnl({"DA": 60.0}) == pytest.approx(60.0)

    def test_current_drawdown_from_peak(self):
        p = Portfolio(initial_cash=1000.0)
        p.add_trade(_trade(qty=10, price=50))
        p.update_equity_curve({"DA": 60.0})  # peak 1100
        assert p.current_drawdown({"DA": 49.0}) == pytest.approx((1100 - 990) / 1100)

    def test_equity_curve_is_bounded(self):
        p = Portfolio(initial_cash=1000.0)
        for _ in range(20):
            p.update_equity_curve({}, max_points=5)
        assert len(p.equity_curve) == 5


class TestVaR:
    def test_historical_var_is_zero_without_losses(self):
        """abs() used to report a gain quantile as a loss."""
        assert PnLCalculator.calculate_var([0.01, 0.02, 0.03, 0.04]) == 0.0
        assert PnLCalculator.calculate_var([-0.05] + [0.01] * 19) == pytest.approx(0.05)

    def test_parametric_var_single_position(self):
        var = parametric_var({"DA": 1000.0}, {"DA": 0.1})
        assert var == pytest.approx(164.485, rel=1e-4)

    def test_parametric_var_scales_with_horizon_and_confidence(self):
        v1 = parametric_var({"DA": 1000.0}, {"DA": 0.1}, horizon_days=4)
        assert v1 == pytest.approx(2 * 164.485, rel=1e-4)
        v99 = parametric_var({"DA": 1000.0}, {"DA": 0.1}, confidence=0.99)
        assert v99 == pytest.approx(232.63, rel=1e-3)

    def test_parametric_var_hedge_and_correlation(self):
        hedged = parametric_var(
            {"A": 1000.0, "B": -1000.0},
            {"A": 0.1, "B": 0.1},
            correlations={"A": {"B": 1.0}, "B": {"A": 1.0}},
        )
        assert hedged == pytest.approx(0.0, abs=1e-9)
        independent = parametric_var(
            {"A": 1000.0, "B": 1000.0}, {"A": 0.1, "B": 0.1}, default_correlation=0.0
        )
        assert independent == pytest.approx(164.485 * 2**0.5, rel=1e-4)

    def test_parametric_var_rejects_unknown_confidence(self):
        with pytest.raises(ValueError):
            parametric_var({"A": 1.0}, {"A": 0.1}, confidence=0.93)

    def test_parametric_var_empty(self):
        assert parametric_var({}, {}) == 0.0


# ---------------------------------------------------------------------------
# Risk manager
# ---------------------------------------------------------------------------


class TestEvaluateOrder:
    def test_new_position_over_limit_is_rejected(self):
        """Old check_order_risk skipped the limit when no position existed."""
        rm = RiskManager(RiskLimits(max_position=50))
        order = MarketOrder("DA", "buy", 60)
        assert not rm.check_order_risk(order, Portfolio(initial_cash=1e6))
        reasons = rm.evaluate_order(order, Portfolio(initial_cash=1e6), reference_price=50)
        assert any("Position limit" in r for r in reasons)

    def test_resting_orders_count_toward_position_limit(self):
        rm = RiskManager(RiskLimits(max_position=50))
        order = LimitOrder("DA", "buy", 30, 40)
        p = Portfolio(initial_cash=1e6)
        assert rm.evaluate_order(order, p, reference_price=45) == []
        reasons = rm.evaluate_order(order, p, reference_price=45, open_order_quantity=30)
        assert any("resting orders" in r for r in reasons)

    def test_risk_reducing_order_always_allowed(self):
        rm = RiskManager(RiskLimits(max_position=5, max_daily_loss=1, var_limit=0.0))
        p = Portfolio(initial_cash=1e6)
        p.add_trade(_trade(qty=10, price=50))
        p.add_trade(_trade(side="sell", qty=2, price=10))  # big realized loss today
        assert (
            rm.evaluate_order(
                MarketOrder("DA", "sell", 5), p, {"DA": 50}, 50, daily_volatility={"DA": 0.5}
            )
            == []
        )
        assert rm.evaluate_order(
            MarketOrder("DA", "buy", 1), p, {"DA": 50}, 50, daily_volatility={"DA": 0.5}
        )

    def test_daily_loss_blocks_new_risk(self):
        rm = RiskManager(RiskLimits(max_daily_loss=50))
        p = Portfolio(initial_cash=1e6)
        p.add_trade(_trade(qty=10, price=50))
        p.add_trade(_trade(side="sell", qty=10, price=40))
        reasons = rm.evaluate_order(MarketOrder("DA", "buy", 1), p, {"DA": 40}, 40)
        assert any("Daily loss" in r for r in reasons)

    def test_var_limit(self):
        rm = RiskManager(RiskLimits(var_limit=100))
        p = Portfolio(initial_cash=1e6)
        reasons = rm.evaluate_order(
            MarketOrder("DA", "buy", 10), p, {"DA": 50}, 50, daily_volatility={"DA": 0.5}
        )
        assert any("VaR" in r for r in reasons)  # 1.645 * 500 * 0.5 = 411
        assert (
            rm.evaluate_order(
                MarketOrder("DA", "buy", 1), p, {"DA": 50}, 50, daily_volatility={"DA": 0.5}
            )
            == []
        )

    def test_concentration_needs_two_markets(self):
        rm = RiskManager(RiskLimits(concentration_limit=0.6))
        p = Portfolio(initial_cash=1e6)
        # Single market: trivially 100% concentrated -- not a breach.
        assert rm.evaluate_order(MarketOrder("DA", "buy", 10), p, {"DA": 50}, 50) == []
        p.add_trade(_trade(market="RT", qty=2, price=50))
        reasons = rm.evaluate_order(MarketOrder("DA", "buy", 10), p, {"DA": 50, "RT": 50}, 50)
        assert any("Concentration" in r for r in reasons)

    def test_drawdown_blocks_new_risk(self):
        rm = RiskManager(RiskLimits(max_drawdown=0.05))
        p = Portfolio(initial_cash=1000.0)
        p.add_trade(_trade(qty=10, price=50))
        p.update_equity_curve({"DA": 50.0})
        reasons = rm.evaluate_order(MarketOrder("DA", "buy", 1), p, {"DA": 40.0}, 40.0)
        assert any("Drawdown" in r for r in reasons)

    def test_assess_portfolio(self):
        rm = RiskManager(RiskLimits(max_position=5, var_limit=10))
        p = Portfolio(initial_cash=1e6)
        p.add_trade(_trade(qty=10, price=50))
        risk = rm.assess_portfolio(p, {"DA": 50.0}, {"DA": 0.2})
        assert risk["breach"] is True
        assert risk["var_95_1d"] == pytest.approx(1.6448536 * 500 * 0.2, rel=1e-6)
        assert risk["gross_exposure"] == pytest.approx(500.0)
        assert risk["concentrations"] == {"DA": 1.0}
        assert not any("Concentration" in b for b in risk["breaches"])


# ---------------------------------------------------------------------------
# Orders / order book / markets
# ---------------------------------------------------------------------------


def _book_with_asks(levels):
    book = OrderBook("RT")
    for price, qty in levels:
        book.add_order(LimitOrder("RT", "sell", qty, price))
    return book


class TestOrderBook:
    def test_level_quantity_tracks_fills(self):
        book = _book_with_asks([(50.0, 10.0), (50.0, 10.0)])
        book.match_order(MarketOrder("RT", "buy", 15))
        assert book.get_volume_at_price(50.0, "sell") == pytest.approx(5.0)
        assert book.asks[50.0].order_count == 1

    def test_filled_level_is_removed(self):
        book = _book_with_asks([(50.0, 10.0), (51.0, 10.0)])
        book.match_order(MarketOrder("RT", "buy", 10))
        assert 50.0 not in book.asks
        assert book.get_best_ask() == 51.0

    def test_fok_is_all_or_nothing_without_side_effects(self):
        book = _book_with_asks([(50.0, 5.0), (51.0, 5.0)])
        fok = FillOrKillOrder("RT", "buy", 20, 60.0)
        assert book.match_order(fok) == []
        assert fok.filled_quantity == 0
        assert book.get_total_volume("sell") == pytest.approx(10.0)
        fok2 = FillOrKillOrder("RT", "buy", 8, 60.0)
        matches = book.match_order(fok2)
        assert sum(m["quantity"] for m in matches) == pytest.approx(8.0)
        assert fok2.status == OrderStatus.FILLED

    def test_stop_limit_fills_on_triggering_price(self):
        order = StopLimitOrder("RT", "buy", 1, stop_price=50.0, limit_price=52.0)
        assert order.is_executable(51.0) is True
        order2 = StopLimitOrder("RT", "buy", 1, stop_price=50.0, limit_price=52.0)
        assert order2.is_executable(49.0) is False
        assert order2.is_executable(53.0) is False  # triggered, but above limit

    def test_validate_stop_limit_parameters(self):
        assert (
            validate_order_parameters("stop_limit", "RT", "buy", 1, stop_price=50, limit_price=52)
            == []
        )
        assert validate_order_parameters("stop_limit", "RT", "buy", 1)


class TestMarkets:
    def _market(self):
        m = RealTimeMarket("RT")
        m.status = MarketStatus.OPEN
        return m

    def test_open_without_sessions(self):
        m = RealTimeMarket("RT")
        assert not m.is_market_open()
        m.status = MarketStatus.OPEN
        assert m.is_market_open()

    def test_tick_and_lot_validation_is_float_safe(self):
        m = self._market()
        assert m.validate_order(LimitOrder("RT", "buy", 3, 0.12)) == []  # 0.12 % 0.01 != 0
        errors = m.validate_order(LimitOrder("RT", "buy", 3, 0.123))
        assert any("tick size" in e for e in errors)
        errors = m.validate_order(LimitOrder("RT", "buy", 2.5, 0.12))
        assert any("lot size" in e for e in errors)

    def test_partial_match_rests_remainder(self):
        m = self._market()
        m.order_book.add_order(LimitOrder("RT", "sell", 4, 50.0))
        order = LimitOrder("RT", "buy", 10, 51.0)
        result = m.execute_order(order)
        assert result["quantity"] == pytest.approx(4)
        assert result["status"] == "partial"
        assert m.order_book.get_best_bid() == 51.0  # remainder rested
        assert result["fees"] == pytest.approx(4 * (m.transaction_fee + m.market_fee))

    def test_market_order_without_liquidity_is_cancelled_not_rested(self):
        m = self._market()
        order = MarketOrder("RT", "buy", 5)
        result = m.execute_order(order)
        assert result["status"] == "cancelled"
        assert order.status == OrderStatus.CANCELLED
        assert m.order_book.get_best_bid() is None


class TestEngine:
    def test_market_orders_pass_validation(self):
        engine = TradingEngine()
        engine.add_market(RealTimeMarket("RT"))
        assert engine._validate_order(MarketOrder("RT", "buy", 1))

    def test_zero_fill_does_not_book_trade_or_mark_filled(self):
        engine = TradingEngine()
        market = RealTimeMarket("RT")
        market.status = MarketStatus.OPEN
        engine.add_market(market)
        order = LimitOrder("RT", "buy", 5, 40.0)
        engine._execute_order(order)
        assert order.status == OrderStatus.PENDING
        assert engine.portfolio_manager.trades == []

    def test_fill_books_trade(self):
        engine = TradingEngine()
        market = RealTimeMarket("RT")
        market.status = MarketStatus.OPEN
        market.order_book.add_order(LimitOrder("RT", "sell", 5, 40.0))
        engine.add_market(market)
        order = MarketOrder("RT", "buy", 5)
        engine._execute_order(order)
        assert order.status == OrderStatus.FILLED
        assert len(engine.portfolio_manager.trades) == 1
        assert engine.get_portfolio().positions["RT"].quantity == 5


# ---------------------------------------------------------------------------
# Data provider / strategies
# ---------------------------------------------------------------------------


class TestSimulatedDataProvider:
    def test_prices_stay_bounded(self):
        """The seasonal factor used to compound every step (x1.2^60/hour)."""
        provider = SimulatedDataProvider(
            {"markets": ["DA"], "base_prices": {"DA": 50.0}, "seed": 1, "mean_reversion": 0.0}
        )
        provider.connect()
        start = datetime(2025, 6, 1, 12)
        prices = [
            provider.generate("DA", start + timedelta(minutes=i)).last_price for i in range(2000)
        ]
        assert min(prices) > 5.0 and max(prices) < 500.0

    def test_seeded_and_timestamped_generation_is_reproducible(self):
        def run():
            p = SimulatedDataProvider({"markets": ["DA"], "seed": 3})
            p.connect()
            t0 = datetime(2025, 1, 1)
            return [p.generate("DA", t0 + timedelta(hours=i)).last_price for i in range(10)]

        assert run() == run()

    def test_seasonal_shape(self):
        p = SimulatedDataProvider({"seasonal_amplitude": 0.2})
        assert p.seasonal_factor(datetime(2025, 1, 1, 12)) == pytest.approx(1.2)
        assert p.seasonal_factor(datetime(2025, 1, 1, 0)) == pytest.approx(0.8)


def _md(market, price, ts):
    return MarketData(market=market, timestamp=ts, last_price=price)


class TestStrategies:
    def test_arbitrage_emits_each_opportunity_once(self):
        strat = ArbitrageStrategy(price_threshold=1.0, transaction_cost=0.0)
        ts = datetime(2025, 1, 1)
        signals = strat.generate_signals(
            {"day_ahead": _md("day_ahead", 40.0, ts), "real_time": _md("real_time", 50.0, ts)},
            Portfolio(initial_cash=1e6),
        )
        assert len(signals) == 2
        assert {s["action"] for s in signals} == {"buy", "sell"}
        assert all(s["timestamp"] == ts for s in signals)

    def test_momentum_uses_data_timestamps(self):
        """With datetime.now() replayed bars were all 'simultaneous'; the 4h
        lookback then spanned the whole replay."""
        strat = MomentumStrategy(lookback_hours=2, momentum_threshold=0.05)
        p = Portfolio(initial_cash=1e6)
        t0 = datetime(2025, 1, 1)
        prices = [100.0, 50.0, 50.0, 50.0, 50.0, 51.0]  # crash long ago, flat recently
        signals = []
        for i, price in enumerate(prices):
            signals = strat.generate_signals({"DA": _md("DA", price, t0 + timedelta(hours=i))}, p)
        assert signals == []

    def test_ml_strategy_does_not_claim_a_model(self):
        strat = MLTradingStrategy(model_path="missing.pkl")
        assert strat.model is None
        assert strat._load_model() is False


def test_single_market_data_type_shared_by_exchange_and_markets():
    """``trading.markets.MarketData`` is ``trading.data.MarketData`` (no bridge cast)."""
    from vpp.trading import markets as markets_mod
    from vpp.trading.markets import DayAheadMarket
    from vpp.trading.simulation import SimulatedExchange

    assert markets_mod.MarketData is MarketData

    market = DayAheadMarket()
    exchange = SimulatedExchange(
        [market], SimulatedDataProvider(config={"seed": 1}), start=datetime(2025, 1, 1)
    )
    snapshot = exchange.snapshot(market.name)
    assert isinstance(market.current_data, MarketData)
    assert market.current_data is snapshot
