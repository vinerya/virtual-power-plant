"""RiskLimits.var_limit and concentration_limit must actually be enforced.

Before this fix, RiskManager.check_limits() checked position size, daily
loss, and drawdown, but silently ignored the var_limit and
concentration_limit fields on RiskLimits despite them being named "limits".
"""

from __future__ import annotations

from datetime import datetime

from vpp.trading.core import RiskLimits, RiskManager
from vpp.trading.portfolio import Portfolio, Trade


def test_check_limits_flags_concentration_breach():
    """A portfolio 100% concentrated in one market must breach a 30% limit."""
    portfolio = Portfolio(initial_cash=100_000.0)
    portfolio.add_trade(Trade(market="DA", side="buy", quantity=1000.0, price=50.0))

    risk_manager = RiskManager(limits=RiskLimits(concentration_limit=0.3))
    result = risk_manager.check_limits(portfolio, market_prices={"DA": 50.0})

    assert result["breach"] is True
    assert any("oncentration" in b for b in result["breaches"])


def test_check_limits_passes_when_diversified_within_limit():
    """Positions spread evenly across markets must not breach concentration."""
    portfolio = Portfolio(initial_cash=100_000.0)
    portfolio.add_trade(Trade(market="DA", side="buy", quantity=100.0, price=50.0))
    portfolio.add_trade(Trade(market="RT", side="buy", quantity=100.0, price=50.0))
    portfolio.add_trade(Trade(market="ANC", side="buy", quantity=100.0, price=50.0))

    risk_manager = RiskManager(limits=RiskLimits(concentration_limit=0.5))
    result = risk_manager.check_limits(
        portfolio, market_prices={"DA": 50.0, "RT": 50.0, "ANC": 50.0}
    )

    assert not any("oncentration" in b for b in result["breaches"])


def test_check_limits_flags_var_breach():
    """A volatile equity curve must breach a tight $ VaR limit."""
    portfolio = Portfolio(initial_cash=100_000.0)
    for equity in (100_000.0, 80_000.0, 100_000.0, 70_000.0, 100_000.0, 60_000.0):
        portfolio.equity_curve.append({"timestamp": datetime.now(), "equity": equity})

    risk_manager = RiskManager(limits=RiskLimits(var_limit=1.0))
    result = risk_manager.check_limits(portfolio)

    assert result["breach"] is True
    assert any("VaR" in b for b in result["breaches"])
    assert result["var_1d"] > 1.0


def test_check_limits_works_without_market_prices():
    """The real call site (TradingEngine._monitor_risk) passes no prices --
    must fall back to each position's average_price, not raise."""
    portfolio = Portfolio(initial_cash=100_000.0)
    portfolio.add_trade(Trade(market="DA", side="buy", quantity=10.0, price=50.0))

    risk_manager = RiskManager(limits=RiskLimits())
    result = risk_manager.check_limits(portfolio)

    assert isinstance(result["breach"], bool)
    assert "var_1d" in result
    assert "concentrations" in result
