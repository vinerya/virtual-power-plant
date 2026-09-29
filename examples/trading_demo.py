"""
Comprehensive VPP Trading System Demonstration

This example showcases the complete trading capabilities including:
- Multi-market trading (day-ahead, real-time, ancillary services)
- Advanced trading strategies (arbitrage, momentum, ML)
- Portfolio management and risk control
- Real-time market data integration
- Performance monitoring and analytics
"""

import sys
import time
from datetime import datetime, timedelta

# Import VPP trading components
from vpp.trading import (
    calculate_arbitrage_opportunity,
    create_arbitrage_strategy,
    create_ml_strategy,
    create_trading_engine,
    optimize_trading_schedule,
)
from vpp.trading.markets import MarketStatus, create_market
from vpp.trading.orders import LimitOrder, create_order
from vpp.trading.portfolio import Portfolio
from vpp.trading.strategies import create_strategy


def demonstrate_arbitrage_trading():
    """Demonstrate arbitrage trading between markets."""
    print("\n" + "=" * 60)
    print("🔄 ARBITRAGE TRADING DEMONSTRATION")
    print("=" * 60)

    # Calculate arbitrage opportunity
    day_ahead_price = 0.08  # $/kWh
    real_time_price = 0.12  # $/kWh
    transaction_cost = 0.002  # $/kWh

    opportunity = calculate_arbitrage_opportunity(
        day_ahead_price, real_time_price, transaction_cost
    )

    print("📊 Arbitrage Analysis:")
    print(f"   Day-ahead price: ${day_ahead_price:.3f}/kWh")
    print(f"   Real-time price: ${real_time_price:.3f}/kWh")
    print(f"   Price difference: ${opportunity['price_difference']:.3f}/kWh")
    print(f"   Transaction cost: ${opportunity['transaction_cost']:.3f}/kWh")
    print(f"   Profit per MWh: ${opportunity['profit_per_mwh']:.2f}")
    print(f"   Profit margin: {opportunity['profit_margin']:.1f}%")
    print(f"   Profitable: {'✅ YES' if opportunity['profitable'] else '❌ NO'}")

    if opportunity["profitable"]:
        # ``market1``/``market2`` are the first/second price arguments.
        names = {"market1": "day-ahead", "market2": "real-time"}
        print(
            f"   Strategy: Buy in {names[opportunity['buy_market']]}, "
            f"Sell in {names[opportunity['sell_market']]}"
        )

    return opportunity


def demonstrate_trading_schedule_optimization():
    """Demonstrate trading schedule optimization."""
    print("\n" + "=" * 60)
    print("📈 TRADING SCHEDULE OPTIMIZATION")
    print("=" * 60)

    # 24-hour price forecast with peak and off-peak periods
    prices = [
        *[0.06, 0.05, 0.04, 0.04, 0.05, 0.06],  # off-peak (midnight to 6 AM)
        *[0.08, 0.12, 0.15],  # morning ramp (6 AM to 9 AM)
        *[0.18, 0.20, 0.22, 0.20, 0.18, 0.16, 0.18, 0.20, 0.22],  # peak (9 AM to 6 PM)
        *[0.15, 0.12, 0.10],  # evening ramp (6 PM to 9 PM)
        *[0.08, 0.07, 0.06],  # off-peak (9 PM to midnight)
    ]

    # Battery specifications
    battery_capacity = 1000.0  # kWh
    efficiency = 0.92  # 92% round-trip efficiency

    print("🔋 Battery Specifications:")
    print(f"   Capacity: {battery_capacity:.0f} kWh")
    print(f"   Efficiency: {efficiency:.1%}")
    print(f"   Max Power: {battery_capacity * 0.25:.0f} kW (4-hour rate)")

    # Optimize trading schedule
    schedule = optimize_trading_schedule(prices, battery_capacity, efficiency)

    print("\n📊 Optimization Results:")
    print(f"   Success: {'✅ YES' if schedule['success'] else '❌ NO'}")

    if schedule["success"]:
        print(f"   Expected profit: ${schedule['total_profit']:.2f}")
        print(f"   Profit per MWh: ${schedule['profit_per_mwh']:.2f}")
        print(f"   Battery utilization: {schedule['utilization']:.1f}%")
        print(f"   Solve time: {schedule['solve_time']:.3f}s")

        # Show hourly schedule
        print("\n⏰ Hourly Trading Schedule:")
        battery_schedule = schedule["battery_schedule"]
        for hour, (price, power) in enumerate(zip(prices, battery_schedule, strict=True)):
            action = "CHARGE" if power > 0.05 else "DISCHARGE" if power < -0.05 else "IDLE"
            print(f"   Hour {hour:2d}: ${price:.3f}/kWh | {power:6.1f} kW | {action}")

    return schedule


def demonstrate_trading_strategies():
    """Demonstrate different trading strategies."""
    print("\n" + "=" * 60)
    print("🎯 TRADING STRATEGIES DEMONSTRATION")
    print("=" * 60)

    # Create different strategies
    strategies = {
        "arbitrage": create_arbitrage_strategy(price_threshold=0.02),
        "momentum": create_strategy("momentum", lookback_hours=4, momentum_threshold=0.05),
        "mean_reversion": create_strategy(
            "mean_reversion", lookback_hours=24, deviation_threshold=2.0
        ),
        "ml_trading": create_ml_strategy(),
    }

    print("🔧 Created Trading Strategies:")
    for name, strategy in strategies.items():
        print(f"   ✅ {name.replace('_', ' ').title()}: {strategy.__class__.__name__}")

    # Simulate market data for strategy testing
    from vpp.trading.data import MarketData

    market_data = {
        "day_ahead": MarketData(
            market="day_ahead",
            timestamp=datetime.now(),
            last_price=0.10,
            bid_price=0.099,
            ask_price=0.101,
            volume=500.0,
        ),
        "real_time": MarketData(
            market="real_time",
            timestamp=datetime.now(),
            last_price=0.13,
            bid_price=0.129,
            ask_price=0.131,
            volume=300.0,
        ),
    }

    # Test arbitrage strategy
    portfolio = Portfolio(initial_cash=10000.0)
    arbitrage_signals = strategies["arbitrage"].generate_signals(market_data, portfolio)

    print("\n📡 Arbitrage Strategy Signals:")
    if arbitrage_signals:
        for signal in arbitrage_signals:
            print(
                f"   {signal['action'].upper()} {signal['quantity']:.0f} kW in {signal['market']} @ ${signal['price']:.3f}"
            )
            print(f"   Expected profit: ${signal['expected_profit']:.2f}")
            print(f"   Confidence: {signal['confidence']:.1%}")
    else:
        print("   No arbitrage opportunities detected")

    return strategies, arbitrage_signals


def demonstrate_portfolio_management():
    """Demonstrate portfolio management and risk metrics."""
    print("\n" + "=" * 60)
    print("💼 PORTFOLIO MANAGEMENT DEMONSTRATION")
    print("=" * 60)

    # Create portfolio with initial cash
    portfolio = Portfolio(initial_cash=50000.0)

    print("💰 Initial Portfolio:")
    print(f"   Cash: ${portfolio.cash:,.2f}")
    print(f"   Positions: {len(portfolio.positions)}")

    # Simulate some trades
    from vpp.trading.portfolio import Trade

    trades = [
        Trade(
            market="day_ahead",
            side="buy",
            quantity=100.0,
            price=0.08,
            timestamp=datetime.now() - timedelta(hours=2),
            strategy="arbitrage",
        ),
        Trade(
            market="real_time",
            side="sell",
            quantity=100.0,
            price=0.12,
            timestamp=datetime.now() - timedelta(hours=1),
            strategy="arbitrage",
        ),
        Trade(
            market="day_ahead",
            side="buy",
            quantity=200.0,
            price=0.09,
            timestamp=datetime.now() - timedelta(minutes=30),
            strategy="momentum",
        ),
    ]

    # Add trades to portfolio
    for trade in trades:
        portfolio.add_trade(trade)

    # Current market prices for P&L calculation
    current_prices = {"day_ahead": 0.095, "real_time": 0.125}

    # Get portfolio summary
    summary = portfolio.get_position_summary(current_prices)

    print("\n📊 Portfolio Summary:")
    print(f"   Total positions: {summary['total_positions']}")
    print(f"   Long positions: {summary['long_positions']}")
    print(f"   Short positions: {summary['short_positions']}")
    print(f"   Total notional: ${summary['total_notional']:,.2f}")
    print(f"   Total P&L: ${summary['total_pnl']:,.2f}")
    print(f"   Current cash: ${portfolio.cash:,.2f}")
    print(f"   Total equity: ${portfolio.get_equity(current_prices):,.2f}")

    # Show individual positions
    print("\n🎯 Individual Positions:")
    for market, position_data in summary["positions"].items():
        print(f"   {market}:")
        print(f"     Quantity: {position_data['quantity']:,.1f} kW")
        print(f"     Avg Price: ${position_data['average_price']:.3f}")
        print(f"     Current Price: ${position_data['current_price']:.3f}")
        print(f"     Unrealized P&L: ${position_data['unrealized_pnl']:,.2f}")
        print(f"     Realized P&L: ${position_data['realized_pnl']:,.2f}")

    # Performance metrics
    performance = portfolio.get_performance_metrics(current_prices)

    if performance:
        print("\n📈 Performance Metrics:")
        print(f"   Total trades: {performance['total_trades']}")
        print(f"   Total return: {performance['total_return']:.1%}")
        print(f"   Win rate: {performance['win_rate']:.1%}")
        print(f"   Avg win: ${performance['avg_win']:.2f}")
        print(f"   Avg loss: ${performance['avg_loss']:.2f}")
        print(f"   Profit factor: {performance['profit_factor']:.2f}")

    return portfolio, summary, performance


def demonstrate_risk_management():
    """Demonstrate risk management capabilities."""
    print("\n" + "=" * 60)
    print("🛡️ RISK MANAGEMENT DEMONSTRATION")
    print("=" * 60)

    # Create portfolio with some positions
    portfolio = Portfolio(initial_cash=100000.0)

    # Simulate larger positions for risk demonstration
    from vpp.trading.portfolio import Trade

    large_trades = [
        Trade(
            market="day_ahead", side="buy", quantity=1000.0, price=0.10, timestamp=datetime.now()
        ),
        Trade(
            market="real_time", side="buy", quantity=800.0, price=0.12, timestamp=datetime.now()
        ),
        Trade(
            market="ancillary_services",
            side="sell",
            quantity=500.0,
            price=45.0,
            timestamp=datetime.now(),
        ),
    ]

    for trade in large_trades:
        portfolio.add_trade(trade)

    # Current market prices
    market_prices = {
        "day_ahead": 0.095,  # 5% loss
        "real_time": 0.125,  # 4% gain
        "ancillary_services": 48.0,  # 7% gain
    }

    # Calculate risk metrics
    from vpp.trading.portfolio import RiskMetrics

    risk_metrics = RiskMetrics(portfolio)
    risk_summary = risk_metrics.get_risk_summary(market_prices)

    print("⚠️ Risk Assessment:")
    print(f"   Total positions: {risk_summary['total_positions']}")
    print(f"   Total notional: ${risk_summary['total_notional']:,.2f}")
    print(f"   Max concentration: {risk_summary['max_concentration']:.1%}")
    print(f"   Current drawdown: {risk_summary['current_drawdown']:.1%}")
    print(f"   Max drawdown: {risk_summary['max_drawdown']:.1%}")

    # Position concentrations
    print("\n🎯 Position Concentrations:")
    for market, concentration in risk_summary["position_concentrations"].items():
        print(f"   {market}: {concentration:.1%}")

    # Risk limits check
    risk_limits = {
        "max_concentration": 0.4,  # 40%
        "max_drawdown": 0.1,  # 10%
        "max_single_position": 50000,  # $50k
    }

    print("\n🚨 Risk Limit Checks:")

    # Check concentration limits
    max_concentration = risk_summary["max_concentration"]
    if max_concentration > risk_limits["max_concentration"]:
        print(
            f"   ❌ Concentration limit breached: {max_concentration:.1%} > {risk_limits['max_concentration']:.1%}"
        )
    else:
        print(f"   ✅ Concentration within limits: {max_concentration:.1%}")

    # Check drawdown limits
    current_drawdown = risk_summary["current_drawdown"]
    if current_drawdown > risk_limits["max_drawdown"]:
        print(
            f"   ❌ Drawdown limit breached: {current_drawdown:.1%} > {risk_limits['max_drawdown']:.1%}"
        )
    else:
        print(f"   ✅ Drawdown within limits: {current_drawdown:.1%}")

    return risk_summary


def demonstrate_complete_trading_system():
    """Demonstrate the complete trading system integration."""
    print("\n" + "=" * 60)
    print("🚀 COMPLETE TRADING SYSTEM DEMONSTRATION")
    print("=" * 60)

    # Create trading engine
    config = {
        "risk_limits": {"max_position": 1000.0, "max_daily_loss": 5000.0, "max_drawdown": 0.1},
        "markets": ["day_ahead", "real_time"],
        "strategies": ["arbitrage"],
        "data_provider": {
            "type": "simulated",
            "update_frequency": 1,  # 1 second for demo
        },
    }

    engine = create_trading_engine(config)

    print("🔧 Trading Engine Configuration:")
    print(f"   Risk limits: Max position {config['risk_limits']['max_position']} kW")
    print(f"   Markets: {', '.join(config['markets'])}")
    print(f"   Data provider: {config['data_provider']['type']}")

    # Add markets. Order routing is by market *name*, so name them after the
    # ``market`` field the orders below use.
    day_ahead = create_market("day_ahead", "day_ahead")
    real_time = create_market("real_time", "real_time")

    # Set markets as open for demo
    day_ahead.status = MarketStatus.OPEN
    real_time.status = MarketStatus.OPEN

    # Seed the simulated real-time book with third-party liquidity so the
    # engine's orders have something to match against.
    real_time.order_book.add_order(LimitOrder("real_time", "sell", 150, 0.115))
    real_time.order_book.add_order(LimitOrder("real_time", "buy", 80, 0.110))

    engine.add_market(day_ahead)
    engine.add_market(real_time)

    # Add trading strategy
    arbitrage_strategy = create_arbitrage_strategy(price_threshold=0.02)
    engine.add_strategy(arbitrage_strategy)

    print("\n📊 System Components:")
    print(f"   ✅ Markets: {len(engine.markets)} configured")
    print(f"   ✅ Strategies: {len(engine.strategies)} active")
    print("   ✅ Risk manager: Configured with limits")
    print("   ✅ Portfolio manager: Ready for tracking")

    # Simulate some trading activity
    print("\n💹 Simulating Trading Activity...")
    if not engine.start():
        raise RuntimeError("trading engine failed to start")

    orders = [
        create_order("limit", "real_time", "buy", 100, 0.12),  # crosses the 0.115 offer
        create_order("market", "real_time", "sell", 50),  # hits the 0.110 bid
        create_order("limit", "day_ahead", "buy", 100, 0.08),  # checked by the risk manager
    ]

    accepted = 0
    for i, order in enumerate(orders, 1):
        ok = engine.submit_order(order)
        accepted += ok
        price = f"${order.price:.3f}" if order.order_type.value != "market" else "market"
        verdict = "accepted" if ok else "rejected by pre-trade checks"
        print(
            f"   📝 Order {i}: {order.side.upper()} {order.quantity} kW in {order.market} "
            f"@ {price} ({order.order_type.value}) -> {verdict}"
        )

    # Orders are executed asynchronously by the engine's worker thread.
    deadline = time.monotonic() + 5.0
    while engine.get_metrics()["orders_processed"] < accepted and time.monotonic() < deadline:
        time.sleep(0.05)
    engine.stop()

    # Get engine metrics
    metrics = engine.get_metrics()
    portfolio = engine.get_portfolio()

    print("\n📈 Trading Engine Metrics:")
    print(f"   Orders processed: {metrics['orders_processed']}")
    print(f"   Trades executed: {metrics['trades_executed']}")
    print(f"   Total volume: {metrics['total_volume']:.1f} kW")
    print(f"   Portfolio cash flow: ${portfolio.cash:,.2f}")
    print(f"   Portfolio positions: {len(portfolio.positions)}")

    return engine


def main():
    """Run the complete trading demonstration."""
    print("🔋 VPP TRADING SYSTEM - COMPREHENSIVE DEMONSTRATION")
    print("=" * 80)
    print("This demo showcases the complete trading capabilities of the VPP library")
    print("including arbitrage, optimization, portfolio management, and risk control.")

    try:
        # Run all demonstrations
        arbitrage_result = demonstrate_arbitrage_trading()
        schedule_result = demonstrate_trading_schedule_optimization()
        strategies_result = demonstrate_trading_strategies()
        portfolio_result = demonstrate_portfolio_management()
        risk_result = demonstrate_risk_management()
        engine = demonstrate_complete_trading_system()

        # Summary
        print("\n" + "=" * 60)
        print("✅ DEMONSTRATION COMPLETE - SUMMARY")
        print("=" * 60)

        print(
            f"🔄 Arbitrage: {'Profitable' if arbitrage_result['profitable'] else 'No opportunity'}"
        )
        print(f"📈 Optimization: {'Success' if schedule_result['success'] else 'Failed'}")
        print(f"🎯 Strategies: {len(strategies_result[0])} strategies tested")
        print(
            f"💼 Portfolio: {portfolio_result[1]['total_positions']} positions, ${portfolio_result[1]['total_pnl']:.2f} P&L"
        )
        print(f"🛡️ Risk: {risk_result['total_positions']} positions monitored")
        print(
            f"🚀 System: trading engine executed {engine.get_metrics()['trades_executed']} trades"
        )

        print("\n🎉 Demo complete. Everything above ran against simulated markets;")
        print("   no orders left this process. Components exercised:")
        print("   ✅ In-memory order matching (simulated venue)")
        print("   ✅ Multi-market arbitrage detection")
        print("   ✅ Portfolio and P&L tracking")
        print("   ✅ Pre-trade risk limit checks")
        print("   ✅ Synthetic market data")
        print("   ✅ Strategy framework (the ML strategy is a research placeholder)")

    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        raise


if __name__ == "__main__":
    sys.exit(main())
