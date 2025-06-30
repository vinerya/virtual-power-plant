"""
Advanced trading module for Virtual Power Plant energy market participation.

This module provides comprehensive trading capabilities including:
- Order management and execution
- Market participation (day-ahead, real-time, ancillary services)
- Portfolio management and risk control
- Advanced trading strategies with ML integration
- Real-time market data integration
"""

from .core import (
    TradingEngine,
    OrderManager,
    PortfolioManager,
    RiskManager,
    MarketDataManager
)

from .orders import (
    OrderType,
    OrderStatus,
    MarketOrder,
    LimitOrder,
    StopOrder,
    Order,
    OrderBook
)

from .markets import (
    MarketType,
    Market,
    DayAheadMarket,
    RealTimeMarket,
    AncillaryServicesMarket,
    BilateralMarket
)

from .strategies import (
    TradingStrategy,
    ArbitrageStrategy,
    MomentumStrategy,
    MeanReversionStrategy,
    MLTradingStrategy,
    MultiMarketStrategy
)

from .portfolio import (
    Position,
    Portfolio,
    Trade,
    PnLCalculator,
    RiskMetrics
)

from .data import (
    MarketData,
    PriceData,
    VolumeData,
    MarketDataProvider,
    SimulatedDataProvider,
    LiveDataProvider
)

# Version information
__version__ = "1.0.0"
__author__ = "VPP Trading Team"

# Export all public classes and functions
__all__ = [
    # Core trading engine
    "TradingEngine",
    "OrderManager", 
    "PortfolioManager",
    "RiskManager",
    "MarketDataManager",
    
    # Order management
    "OrderType",
    "OrderStatus",
    "MarketOrder",
    "LimitOrder", 
    "StopOrder",
    "Order",
    "OrderBook",
    
    # Market types
    "MarketType",
    "Market",
    "DayAheadMarket",
    "RealTimeMarket",
    "AncillaryServicesMarket",
    "BilateralMarket",
    
    # Trading strategies
    "TradingStrategy",
    "ArbitrageStrategy",
    "MomentumStrategy",
    "MeanReversionStrategy", 
    "MLTradingStrategy",
    "MultiMarketStrategy",
    
    # Portfolio management
    "Position",
    "Portfolio",
    "Trade",
    "PnLCalculator",
    "RiskMetrics",
    
    # Market data
    "MarketData",
    "PriceData",
    "VolumeData",
    "MarketDataProvider",
    "SimulatedDataProvider",
    "LiveDataProvider",
]


def create_trading_engine(config: dict = None) -> TradingEngine:
    """
    Create a pre-configured trading engine.
    
    Args:
        config: Trading engine configuration
        
    Returns:
        Configured TradingEngine instance
        
    Example:
        >>> engine = create_trading_engine({
        ...     "risk_limits": {"max_position": 1000, "max_daily_loss": 5000},
        ...     "markets": ["day_ahead", "real_time", "ancillary"],
        ...     "strategies": ["arbitrage", "momentum"]
        ... })
        >>> engine.start()
    """
    return TradingEngine(config or {})


def create_arbitrage_strategy(price_threshold: float = 0.02) -> ArbitrageStrategy:
    """
    Create an arbitrage trading strategy.
    
    Args:
        price_threshold: Minimum price difference for arbitrage ($/kWh)
        
    Returns:
        Configured ArbitrageStrategy instance
        
    Example:
        >>> strategy = create_arbitrage_strategy(price_threshold=0.03)
        >>> engine.add_strategy(strategy)
    """
    return ArbitrageStrategy(price_threshold=price_threshold)


def create_ml_strategy(model_path: str = None) -> MLTradingStrategy:
    """
    Create a machine learning trading strategy.
    
    Args:
        model_path: Path to trained ML model
        
    Returns:
        Configured MLTradingStrategy instance
        
    Example:
        >>> strategy = create_ml_strategy("models/lstm_price_predictor.pkl")
        >>> engine.add_strategy(strategy)
    """
    return MLTradingStrategy(model_path=model_path)


# Convenience functions for common trading operations
def calculate_arbitrage_opportunity(price1: float, price2: float, 
                                  transaction_cost: float = 0.001) -> dict:
    """
    Calculate arbitrage opportunity between two markets.
    
    Args:
        price1: Price in market 1 ($/kWh)
        price2: Price in market 2 ($/kWh)
        transaction_cost: Transaction cost per trade ($/kWh)
        
    Returns:
        Dictionary with arbitrage analysis
        
    Example:
        >>> opportunity = calculate_arbitrage_opportunity(0.12, 0.15, 0.002)
        >>> if opportunity['profitable']:
        ...     print(f"Profit: ${opportunity['profit_per_mwh']:.2f}/MWh")
    """
    price_diff = abs(price2 - price1)
    total_cost = 2 * transaction_cost  # Buy and sell
    profit = price_diff - total_cost
    
    return {
        "profitable": profit > 0,
        "price_difference": price_diff,
        "transaction_cost": total_cost,
        "profit_per_mwh": profit * 1000,  # Convert to $/MWh
        "profit_margin": (profit / max(price1, price2)) * 100 if profit > 0 else 0,
        "buy_market": "market1" if price1 < price2 else "market2",
        "sell_market": "market2" if price1 < price2 else "market1"
    }


def optimize_trading_schedule(prices: list, capacity: float, 
                            efficiency: float = 0.95) -> dict:
    """
    Optimize trading schedule for given price forecast.
    
    Args:
        prices: Hourly price forecast ($/kWh)
        capacity: Storage capacity (kWh)
        efficiency: Round-trip efficiency
        
    Returns:
        Optimal trading schedule
        
    Example:
        >>> prices = [0.08, 0.12, 0.15, 0.10, 0.06, 0.14]
        >>> schedule = optimize_trading_schedule(prices, capacity=1000)
        >>> print(f"Expected profit: ${schedule['total_profit']:.2f}")
    """
    from ..optimization import create_stochastic_problem, solve_with_fallback
    
    # Create optimization problem
    base_data = {
        "base_prices": prices,
        "renewable_forecast": [0] * len(prices),  # No renewable for pure trading
        "battery_capacity": capacity,
        "max_power": capacity * 0.25,  # 4-hour discharge rate
        "efficiency": efficiency
    }
    
    problem = create_stochastic_problem(base_data, num_scenarios=1)
    result = solve_with_fallback(problem, timeout_ms=5000)
    
    if result.status.value in ["success", "fallback_used"]:
        battery_schedule = result.solution.get("battery_power", [0] * len(prices))
        
        # Calculate profit
        total_profit = 0
        for i, (price, power) in enumerate(zip(prices, battery_schedule)):
            if power > 0:  # Charging (buying)
                total_profit -= power * price
            else:  # Discharging (selling)
                total_profit -= power * price  # power is negative, so this adds profit
        
        return {
            "success": True,
            "battery_schedule": battery_schedule,
            "total_profit": total_profit,
            "profit_per_mwh": total_profit / (capacity / 1000),
            "utilization": max(battery_schedule) / (capacity * 0.25) * 100,
            "solve_time": result.solve_time
        }
    else:
        return {
            "success": False,
            "error": "Optimization failed",
            "battery_schedule": [0] * len(prices),
            "total_profit": 0
        }


def validate_trading_config(config: dict) -> list:
    """
    Validate trading configuration parameters.
    
    Args:
        config: Trading configuration dictionary
        
    Returns:
        List of validation error messages (empty if valid)
        
    Example:
        >>> config = {
        ...     "risk_limits": {"max_position": 1000},
        ...     "markets": ["day_ahead", "real_time"]
        ... }
        >>> errors = validate_trading_config(config)
        >>> if not errors:
        ...     print("Configuration is valid")
    """
    errors = []
    
    # Validate risk limits
    risk_limits = config.get("risk_limits", {})
    if "max_position" in risk_limits:
        if not isinstance(risk_limits["max_position"], (int, float)) or risk_limits["max_position"] <= 0:
            errors.append("max_position must be a positive number")
    
    if "max_daily_loss" in risk_limits:
        if not isinstance(risk_limits["max_daily_loss"], (int, float)) or risk_limits["max_daily_loss"] <= 0:
            errors.append("max_daily_loss must be a positive number")
    
    # Validate markets
    valid_markets = ["day_ahead", "real_time", "ancillary", "bilateral"]
    markets = config.get("markets", [])
    for market in markets:
        if market not in valid_markets:
            errors.append(f"Invalid market '{market}'. Must be one of {valid_markets}")
    
    # Validate strategies
    valid_strategies = ["arbitrage", "momentum", "mean_reversion", "ml", "multi_market"]
    strategies = config.get("strategies", [])
    for strategy in strategies:
        if strategy not in valid_strategies:
            errors.append(f"Invalid strategy '{strategy}'. Must be one of {valid_strategies}")
    
    # Validate data providers
    data_config = config.get("data_provider", {})
    provider_type = data_config.get("type")
    if provider_type and provider_type not in ["simulated", "live", "historical"]:
        errors.append(f"Invalid data provider type '{provider_type}'")
    
    return errors


# Module-level configuration
_default_trading_config = {
    "risk_limits": {
        "max_position": 1000.0,  # kW
        "max_daily_loss": 5000.0,  # $
        "max_drawdown": 0.1,  # 10%
        "position_limit_check_interval": 60  # seconds
    },
    "markets": ["day_ahead", "real_time"],
    "strategies": ["arbitrage"],
    "data_provider": {
        "type": "simulated",
        "update_frequency": 60  # seconds
    },
    "execution": {
        "order_timeout": 300,  # seconds
        "max_slippage": 0.01,  # $/kWh
        "partial_fill_threshold": 0.1  # 10%
    }
}


def configure_trading(config: dict) -> None:
    """
    Configure global trading settings.
    
    Args:
        config: Configuration dictionary
        
    Example:
        >>> configure_trading({
        ...     "risk_limits": {"max_position": 2000},
        ...     "markets": ["day_ahead", "real_time", "ancillary"]
        ... })
    """
    global _default_trading_config
    _default_trading_config.update(config)


def get_trading_config() -> dict:
    """
    Get current global trading configuration.
    
    Returns:
        Current configuration dictionary
    """
    return _default_trading_config.copy()


# Performance monitoring utilities
def calculate_trading_metrics(trades: list) -> dict:
    """
    Calculate comprehensive trading performance metrics.
    
    Args:
        trades: List of completed trades
        
    Returns:
        Dictionary with performance metrics
        
    Example:
        >>> metrics = calculate_trading_metrics(portfolio.get_completed_trades())
        >>> print(f"Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
        >>> print(f"Win rate: {metrics['win_rate']:.1%}")
    """
    if not trades:
        return {}
    
    # Calculate basic metrics
    profits = [trade.get("profit", 0) for trade in trades]
    total_profit = sum(profits)
    avg_profit = total_profit / len(trades)
    
    # Calculate risk metrics
    profit_std = (sum((p - avg_profit) ** 2 for p in profits) / len(profits)) ** 0.5
    sharpe_ratio = avg_profit / profit_std if profit_std > 0 else 0
    
    # Calculate win/loss metrics
    winning_trades = [p for p in profits if p > 0]
    losing_trades = [p for p in profits if p < 0]
    
    win_rate = len(winning_trades) / len(trades)
    avg_win = sum(winning_trades) / len(winning_trades) if winning_trades else 0
    avg_loss = sum(losing_trades) / len(losing_trades) if losing_trades else 0
    
    # Calculate drawdown
    cumulative_profits = []
    running_total = 0
    for profit in profits:
        running_total += profit
        cumulative_profits.append(running_total)
    
    peak = cumulative_profits[0]
    max_drawdown = 0
    for profit in cumulative_profits:
        if profit > peak:
            peak = profit
        drawdown = (peak - profit) / peak if peak > 0 else 0
        max_drawdown = max(max_drawdown, drawdown)
    
    return {
        "total_trades": len(trades),
        "total_profit": total_profit,
        "avg_profit_per_trade": avg_profit,
        "profit_std": profit_std,
        "sharpe_ratio": sharpe_ratio,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": abs(avg_win / avg_loss) if avg_loss != 0 else float('inf'),
        "max_drawdown": max_drawdown,
        "total_volume": sum(trade.get("volume", 0) for trade in trades)
    }


if __name__ == "__main__":
    # Example usage
    print("VPP Trading Module - Example Usage")
    print("=" * 40)
    
    # Create trading engine
    engine = create_trading_engine({
        "risk_limits": {"max_position": 1000, "max_daily_loss": 5000},
        "markets": ["day_ahead", "real_time"],
        "strategies": ["arbitrage"]
    })
    
    # Example arbitrage calculation
    opportunity = calculate_arbitrage_opportunity(0.12, 0.15, 0.002)
    print(f"Arbitrage opportunity: {opportunity}")
    
    # Example trading schedule optimization
    prices = [0.08, 0.12, 0.15, 0.10, 0.06, 0.14, 0.18, 0.09]
    schedule = optimize_trading_schedule(prices, capacity=1000)
    print(f"Optimized trading schedule: {schedule}")
