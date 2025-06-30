"""
Portfolio management for VPP trading operations.

This module provides comprehensive portfolio tracking, position management,
P&L calculation, and risk metrics for energy trading.
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


@dataclass
class Trade:
    """Individual trade record."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    order_id: str = ""
    market: str = ""
    side: str = ""  # "buy" or "sell"
    quantity: float = 0.0
    price: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    fees: float = 0.0
    commission: float = 0.0
    
    # Trade metadata
    strategy: Optional[str] = None
    execution_venue: Optional[str] = None
    counterparty: Optional[str] = None
    settlement_date: Optional[datetime] = None
    
    # P&L tracking
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    def get_notional_value(self) -> float:
        """Get notional value of the trade."""
        return self.quantity * self.price
    
    def get_total_cost(self) -> float:
        """Get total cost including fees and commission."""
        return self.get_notional_value() + self.fees + self.commission
    
    def calculate_pnl(self, current_price: float) -> float:
        """Calculate P&L at current market price."""
        if self.side == "buy":
            return (current_price - self.price) * self.quantity - self.fees - self.commission
        else:  # sell
            return (self.price - current_price) * self.quantity - self.fees - self.commission


@dataclass
class Position:
    """Portfolio position in a specific market."""
    market: str = ""
    quantity: float = 0.0  # Net position (positive = long, negative = short)
    average_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    # Position tracking
    total_bought: float = 0.0
    total_sold: float = 0.0
    total_cost: float = 0.0
    
    # Risk metrics
    var_1d: float = 0.0  # 1-day Value at Risk
    max_drawdown: float = 0.0
    
    # Timestamps
    first_trade_time: Optional[datetime] = None
    last_trade_time: Optional[datetime] = None
    
    def update_from_trade(self, trade: Trade) -> None:
        """Update position from a new trade."""
        if trade.market != self.market:
            raise ValueError(f"Trade market {trade.market} doesn't match position market {self.market}")
        
        # Update timestamps
        if self.first_trade_time is None:
            self.first_trade_time = trade.timestamp
        self.last_trade_time = trade.timestamp
        
        # Update position
        if trade.side == "buy":
            self.total_bought += trade.quantity
            new_quantity = self.quantity + trade.quantity
            
            if self.quantity >= 0:  # Adding to long position
                total_value = self.average_price * self.quantity + trade.price * trade.quantity
                self.average_price = total_value / new_quantity if new_quantity != 0 else 0
            else:  # Covering short position
                if new_quantity >= 0:  # Position flipped to long
                    covered_quantity = abs(self.quantity)
                    remaining_quantity = trade.quantity - covered_quantity
                    
                    # Realize P&L on covered portion
                    self.realized_pnl += (self.average_price - trade.price) * covered_quantity
                    
                    # Set new average price for remaining long position
                    self.average_price = trade.price if remaining_quantity > 0 else 0
                else:  # Still short, update average
                    total_value = self.average_price * abs(self.quantity) + trade.price * trade.quantity
                    self.average_price = total_value / abs(new_quantity)
            
            self.quantity = new_quantity
            
        else:  # sell
            self.total_sold += trade.quantity
            new_quantity = self.quantity - trade.quantity
            
            if self.quantity <= 0:  # Adding to short position
                total_value = self.average_price * abs(self.quantity) + trade.price * trade.quantity
                self.average_price = total_value / abs(new_quantity) if new_quantity != 0 else 0
            else:  # Reducing long position
                if new_quantity <= 0:  # Position flipped to short
                    sold_quantity = self.quantity
                    remaining_quantity = trade.quantity - sold_quantity
                    
                    # Realize P&L on sold portion
                    self.realized_pnl += (trade.price - self.average_price) * sold_quantity
                    
                    # Set new average price for remaining short position
                    self.average_price = trade.price if remaining_quantity > 0 else 0
                else:  # Still long, realize P&L on sold portion
                    self.realized_pnl += (trade.price - self.average_price) * trade.quantity
            
            self.quantity = new_quantity
        
        # Update total cost
        self.total_cost += trade.get_total_cost()
    
    def calculate_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L at current market price."""
        if self.quantity == 0:
            return 0.0
        
        if self.quantity > 0:  # Long position
            self.unrealized_pnl = (current_price - self.average_price) * self.quantity
        else:  # Short position
            self.unrealized_pnl = (self.average_price - current_price) * abs(self.quantity)
        
        return self.unrealized_pnl
    
    def get_total_pnl(self, current_price: float) -> float:
        """Get total P&L (realized + unrealized)."""
        return self.realized_pnl + self.calculate_unrealized_pnl(current_price)
    
    def get_notional_value(self, current_price: float) -> float:
        """Get notional value of position."""
        return abs(self.quantity) * current_price
    
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.quantity > 0
    
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.quantity < 0
    
    def is_flat(self) -> bool:
        """Check if position is flat (no exposure)."""
        return self.quantity == 0


class Portfolio:
    """Portfolio manager for tracking positions and P&L."""
    
    def __init__(self, initial_cash: float = 0.0):
        """Initialize portfolio."""
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        
        # Performance tracking
        self.daily_pnl_history: List[Dict[str, Any]] = []
        self.equity_curve: List[Dict[str, Any]] = []
        
        # Risk metrics
        self.max_drawdown = 0.0
        self.peak_equity = initial_cash
        
        # Timestamps
        self.creation_time = datetime.now()
        self.last_update_time = datetime.now()
    
    def add_trade(self, trade: Trade) -> None:
        """Add a trade to the portfolio."""
        self.trades.append(trade)
        
        # Update cash position
        if trade.side == "buy":
            self.cash -= trade.get_total_cost()
        else:  # sell
            self.cash += trade.get_notional_value() - trade.fees - trade.commission
        
        # Update position
        if trade.market not in self.positions:
            self.positions[trade.market] = Position(market=trade.market)
        
        self.positions[trade.market].update_from_trade(trade)
        self.last_update_time = datetime.now()
    
    def update_from_trade(self, trade: Trade) -> None:
        """Alias for add_trade for compatibility."""
        self.add_trade(trade)
    
    def get_position(self, market: str) -> Optional[Position]:
        """Get position for a specific market."""
        return self.positions.get(market)
    
    def get_total_notional(self, market_prices: Dict[str, float]) -> float:
        """Get total notional value of all positions."""
        total = 0.0
        for market, position in self.positions.items():
            if market in market_prices:
                total += position.get_notional_value(market_prices[market])
        return total
    
    def calculate_total_pnl(self, market_prices: Dict[str, float] = None) -> float:
        """Calculate total portfolio P&L."""
        if market_prices is None:
            market_prices = {}
        
        total_pnl = 0.0
        for market, position in self.positions.items():
            current_price = market_prices.get(market, position.average_price)
            total_pnl += position.get_total_pnl(current_price)
        
        return total_pnl
    
    def calculate_daily_pnl(self, date: datetime = None) -> float:
        """Calculate P&L for a specific day."""
        if date is None:
            date = datetime.now().date()
        
        daily_trades = [trade for trade in self.trades 
                       if trade.timestamp.date() == date]
        
        daily_pnl = 0.0
        for trade in daily_trades:
            if trade.side == "sell":
                daily_pnl += trade.realized_pnl
        
        return daily_pnl
    
    def calculate_max_drawdown(self, market_prices: Dict[str, float] = None) -> float:
        """Calculate maximum drawdown."""
        if not self.equity_curve:
            return 0.0
        
        peak = self.initial_cash
        max_dd = 0.0
        
        for point in self.equity_curve:
            equity = point["equity"]
            if equity > peak:
                peak = equity
            
            drawdown = (peak - equity) / peak if peak > 0 else 0
            max_dd = max(max_dd, drawdown)
        
        self.max_drawdown = max_dd
        return max_dd
    
    def get_equity(self, market_prices: Dict[str, float] = None) -> float:
        """Get current portfolio equity."""
        return self.cash + self.calculate_total_pnl(market_prices)
    
    def update_equity_curve(self, market_prices: Dict[str, float] = None) -> None:
        """Update equity curve with current values."""
        equity = self.get_equity(market_prices)
        total_pnl = self.calculate_total_pnl(market_prices)
        
        self.equity_curve.append({
            "timestamp": datetime.now(),
            "equity": equity,
            "cash": self.cash,
            "total_pnl": total_pnl,
            "positions_value": equity - self.cash
        })
        
        # Update peak equity and drawdown
        if equity > self.peak_equity:
            self.peak_equity = equity
        
        current_drawdown = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
    
    def get_position_summary(self, market_prices: Dict[str, float] = None) -> Dict[str, Any]:
        """Get summary of all positions."""
        if market_prices is None:
            market_prices = {}
        
        summary = {
            "total_positions": len(self.positions),
            "long_positions": 0,
            "short_positions": 0,
            "flat_positions": 0,
            "total_notional": 0.0,
            "total_pnl": 0.0,
            "positions": {}
        }
        
        for market, position in self.positions.items():
            current_price = market_prices.get(market, position.average_price)
            
            if position.is_long():
                summary["long_positions"] += 1
            elif position.is_short():
                summary["short_positions"] += 1
            else:
                summary["flat_positions"] += 1
            
            notional = position.get_notional_value(current_price)
            pnl = position.get_total_pnl(current_price)
            
            summary["total_notional"] += notional
            summary["total_pnl"] += pnl
            
            summary["positions"][market] = {
                "quantity": position.quantity,
                "average_price": position.average_price,
                "current_price": current_price,
                "notional_value": notional,
                "unrealized_pnl": position.calculate_unrealized_pnl(current_price),
                "realized_pnl": position.realized_pnl,
                "total_pnl": pnl
            }
        
        return summary
    
    def get_performance_metrics(self, market_prices: Dict[str, float] = None) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        if not self.trades:
            return {}
        
        # Basic metrics
        total_trades = len(self.trades)
        total_pnl = self.calculate_total_pnl(market_prices)
        current_equity = self.get_equity(market_prices)
        
        # Calculate returns
        total_return = (current_equity - self.initial_cash) / self.initial_cash if self.initial_cash > 0 else 0
        
        # Calculate daily returns for Sharpe ratio
        daily_returns = []
        if len(self.equity_curve) > 1:
            for i in range(1, len(self.equity_curve)):
                prev_equity = self.equity_curve[i-1]["equity"]
                curr_equity = self.equity_curve[i]["equity"]
                daily_return = (curr_equity - prev_equity) / prev_equity if prev_equity > 0 else 0
                daily_returns.append(daily_return)
        
        # Risk metrics
        if daily_returns:
            avg_daily_return = np.mean(daily_returns)
            daily_volatility = np.std(daily_returns)
            sharpe_ratio = avg_daily_return / daily_volatility if daily_volatility > 0 else 0
            sharpe_ratio *= np.sqrt(252)  # Annualize
        else:
            avg_daily_return = 0
            daily_volatility = 0
            sharpe_ratio = 0
        
        # Win/loss metrics
        winning_trades = [t for t in self.trades if t.realized_pnl > 0]
        losing_trades = [t for t in self.trades if t.realized_pnl < 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        avg_win = np.mean([t.realized_pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.realized_pnl for t in losing_trades]) if losing_trades else 0
        
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        return {
            "total_trades": total_trades,
            "total_pnl": total_pnl,
            "total_return": total_return,
            "current_equity": current_equity,
            "max_drawdown": self.calculate_max_drawdown(market_prices),
            "sharpe_ratio": sharpe_ratio,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "daily_volatility": daily_volatility,
            "avg_daily_return": avg_daily_return
        }


class PnLCalculator:
    """Utility class for P&L calculations."""
    
    @staticmethod
    def calculate_trade_pnl(entry_price: float, exit_price: float, 
                           quantity: float, side: str, fees: float = 0.0) -> float:
        """Calculate P&L for a single trade."""
        if side == "buy":
            # Long position: profit when price goes up
            return (exit_price - entry_price) * quantity - fees
        else:  # sell
            # Short position: profit when price goes down
            return (entry_price - exit_price) * quantity - fees
    
    @staticmethod
    def calculate_position_pnl(trades: List[Trade], current_price: float) -> Dict[str, float]:
        """Calculate P&L for a position from list of trades."""
        if not trades:
            return {"realized_pnl": 0.0, "unrealized_pnl": 0.0, "total_pnl": 0.0}
        
        # Group trades by market
        market_trades = {}
        for trade in trades:
            if trade.market not in market_trades:
                market_trades[trade.market] = []
            market_trades[trade.market].append(trade)
        
        total_realized = 0.0
        total_unrealized = 0.0
        
        for market, market_trade_list in market_trades.items():
            # Create position and update with trades
            position = Position(market=market)
            for trade in market_trade_list:
                position.update_from_trade(trade)
            
            # Calculate P&L
            total_realized += position.realized_pnl
            total_unrealized += position.calculate_unrealized_pnl(current_price)
        
        return {
            "realized_pnl": total_realized,
            "unrealized_pnl": total_unrealized,
            "total_pnl": total_realized + total_unrealized
        }
    
    @staticmethod
    def calculate_var(returns: List[float], confidence_level: float = 0.05) -> float:
        """Calculate Value at Risk (VaR)."""
        if not returns:
            return 0.0
        
        sorted_returns = sorted(returns)
        var_index = int(len(sorted_returns) * confidence_level)
        return abs(sorted_returns[var_index]) if var_index < len(sorted_returns) else 0.0
    
    @staticmethod
    def calculate_expected_shortfall(returns: List[float], confidence_level: float = 0.05) -> float:
        """Calculate Expected Shortfall (Conditional VaR)."""
        if not returns:
            return 0.0
        
        sorted_returns = sorted(returns)
        var_index = int(len(sorted_returns) * confidence_level)
        tail_returns = sorted_returns[:var_index]
        
        return abs(np.mean(tail_returns)) if tail_returns else 0.0


class RiskMetrics:
    """Risk metrics calculator for portfolio analysis."""
    
    def __init__(self, portfolio: Portfolio):
        """Initialize with portfolio reference."""
        self.portfolio = portfolio
    
    def calculate_portfolio_var(self, market_prices: Dict[str, float], 
                              confidence_level: float = 0.05, 
                              time_horizon: int = 1) -> float:
        """Calculate portfolio Value at Risk."""
        # Get daily returns from equity curve
        daily_returns = []
        if len(self.portfolio.equity_curve) > 1:
            for i in range(1, len(self.portfolio.equity_curve)):
                prev_equity = self.portfolio.equity_curve[i-1]["equity"]
                curr_equity = self.portfolio.equity_curve[i]["equity"]
                daily_return = (curr_equity - prev_equity) / prev_equity if prev_equity > 0 else 0
                daily_returns.append(daily_return)
        
        if not daily_returns:
            return 0.0
        
        # Calculate VaR
        var_1d = PnLCalculator.calculate_var(daily_returns, confidence_level)
        current_equity = self.portfolio.get_equity(market_prices)
        
        # Scale by time horizon
        var_scaled = var_1d * np.sqrt(time_horizon) * current_equity
        
        return var_scaled
    
    def calculate_position_concentration(self, market_prices: Dict[str, float]) -> Dict[str, float]:
        """Calculate position concentration risk."""
        total_notional = self.portfolio.get_total_notional(market_prices)
        
        concentrations = {}
        for market, position in self.portfolio.positions.items():
            if market in market_prices:
                position_notional = position.get_notional_value(market_prices[market])
                concentration = position_notional / total_notional if total_notional > 0 else 0
                concentrations[market] = concentration
        
        return concentrations
    
    def calculate_correlation_risk(self, price_history: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
        """Calculate correlation matrix for portfolio positions."""
        markets = list(self.portfolio.positions.keys())
        correlation_matrix = {}
        
        for market1 in markets:
            correlation_matrix[market1] = {}
            for market2 in markets:
                if market1 in price_history and market2 in price_history:
                    prices1 = price_history[market1]
                    prices2 = price_history[market2]
                    
                    if len(prices1) == len(prices2) and len(prices1) > 1:
                        correlation = np.corrcoef(prices1, prices2)[0, 1]
                        correlation_matrix[market1][market2] = correlation
                    else:
                        correlation_matrix[market1][market2] = 0.0
                else:
                    correlation_matrix[market1][market2] = 0.0
        
        return correlation_matrix
    
    def get_risk_summary(self, market_prices: Dict[str, float]) -> Dict[str, Any]:
        """Get comprehensive risk summary."""
        var_1d = self.calculate_portfolio_var(market_prices, 0.05, 1)
        var_5d = self.calculate_portfolio_var(market_prices, 0.05, 5)
        
        concentrations = self.calculate_position_concentration(market_prices)
        max_concentration = max(concentrations.values()) if concentrations else 0.0
        
        current_equity = self.portfolio.get_equity(market_prices)
        max_drawdown = self.portfolio.calculate_max_drawdown(market_prices)
        
        return {
            "var_1d_95": var_1d,
            "var_5d_95": var_5d,
            "max_concentration": max_concentration,
            "current_drawdown": (self.portfolio.peak_equity - current_equity) / self.portfolio.peak_equity if self.portfolio.peak_equity > 0 else 0,
            "max_drawdown": max_drawdown,
            "position_concentrations": concentrations,
            "total_positions": len(self.portfolio.positions),
            "total_notional": self.portfolio.get_total_notional(market_prices)
        }
