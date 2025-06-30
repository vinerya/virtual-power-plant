"""
Core trading engine for Virtual Power Plant energy market participation.

This module provides the main trading engine with order management,
portfolio tracking, risk management, and market data integration.
"""

import time
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import queue
import uuid

from .orders import Order, OrderStatus, OrderType, OrderBook
from .markets import Market, MarketType
from .portfolio import Portfolio, Position, Trade
from .data import MarketDataProvider, MarketData


class TradingEngineStatus(Enum):
    """Trading engine status."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class RiskLimits:
    """Risk management limits."""
    max_position: float = 1000.0  # kW
    max_daily_loss: float = 5000.0  # $
    max_drawdown: float = 0.1  # 10%
    position_limit_check_interval: int = 60  # seconds
    var_limit: float = 10000.0  # $ Value at Risk
    concentration_limit: float = 0.3  # 30% max in single market


@dataclass
class ExecutionConfig:
    """Order execution configuration."""
    order_timeout: int = 300  # seconds
    max_slippage: float = 0.01  # $/kWh
    partial_fill_threshold: float = 0.1  # 10%
    retry_attempts: int = 3
    retry_delay: float = 1.0  # seconds


class TradingEngine:
    """
    Main trading engine for VPP energy market participation.
    
    Provides comprehensive trading capabilities including:
    - Order management and execution
    - Portfolio and position tracking
    - Risk management and monitoring
    - Market data integration
    - Strategy execution
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize trading engine.
        
        Args:
            config: Trading engine configuration
        """
        self.config = config or {}
        self.logger = logging.getLogger("trading.engine")
        
        # Engine state
        self.status = TradingEngineStatus.STOPPED
        self.start_time: Optional[datetime] = None
        self.stop_time: Optional[datetime] = None
        
        # Core components
        self.order_manager = OrderManager()
        self.portfolio_manager = PortfolioManager()
        self.risk_manager = RiskManager(
            limits=RiskLimits(**self.config.get("risk_limits", {}))
        )
        self.market_data_manager = MarketDataManager()
        
        # Markets and strategies
        self.markets: Dict[str, Market] = {}
        self.strategies: List[Any] = []  # TradingStrategy instances
        
        # Threading and execution
        self._stop_event = threading.Event()
        self._threads: List[threading.Thread] = []
        self._order_queue = queue.Queue()
        
        # Performance tracking
        self.metrics = {
            "orders_processed": 0,
            "trades_executed": 0,
            "total_volume": 0.0,
            "total_pnl": 0.0,
            "uptime": 0.0
        }
        
        # Event callbacks
        self.callbacks = {
            "on_order_filled": [],
            "on_trade_executed": [],
            "on_risk_breach": [],
            "on_market_data": []
        }
    
    def start(self) -> bool:
        """
        Start the trading engine.
        
        Returns:
            True if started successfully, False otherwise
        """
        try:
            self.logger.info("Starting trading engine...")
            self.status = TradingEngineStatus.STARTING
            
            # Validate configuration
            if not self._validate_config():
                self.status = TradingEngineStatus.ERROR
                return False
            
            # Initialize components
            self._initialize_components()
            
            # Start worker threads
            self._start_threads()
            
            self.status = TradingEngineStatus.RUNNING
            self.start_time = datetime.now()
            self.logger.info("Trading engine started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start trading engine: {e}")
            self.status = TradingEngineStatus.ERROR
            return False
    
    def stop(self) -> bool:
        """
        Stop the trading engine.
        
        Returns:
            True if stopped successfully, False otherwise
        """
        try:
            self.logger.info("Stopping trading engine...")
            self.status = TradingEngineStatus.STOPPING
            
            # Signal stop to all threads
            self._stop_event.set()
            
            # Wait for threads to finish
            for thread in self._threads:
                thread.join(timeout=10.0)
            
            # Cancel pending orders
            self.order_manager.cancel_all_orders()
            
            self.status = TradingEngineStatus.STOPPED
            self.stop_time = datetime.now()
            self.logger.info("Trading engine stopped successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop trading engine: {e}")
            self.status = TradingEngineStatus.ERROR
            return False
    
    def submit_order(self, order: Order) -> bool:
        """
        Submit an order for execution.
        
        Args:
            order: Order to submit
            
        Returns:
            True if order was accepted, False otherwise
        """
        try:
            # Validate order
            if not self._validate_order(order):
                return False
            
            # Check risk limits
            if not self.risk_manager.check_order_risk(order, self.portfolio_manager.portfolio):
                self.logger.warning(f"Order {order.id} rejected due to risk limits")
                return False
            
            # Add to order queue
            self._order_queue.put(order)
            self.logger.info(f"Order {order.id} submitted: {order.order_type.value} {order.quantity} @ {order.price}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to submit order: {e}")
            return False
    
    def add_market(self, market: Market) -> None:
        """Add a market for trading."""
        self.markets[market.name] = market
        self.logger.info(f"Added market: {market.name}")
    
    def add_strategy(self, strategy: Any) -> None:
        """Add a trading strategy."""
        self.strategies.append(strategy)
        self.logger.info(f"Added strategy: {strategy.__class__.__name__}")
    
    def get_portfolio(self) -> Portfolio:
        """Get current portfolio."""
        return self.portfolio_manager.portfolio
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get trading engine metrics."""
        if self.start_time:
            self.metrics["uptime"] = (datetime.now() - self.start_time).total_seconds()
        return self.metrics.copy()
    
    def register_callback(self, event: str, callback: Callable) -> None:
        """Register event callback."""
        if event in self.callbacks:
            self.callbacks[event].append(callback)
    
    def _validate_config(self) -> bool:
        """Validate trading engine configuration."""
        # Add configuration validation logic
        return True
    
    def _validate_order(self, order: Order) -> bool:
        """Validate order parameters."""
        if order.quantity <= 0:
            self.logger.error(f"Invalid order quantity: {order.quantity}")
            return False
        
        if order.price <= 0:
            self.logger.error(f"Invalid order price: {order.price}")
            return False
        
        if order.market not in self.markets:
            self.logger.error(f"Unknown market: {order.market}")
            return False
        
        return True
    
    def _initialize_components(self) -> None:
        """Initialize trading engine components."""
        # Initialize market data manager
        data_config = self.config.get("data_provider", {})
        self.market_data_manager.initialize(data_config)
        
        # Initialize markets
        for market_name in self.config.get("markets", []):
            # Create market instances based on configuration
            pass
    
    def _start_threads(self) -> None:
        """Start worker threads."""
        # Order processing thread
        order_thread = threading.Thread(target=self._process_orders, daemon=True)
        order_thread.start()
        self._threads.append(order_thread)
        
        # Risk monitoring thread
        risk_thread = threading.Thread(target=self._monitor_risk, daemon=True)
        risk_thread.start()
        self._threads.append(risk_thread)
        
        # Strategy execution thread
        strategy_thread = threading.Thread(target=self._execute_strategies, daemon=True)
        strategy_thread.start()
        self._threads.append(strategy_thread)
        
        # Market data processing thread
        data_thread = threading.Thread(target=self._process_market_data, daemon=True)
        data_thread.start()
        self._threads.append(data_thread)
    
    def _process_orders(self) -> None:
        """Process orders from the queue."""
        while not self._stop_event.is_set():
            try:
                # Get order from queue with timeout
                order = self._order_queue.get(timeout=1.0)
                
                # Execute order
                self._execute_order(order)
                self.metrics["orders_processed"] += 1
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error processing order: {e}")
    
    def _execute_order(self, order: Order) -> None:
        """Execute a single order."""
        try:
            # Get market
            market = self.markets.get(order.market)
            if not market:
                order.status = OrderStatus.REJECTED
                return
            
            # Execute order in market
            execution_result = market.execute_order(order)
            
            if execution_result.get("success", False):
                # Create trade
                trade = Trade(
                    id=str(uuid.uuid4()),
                    order_id=order.id,
                    market=order.market,
                    side=order.side,
                    quantity=execution_result["quantity"],
                    price=execution_result["price"],
                    timestamp=datetime.now(),
                    fees=execution_result.get("fees", 0.0)
                )
                
                # Update portfolio
                self.portfolio_manager.add_trade(trade)
                
                # Update order status
                order.status = OrderStatus.FILLED
                order.filled_quantity = execution_result["quantity"]
                order.average_price = execution_result["price"]
                
                # Update metrics
                self.metrics["trades_executed"] += 1
                self.metrics["total_volume"] += trade.quantity
                
                # Trigger callbacks
                for callback in self.callbacks["on_trade_executed"]:
                    callback(trade)
                
                self.logger.info(f"Order {order.id} executed: {trade.quantity} @ {trade.price}")
            else:
                order.status = OrderStatus.REJECTED
                self.logger.warning(f"Order {order.id} rejected: {execution_result.get('reason', 'Unknown')}")
                
        except Exception as e:
            self.logger.error(f"Failed to execute order {order.id}: {e}")
            order.status = OrderStatus.REJECTED
    
    def _monitor_risk(self) -> None:
        """Monitor risk limits continuously."""
        while not self._stop_event.is_set():
            try:
                # Check risk limits
                risk_status = self.risk_manager.check_limits(self.portfolio_manager.portfolio)
                
                if risk_status.get("breach", False):
                    self.logger.warning(f"Risk limit breach: {risk_status}")
                    
                    # Trigger risk callbacks
                    for callback in self.callbacks["on_risk_breach"]:
                        callback(risk_status)
                
                # Sleep before next check
                time.sleep(self.risk_manager.limits.position_limit_check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in risk monitoring: {e}")
                time.sleep(60)  # Wait before retrying
    
    def _execute_strategies(self) -> None:
        """Execute trading strategies."""
        while not self._stop_event.is_set():
            try:
                for strategy in self.strategies:
                    # Get market data
                    market_data = self.market_data_manager.get_latest_data()
                    
                    # Execute strategy
                    signals = strategy.generate_signals(
                        market_data, 
                        self.portfolio_manager.portfolio
                    )
                    
                    # Process signals
                    for signal in signals:
                        if signal.get("action") == "buy":
                            order = self._create_buy_order(signal)
                            self.submit_order(order)
                        elif signal.get("action") == "sell":
                            order = self._create_sell_order(signal)
                            self.submit_order(order)
                
                # Sleep before next strategy execution
                time.sleep(60)  # Execute strategies every minute
                
            except Exception as e:
                self.logger.error(f"Error executing strategies: {e}")
                time.sleep(60)
    
    def _process_market_data(self) -> None:
        """Process incoming market data."""
        while not self._stop_event.is_set():
            try:
                # Get latest market data
                data = self.market_data_manager.get_updates()
                
                if data:
                    # Trigger market data callbacks
                    for callback in self.callbacks["on_market_data"]:
                        callback(data)
                
                time.sleep(1)  # Check for updates every second
                
            except Exception as e:
                self.logger.error(f"Error processing market data: {e}")
                time.sleep(5)
    
    def _create_buy_order(self, signal: Dict[str, Any]) -> Order:
        """Create a buy order from trading signal."""
        from .orders import MarketOrder, LimitOrder
        
        if signal.get("order_type") == "market":
            return MarketOrder(
                market=signal["market"],
                side="buy",
                quantity=signal["quantity"]
            )
        else:
            return LimitOrder(
                market=signal["market"],
                side="buy",
                quantity=signal["quantity"],
                price=signal["price"]
            )
    
    def _create_sell_order(self, signal: Dict[str, Any]) -> Order:
        """Create a sell order from trading signal."""
        from .orders import MarketOrder, LimitOrder
        
        if signal.get("order_type") == "market":
            return MarketOrder(
                market=signal["market"],
                side="sell",
                quantity=signal["quantity"]
            )
        else:
            return LimitOrder(
                market=signal["market"],
                side="sell",
                quantity=signal["quantity"],
                price=signal["price"]
            )


class OrderManager:
    """Manages order lifecycle and execution."""
    
    def __init__(self):
        """Initialize order manager."""
        self.orders: Dict[str, Order] = {}
        self.order_books: Dict[str, OrderBook] = {}
        self.logger = logging.getLogger("trading.orders")
    
    def add_order(self, order: Order) -> None:
        """Add order to management."""
        self.orders[order.id] = order
        self.logger.debug(f"Added order {order.id} to management")
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self.orders.get(order_id)
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        order = self.orders.get(order_id)
        if order and order.status in [OrderStatus.PENDING, OrderStatus.PARTIAL]:
            order.status = OrderStatus.CANCELLED
            self.logger.info(f"Cancelled order {order_id}")
            return True
        return False
    
    def cancel_all_orders(self) -> int:
        """Cancel all pending orders."""
        cancelled = 0
        for order in self.orders.values():
            if order.status in [OrderStatus.PENDING, OrderStatus.PARTIAL]:
                order.status = OrderStatus.CANCELLED
                cancelled += 1
        
        self.logger.info(f"Cancelled {cancelled} orders")
        return cancelled
    
    def get_active_orders(self) -> List[Order]:
        """Get all active orders."""
        return [order for order in self.orders.values() 
                if order.status in [OrderStatus.PENDING, OrderStatus.PARTIAL]]


class PortfolioManager:
    """Manages portfolio positions and trades."""
    
    def __init__(self):
        """Initialize portfolio manager."""
        self.portfolio = Portfolio()
        self.trades: List[Trade] = []
        self.logger = logging.getLogger("trading.portfolio")
    
    def add_trade(self, trade: Trade) -> None:
        """Add a trade to the portfolio."""
        self.trades.append(trade)
        self.portfolio.update_from_trade(trade)
        self.logger.info(f"Added trade: {trade.side} {trade.quantity} @ {trade.price}")
    
    def get_position(self, market: str) -> Optional[Position]:
        """Get position for a market."""
        return self.portfolio.positions.get(market)
    
    def get_total_pnl(self) -> float:
        """Get total portfolio P&L."""
        return self.portfolio.calculate_total_pnl()
    
    def get_trades(self, market: str = None) -> List[Trade]:
        """Get trades, optionally filtered by market."""
        if market:
            return [trade for trade in self.trades if trade.market == market]
        return self.trades.copy()


class RiskManager:
    """Manages trading risk and limits."""
    
    def __init__(self, limits: RiskLimits):
        """Initialize risk manager."""
        self.limits = limits
        self.logger = logging.getLogger("trading.risk")
    
    def check_order_risk(self, order: Order, portfolio: Portfolio) -> bool:
        """Check if order violates risk limits."""
        # Check position limits
        current_position = portfolio.positions.get(order.market)
        if current_position:
            new_position_size = abs(current_position.quantity + 
                                  (order.quantity if order.side == "buy" else -order.quantity))
            if new_position_size > self.limits.max_position:
                self.logger.warning(f"Order would exceed position limit: {new_position_size} > {self.limits.max_position}")
                return False
        
        # Check daily loss limits
        daily_pnl = portfolio.calculate_daily_pnl()
        if daily_pnl < -self.limits.max_daily_loss:
            self.logger.warning(f"Daily loss limit exceeded: {daily_pnl} < {-self.limits.max_daily_loss}")
            return False
        
        return True
    
    def check_limits(self, portfolio: Portfolio) -> Dict[str, Any]:
        """Check all risk limits."""
        breaches = []
        
        # Check position limits
        for market, position in portfolio.positions.items():
            if abs(position.quantity) > self.limits.max_position:
                breaches.append(f"Position limit exceeded in {market}: {abs(position.quantity)} > {self.limits.max_position}")
        
        # Check daily loss
        daily_pnl = portfolio.calculate_daily_pnl()
        if daily_pnl < -self.limits.max_daily_loss:
            breaches.append(f"Daily loss limit exceeded: {daily_pnl} < {-self.limits.max_daily_loss}")
        
        # Check drawdown
        max_drawdown = portfolio.calculate_max_drawdown()
        if max_drawdown > self.limits.max_drawdown:
            breaches.append(f"Drawdown limit exceeded: {max_drawdown} > {self.limits.max_drawdown}")
        
        return {
            "breach": len(breaches) > 0,
            "breaches": breaches,
            "daily_pnl": daily_pnl,
            "max_drawdown": max_drawdown
        }


class MarketDataManager:
    """Manages market data feeds and distribution."""
    
    def __init__(self):
        """Initialize market data manager."""
        self.providers: Dict[str, MarketDataProvider] = {}
        self.latest_data: Dict[str, MarketData] = {}
        self.logger = logging.getLogger("trading.data")
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize market data providers."""
        provider_type = config.get("type", "simulated")
        
        if provider_type == "simulated":
            from .data import SimulatedDataProvider
            provider = SimulatedDataProvider()
            self.providers["default"] = provider
        elif provider_type == "live":
            from .data import LiveDataProvider
            provider = LiveDataProvider(config)
            self.providers["default"] = provider
        
        self.logger.info(f"Initialized {provider_type} data provider")
    
    def get_latest_data(self) -> Dict[str, MarketData]:
        """Get latest market data."""
        return self.latest_data.copy()
    
    def get_updates(self) -> Optional[Dict[str, MarketData]]:
        """Get market data updates."""
        updates = {}
        for name, provider in self.providers.items():
            data = provider.get_latest()
            if data:
                self.latest_data[name] = data
                updates[name] = data
        
        return updates if updates else None
