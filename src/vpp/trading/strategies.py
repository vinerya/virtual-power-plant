"""
Trading strategies for VPP energy market participation.

This module provides various trading strategies including arbitrage,
momentum, mean reversion, and machine learning-based approaches.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import logging

from .portfolio import Portfolio
from .orders import create_order


class TradingStrategy(ABC):
    """Base class for trading strategies."""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        """Initialize trading strategy."""
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"trading.strategy.{name}")
        
        # Strategy state
        self.is_active = True
        self.last_signal_time: Optional[datetime] = None
        
        # Performance tracking
        self.signals_generated = 0
        self.trades_executed = 0
        self.total_pnl = 0.0
        
        # Risk management
        self.max_position_size = self.config.get("max_position_size", 1000.0)
        self.max_daily_trades = self.config.get("max_daily_trades", 10)
        self.stop_loss_pct = self.config.get("stop_loss_pct", 0.05)  # 5%
        
        # Strategy parameters
        self.lookback_period = self.config.get("lookback_period", 24)  # hours
        self.signal_threshold = self.config.get("signal_threshold", 0.02)  # 2%
        self.min_profit_threshold = self.config.get("min_profit_threshold", 0.01)  # 1%
    
    @abstractmethod
    def generate_signals(self, market_data: Dict[str, Any], 
                        portfolio: Portfolio) -> List[Dict[str, Any]]:
        """Generate trading signals based on market data and portfolio state."""
        pass
    
    def validate_signal(self, signal: Dict[str, Any], portfolio: Portfolio) -> bool:
        """Validate trading signal against risk limits."""
        # Check if strategy is active
        if not self.is_active:
            return False
        
        # Check position size limits
        market = signal.get("market")
        quantity = signal.get("quantity", 0)
        
        if market and quantity > 0:
            current_position = portfolio.get_position(market)
            current_quantity = current_position.quantity if current_position else 0
            
            if signal.get("action") == "buy":
                new_quantity = current_quantity + quantity
            else:  # sell
                new_quantity = current_quantity - quantity
            
            if abs(new_quantity) > self.max_position_size:
                self.logger.warning(f"Signal rejected: position size limit exceeded")
                return False
        
        # Check daily trade limits
        today = datetime.now().date()
        daily_trades = sum(1 for trade in portfolio.trades 
                          if trade.timestamp.date() == today and 
                          trade.strategy == self.name)
        
        if daily_trades >= self.max_daily_trades:
            self.logger.warning(f"Signal rejected: daily trade limit exceeded")
            return False
        
        return True
    
    def calculate_position_size(self, signal: Dict[str, Any], 
                              portfolio: Portfolio) -> float:
        """Calculate appropriate position size for signal."""
        base_quantity = signal.get("quantity", 100.0)
        confidence = signal.get("confidence", 0.5)
        
        # Adjust size based on confidence
        adjusted_quantity = base_quantity * confidence
        
        # Apply position size limits
        return min(adjusted_quantity, self.max_position_size)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get strategy performance metrics."""
        return {
            "name": self.name,
            "signals_generated": self.signals_generated,
            "trades_executed": self.trades_executed,
            "total_pnl": self.total_pnl,
            "signal_to_trade_ratio": self.trades_executed / max(1, self.signals_generated),
            "avg_pnl_per_trade": self.total_pnl / max(1, self.trades_executed),
            "is_active": self.is_active,
            "last_signal_time": self.last_signal_time
        }


class ArbitrageStrategy(TradingStrategy):
    """Arbitrage strategy for price differences between markets."""
    
    def __init__(self, price_threshold: float = 0.02, **kwargs):
        """Initialize arbitrage strategy."""
        super().__init__("arbitrage", kwargs)
        self.price_threshold = price_threshold
        self.transaction_cost = kwargs.get("transaction_cost", 0.001)
        self.markets_to_monitor = kwargs.get("markets", ["day_ahead", "real_time"])
    
    def generate_signals(self, market_data: Dict[str, Any], 
                        portfolio: Portfolio) -> List[Dict[str, Any]]:
        """Generate arbitrage signals based on price differences."""
        signals = []
        
        try:
            # Get current prices for monitored markets
            prices = {}
            for market_name in self.markets_to_monitor:
                if market_name in market_data:
                    data = market_data[market_name]
                    if hasattr(data, 'last_price') and data.last_price is not None:
                        prices[market_name] = data.last_price
                    elif hasattr(data, 'clearing_price') and data.clearing_price is not None:
                        prices[market_name] = data.clearing_price
            
            if len(prices) < 2:
                return signals
            
            # Find arbitrage opportunities
            market_pairs = [(m1, m2) for m1 in prices.keys() for m2 in prices.keys() if m1 != m2]
            
            for market1, market2 in market_pairs:
                price1 = prices[market1]
                price2 = prices[market2]
                
                price_diff = abs(price2 - price1)
                total_cost = 2 * self.transaction_cost  # Buy and sell costs
                
                if price_diff > total_cost + self.price_threshold:
                    # Profitable arbitrage opportunity
                    buy_market = market1 if price1 < price2 else market2
                    sell_market = market2 if price1 < price2 else market1
                    buy_price = min(price1, price2)
                    sell_price = max(price1, price2)
                    
                    profit_per_unit = price_diff - total_cost
                    confidence = min(1.0, profit_per_unit / self.price_threshold)
                    
                    # Calculate optimal quantity
                    base_quantity = self.config.get("base_quantity", 100.0)
                    quantity = self.calculate_position_size({
                        "quantity": base_quantity,
                        "confidence": confidence
                    }, portfolio)
                    
                    # Generate buy signal
                    buy_signal = {
                        "strategy": self.name,
                        "action": "buy",
                        "market": buy_market,
                        "quantity": quantity,
                        "price": buy_price,
                        "order_type": "limit",
                        "confidence": confidence,
                        "expected_profit": profit_per_unit * quantity,
                        "arbitrage_pair": sell_market,
                        "timestamp": datetime.now()
                    }
                    
                    # Generate sell signal
                    sell_signal = {
                        "strategy": self.name,
                        "action": "sell",
                        "market": sell_market,
                        "quantity": quantity,
                        "price": sell_price,
                        "order_type": "limit",
                        "confidence": confidence,
                        "expected_profit": profit_per_unit * quantity,
                        "arbitrage_pair": buy_market,
                        "timestamp": datetime.now()
                    }
                    
                    if self.validate_signal(buy_signal, portfolio):
                        signals.append(buy_signal)
                    
                    if self.validate_signal(sell_signal, portfolio):
                        signals.append(sell_signal)
                    
                    self.logger.info(f"Arbitrage opportunity: {buy_market} @ {buy_price:.3f} -> "
                                   f"{sell_market} @ {sell_price:.3f}, profit: {profit_per_unit:.3f}")
            
            self.signals_generated += len(signals)
            if signals:
                self.last_signal_time = datetime.now()
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating arbitrage signals: {e}")
            return []


class MomentumStrategy(TradingStrategy):
    """Momentum strategy based on price trends."""
    
    def __init__(self, lookback_hours: int = 4, momentum_threshold: float = 0.05, **kwargs):
        """Initialize momentum strategy."""
        super().__init__("momentum", kwargs)
        self.lookback_hours = lookback_hours
        self.momentum_threshold = momentum_threshold
        self.price_history: Dict[str, List[Tuple[datetime, float]]] = {}
    
    def generate_signals(self, market_data: Dict[str, Any], 
                        portfolio: Portfolio) -> List[Dict[str, Any]]:
        """Generate momentum signals based on price trends."""
        signals = []
        
        try:
            current_time = datetime.now()
            
            # Update price history
            for market_name, data in market_data.items():
                if market_name not in self.price_history:
                    self.price_history[market_name] = []
                
                current_price = None
                if hasattr(data, 'last_price') and data.last_price is not None:
                    current_price = data.last_price
                elif hasattr(data, 'clearing_price') and data.clearing_price is not None:
                    current_price = data.clearing_price
                
                if current_price is not None:
                    self.price_history[market_name].append((current_time, current_price))
                    
                    # Keep only recent history
                    cutoff_time = current_time - timedelta(hours=self.lookback_hours * 2)
                    self.price_history[market_name] = [
                        (t, p) for t, p in self.price_history[market_name] 
                        if t >= cutoff_time
                    ]
            
            # Generate momentum signals
            for market_name, price_data in self.price_history.items():
                if len(price_data) < 2:
                    continue
                
                # Calculate momentum
                recent_prices = [p for t, p in price_data 
                               if t >= current_time - timedelta(hours=self.lookback_hours)]
                
                if len(recent_prices) < 2:
                    continue
                
                price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
                momentum = price_change
                
                # Generate signal if momentum exceeds threshold
                if abs(momentum) > self.momentum_threshold:
                    action = "buy" if momentum > 0 else "sell"
                    confidence = min(1.0, abs(momentum) / self.momentum_threshold)
                    
                    # Calculate position size
                    base_quantity = self.config.get("base_quantity", 100.0)
                    quantity = self.calculate_position_size({
                        "quantity": base_quantity,
                        "confidence": confidence
                    }, portfolio)
                    
                    signal = {
                        "strategy": self.name,
                        "action": action,
                        "market": market_name,
                        "quantity": quantity,
                        "price": recent_prices[-1],
                        "order_type": "market",
                        "confidence": confidence,
                        "momentum": momentum,
                        "lookback_hours": self.lookback_hours,
                        "timestamp": current_time
                    }
                    
                    if self.validate_signal(signal, portfolio):
                        signals.append(signal)
                        self.logger.info(f"Momentum signal: {action} {market_name}, "
                                       f"momentum: {momentum:.3f}, confidence: {confidence:.2f}")
            
            self.signals_generated += len(signals)
            if signals:
                self.last_signal_time = current_time
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating momentum signals: {e}")
            return []


class MeanReversionStrategy(TradingStrategy):
    """Mean reversion strategy based on price deviations from moving average."""
    
    def __init__(self, lookback_hours: int = 24, deviation_threshold: float = 2.0, **kwargs):
        """Initialize mean reversion strategy."""
        super().__init__("mean_reversion", kwargs)
        self.lookback_hours = lookback_hours
        self.deviation_threshold = deviation_threshold  # Standard deviations
        self.price_history: Dict[str, List[Tuple[datetime, float]]] = {}
    
    def generate_signals(self, market_data: Dict[str, Any], 
                        portfolio: Portfolio) -> List[Dict[str, Any]]:
        """Generate mean reversion signals."""
        signals = []
        
        try:
            current_time = datetime.now()
            
            # Update price history
            for market_name, data in market_data.items():
                if market_name not in self.price_history:
                    self.price_history[market_name] = []
                
                current_price = None
                if hasattr(data, 'last_price') and data.last_price is not None:
                    current_price = data.last_price
                elif hasattr(data, 'clearing_price') and data.clearing_price is not None:
                    current_price = data.clearing_price
                
                if current_price is not None:
                    self.price_history[market_name].append((current_time, current_price))
                    
                    # Keep only recent history
                    cutoff_time = current_time - timedelta(hours=self.lookback_hours * 2)
                    self.price_history[market_name] = [
                        (t, p) for t, p in self.price_history[market_name] 
                        if t >= cutoff_time
                    ]
            
            # Generate mean reversion signals
            for market_name, price_data in self.price_history.items():
                if len(price_data) < 10:  # Need sufficient history
                    continue
                
                # Calculate moving average and standard deviation
                recent_prices = [p for t, p in price_data 
                               if t >= current_time - timedelta(hours=self.lookback_hours)]
                
                if len(recent_prices) < 10:
                    continue
                
                mean_price = np.mean(recent_prices)
                std_price = np.std(recent_prices)
                current_price = recent_prices[-1]
                
                if std_price == 0:
                    continue
                
                # Calculate z-score
                z_score = (current_price - mean_price) / std_price
                
                # Generate signal if price deviates significantly from mean
                if abs(z_score) > self.deviation_threshold:
                    # Price is too high -> sell, price is too low -> buy
                    action = "sell" if z_score > 0 else "buy"
                    confidence = min(1.0, abs(z_score) / self.deviation_threshold)
                    
                    # Calculate position size
                    base_quantity = self.config.get("base_quantity", 100.0)
                    quantity = self.calculate_position_size({
                        "quantity": base_quantity,
                        "confidence": confidence
                    }, portfolio)
                    
                    # Target price for mean reversion
                    target_price = mean_price
                    
                    signal = {
                        "strategy": self.name,
                        "action": action,
                        "market": market_name,
                        "quantity": quantity,
                        "price": target_price,
                        "order_type": "limit",
                        "confidence": confidence,
                        "z_score": z_score,
                        "mean_price": mean_price,
                        "current_price": current_price,
                        "timestamp": current_time
                    }
                    
                    if self.validate_signal(signal, portfolio):
                        signals.append(signal)
                        self.logger.info(f"Mean reversion signal: {action} {market_name}, "
                                       f"z-score: {z_score:.2f}, target: {target_price:.3f}")
            
            self.signals_generated += len(signals)
            if signals:
                self.last_signal_time = current_time
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating mean reversion signals: {e}")
            return []


class MLTradingStrategy(TradingStrategy):
    """Machine learning-based trading strategy."""
    
    def __init__(self, model_path: str = None, **kwargs):
        """Initialize ML trading strategy."""
        super().__init__("ml_trading", kwargs)
        self.model_path = model_path
        self.model = None
        self.feature_history: Dict[str, List[Dict[str, float]]] = {}
        self.prediction_threshold = kwargs.get("prediction_threshold", 0.6)
        
        # Try to load model
        if model_path:
            self._load_model()
    
    def _load_model(self) -> bool:
        """Load ML model from file."""
        try:
            # This is a placeholder - in practice you'd load your trained model
            # import joblib
            # self.model = joblib.load(self.model_path)
            self.logger.info(f"ML model loaded from {self.model_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load ML model: {e}")
            return False
    
    def _extract_features(self, market_data: Dict[str, Any], 
                         portfolio: Portfolio) -> Dict[str, float]:
        """Extract features for ML model."""
        features = {}
        
        try:
            # Price-based features
            for market_name, data in market_data.items():
                if hasattr(data, 'last_price') and data.last_price is not None:
                    features[f"{market_name}_price"] = data.last_price
                
                if hasattr(data, 'volume'):
                    features[f"{market_name}_volume"] = data.volume
                
                if hasattr(data, 'bid_price') and hasattr(data, 'ask_price'):
                    if data.bid_price and data.ask_price:
                        features[f"{market_name}_spread"] = data.ask_price - data.bid_price
            
            # Portfolio features
            total_positions = len(portfolio.positions)
            total_pnl = portfolio.calculate_total_pnl()
            
            features["portfolio_positions"] = total_positions
            features["portfolio_pnl"] = total_pnl
            features["portfolio_cash"] = portfolio.cash
            
            # Time-based features
            now = datetime.now()
            features["hour_of_day"] = now.hour
            features["day_of_week"] = now.weekday()
            features["month"] = now.month
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {e}")
            return {}
    
    def _predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Make prediction using ML model."""
        if self.model is None:
            # Fallback to simple heuristic
            return self._simple_heuristic_prediction(features)
        
        try:
            # This is a placeholder - implement your model prediction logic
            # feature_vector = np.array([features.get(key, 0) for key in expected_features])
            # prediction = self.model.predict(feature_vector.reshape(1, -1))[0]
            # confidence = self.model.predict_proba(feature_vector.reshape(1, -1)).max()
            
            # For now, return a simple prediction
            return self._simple_heuristic_prediction(features)
            
        except Exception as e:
            self.logger.error(f"Error making ML prediction: {e}")
            return {"action": "hold", "confidence": 0.0}
    
    def _simple_heuristic_prediction(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Simple heuristic prediction as fallback."""
        # Simple rule: buy when prices are low, sell when high
        day_ahead_price = features.get("day_ahead_price", 0.1)
        real_time_price = features.get("real_time_price", 0.1)
        
        if day_ahead_price > 0 and real_time_price > 0:
            price_ratio = real_time_price / day_ahead_price
            
            if price_ratio > 1.05:  # Real-time 5% higher
                return {"action": "sell", "market": "real_time", "confidence": 0.7}
            elif price_ratio < 0.95:  # Real-time 5% lower
                return {"action": "buy", "market": "real_time", "confidence": 0.7}
        
        return {"action": "hold", "confidence": 0.5}
    
    def generate_signals(self, market_data: Dict[str, Any], 
                        portfolio: Portfolio) -> List[Dict[str, Any]]:
        """Generate ML-based trading signals."""
        signals = []
        
        try:
            # Extract features
            features = self._extract_features(market_data, portfolio)
            
            if not features:
                return signals
            
            # Store feature history
            current_time = datetime.now()
            for market_name in market_data.keys():
                if market_name not in self.feature_history:
                    self.feature_history[market_name] = []
                
                self.feature_history[market_name].append({
                    "timestamp": current_time,
                    **features
                })
                
                # Keep limited history
                if len(self.feature_history[market_name]) > 1000:
                    self.feature_history[market_name] = self.feature_history[market_name][-1000:]
            
            # Make prediction
            prediction = self._predict(features)
            
            if prediction["confidence"] > self.prediction_threshold:
                action = prediction["action"]
                
                if action in ["buy", "sell"]:
                    market = prediction.get("market", "real_time")
                    confidence = prediction["confidence"]
                    
                    # Calculate position size
                    base_quantity = self.config.get("base_quantity", 100.0)
                    quantity = self.calculate_position_size({
                        "quantity": base_quantity,
                        "confidence": confidence
                    }, portfolio)
                    
                    # Get current price
                    current_price = features.get(f"{market}_price", 0.1)
                    
                    signal = {
                        "strategy": self.name,
                        "action": action,
                        "market": market,
                        "quantity": quantity,
                        "price": current_price,
                        "order_type": "market",
                        "confidence": confidence,
                        "ml_prediction": prediction,
                        "timestamp": current_time
                    }
                    
                    if self.validate_signal(signal, portfolio):
                        signals.append(signal)
                        self.logger.info(f"ML signal: {action} {market}, "
                                       f"confidence: {confidence:.2f}")
            
            self.signals_generated += len(signals)
            if signals:
                self.last_signal_time = current_time
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating ML signals: {e}")
            return []


class MultiMarketStrategy(TradingStrategy):
    """Multi-market strategy that combines multiple approaches."""
    
    def __init__(self, strategies: List[TradingStrategy] = None, **kwargs):
        """Initialize multi-market strategy."""
        super().__init__("multi_market", kwargs)
        self.strategies = strategies or []
        self.strategy_weights = kwargs.get("strategy_weights", {})
        self.signal_aggregation = kwargs.get("signal_aggregation", "weighted_average")
    
    def add_strategy(self, strategy: TradingStrategy, weight: float = 1.0) -> None:
        """Add a strategy to the multi-market approach."""
        self.strategies.append(strategy)
        self.strategy_weights[strategy.name] = weight
        self.logger.info(f"Added strategy {strategy.name} with weight {weight}")
    
    def generate_signals(self, market_data: Dict[str, Any], 
                        portfolio: Portfolio) -> List[Dict[str, Any]]:
        """Generate signals by combining multiple strategies."""
        all_signals = []
        
        try:
            # Collect signals from all strategies
            strategy_signals = {}
            for strategy in self.strategies:
                if strategy.is_active:
                    signals = strategy.generate_signals(market_data, portfolio)
                    strategy_signals[strategy.name] = signals
                    all_signals.extend(signals)
            
            # Aggregate signals if multiple strategies target same market
            if self.signal_aggregation == "weighted_average":
                aggregated_signals = self._aggregate_signals_weighted(strategy_signals)
            elif self.signal_aggregation == "majority_vote":
                aggregated_signals = self._aggregate_signals_majority(strategy_signals)
            else:
                aggregated_signals = all_signals
            
            # Validate aggregated signals
            final_signals = []
            for signal in aggregated_signals:
                if self.validate_signal(signal, portfolio):
                    final_signals.append(signal)
            
            self.signals_generated += len(final_signals)
            if final_signals:
                self.last_signal_time = datetime.now()
            
            return final_signals
            
        except Exception as e:
            self.logger.error(f"Error generating multi-market signals: {e}")
            return []
    
    def _aggregate_signals_weighted(self, strategy_signals: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Aggregate signals using weighted average."""
        market_signals = {}
        
        # Group signals by market
        for strategy_name, signals in strategy_signals.items():
            weight = self.strategy_weights.get(strategy_name, 1.0)
            
            for signal in signals:
                market = signal.get("market")
                if market not in market_signals:
                    market_signals[market] = []
                
                signal["strategy_weight"] = weight
                market_signals[market].append(signal)
        
        # Aggregate signals for each market
        aggregated = []
        for market, signals in market_signals.items():
            if len(signals) == 1:
                aggregated.append(signals[0])
            else:
                # Weighted average of signals
                total_weight = sum(s["strategy_weight"] for s in signals)
                
                # Calculate weighted averages
                avg_confidence = sum(s["confidence"] * s["strategy_weight"] for s in signals) / total_weight
                avg_quantity = sum(s["quantity"] * s["strategy_weight"] for s in signals) / total_weight
                
                # Determine action by majority vote weighted by confidence
                buy_weight = sum(s["strategy_weight"] * s["confidence"] for s in signals if s["action"] == "buy")
                sell_weight = sum(s["strategy_weight"] * s["confidence"] for s in signals if s["action"] == "sell")
                
                if buy_weight > sell_weight:
                    action = "buy"
                elif sell_weight > buy_weight:
                    action = "sell"
                else:
                    continue  # No clear signal
                
                # Create aggregated signal
                aggregated_signal = {
                    "strategy": self.name,
                    "action": action,
                    "market": market,
                    "quantity": avg_quantity,
                    "price": signals[0]["price"],  # Use first signal's price
                    "order_type": "limit",
                    "confidence": avg_confidence,
                    "contributing_strategies": [s["strategy"] for s in signals],
                    "timestamp": datetime.now()
                }
                
                aggregated.append(aggregated_signal)
        
        return aggregated
    
    def _aggregate_signals_majority(self, strategy_signals: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Aggregate signals using majority vote."""
        market_signals = {}
        
        # Group signals by market
        for strategy_name, signals in strategy_signals.items():
            for signal in signals:
                market = signal.get("market")
                if market not in market_signals:
                    market_signals[market] = []
                market_signals[market].append(signal)
        
        # Majority vote for each market
        aggregated = []
        for market, signals in market_signals.items():
            buy_votes = sum(1 for s in signals if s["action"] == "buy")
            sell_votes = sum(1 for s in signals if s["action"] == "sell")
            
            if buy_votes > sell_votes:
                action = "buy"
                relevant_signals = [s for s in signals if s["action"] == "buy"]
            elif sell_votes > buy_votes:
                action = "sell"
                relevant_signals = [s for s in signals if s["action"] == "sell"]
            else:
                continue  # Tie, no signal
            
            # Average the relevant signals
            avg_confidence = np.mean([s["confidence"] for s in relevant_signals])
            avg_quantity = np.mean([s["quantity"] for s in relevant_signals])
            
            aggregated_signal = {
                "strategy": self.name,
                "action": action,
                "market": market,
                "quantity": avg_quantity,
                "price": relevant_signals[0]["price"],
                "order_type": "limit",
                "confidence": avg_confidence,
                "vote_count": len(relevant_signals),
                "timestamp": datetime.now()
            }
            
            aggregated.append(aggregated_signal)
        
        return aggregated


def create_strategy(strategy_type: str, **kwargs) -> TradingStrategy:
    """
    Factory function to create trading strategies.
    
    Args:
        strategy_type: Type of strategy to create
        **kwargs: Strategy configuration parameters
        
    Returns:
        TradingStrategy instance
        
    Example:
        >>> strategy = create_strategy("arbitrage", price_threshold=0.03)
        >>> strategy = create_strategy("momentum", lookback_hours=6)
    """
    strategy_type = strategy_type.lower()
    
    if strategy_type == "arbitrage":
        return ArbitrageStrategy(**kwargs)
    elif strategy_type == "momentum":
        return MomentumStrategy(**kwargs)
    elif strategy_type == "mean_reversion":
        return MeanReversionStrategy(**kwargs)
    elif strategy_type == "ml" or strategy_type == "machine_learning":
        return MLTradingStrategy(**kwargs)
    elif strategy_type == "multi_market":
        return MultiMarketStrategy(**kwargs)
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")
