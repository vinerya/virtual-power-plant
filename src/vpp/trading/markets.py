"""
Energy market implementations for VPP trading.

This module provides different energy market types including day-ahead,
real-time, ancillary services, and bilateral markets.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import logging

from .orders import Order, OrderStatus, OrderBook


class MarketType(Enum):
    """Energy market types."""
    DAY_AHEAD = "day_ahead"
    REAL_TIME = "real_time"
    ANCILLARY_SERVICES = "ancillary_services"
    BILATERAL = "bilateral"
    CAPACITY = "capacity"
    EMISSIONS = "emissions"


class MarketStatus(Enum):
    """Market operational status."""
    CLOSED = "closed"
    PRE_OPEN = "pre_open"
    OPEN = "open"
    AUCTION = "auction"
    POST_CLOSE = "post_close"
    SUSPENDED = "suspended"


@dataclass
class MarketSession:
    """Market trading session definition."""
    name: str
    start_time: datetime
    end_time: datetime
    market_type: MarketType
    auction_time: Optional[datetime] = None
    settlement_time: Optional[datetime] = None


@dataclass
class MarketData:
    """Market data snapshot."""
    market: str
    timestamp: datetime
    bid_price: Optional[float] = None
    ask_price: Optional[float] = None
    last_price: Optional[float] = None
    volume: float = 0.0
    high: Optional[float] = None
    low: Optional[float] = None
    open_price: Optional[float] = None
    close_price: Optional[float] = None
    
    # Market depth
    bid_levels: List[Tuple[float, float]] = field(default_factory=list)  # (price, quantity)
    ask_levels: List[Tuple[float, float]] = field(default_factory=list)
    
    # Additional market info
    total_demand: float = 0.0
    total_supply: float = 0.0
    clearing_price: Optional[float] = None
    system_lambda: Optional[float] = None  # Marginal price


class Market(ABC):
    """Base class for energy markets."""
    
    def __init__(self, name: str, market_type: MarketType):
        """Initialize market."""
        self.name = name
        self.market_type = market_type
        self.status = MarketStatus.CLOSED
        self.logger = logging.getLogger(f"trading.market.{name}")
        
        # Market configuration
        self.tick_size = 0.01  # Minimum price increment
        self.lot_size = 1.0    # Minimum quantity increment
        self.max_price = 1000.0  # Maximum allowed price
        self.min_price = 0.0     # Minimum allowed price
        
        # Order book
        self.order_book = OrderBook(name)
        
        # Market data
        self.current_data = MarketData(market=name, timestamp=datetime.now())
        self.price_history: List[float] = []
        self.volume_history: List[float] = []
        
        # Trading sessions
        self.sessions: List[MarketSession] = []
        self.current_session: Optional[MarketSession] = None
        
        # Market participants
        self.participants: Dict[str, Dict[str, Any]] = {}
        
        # Fees and costs
        self.transaction_fee = 0.001  # $/kWh
        self.market_fee = 0.0005     # $/kWh
    
    @abstractmethod
    def execute_order(self, order: Order) -> Dict[str, Any]:
        """Execute an order in this market."""
        pass
    
    @abstractmethod
    def calculate_clearing_price(self) -> Optional[float]:
        """Calculate market clearing price."""
        pass
    
    def add_session(self, session: MarketSession) -> None:
        """Add a trading session."""
        self.sessions.append(session)
        self.logger.info(f"Added session {session.name} from {session.start_time} to {session.end_time}")
    
    def get_current_session(self) -> Optional[MarketSession]:
        """Get current active session."""
        now = datetime.now()
        for session in self.sessions:
            if session.start_time <= now <= session.end_time:
                return session
        return None
    
    def is_market_open(self) -> bool:
        """Check if market is currently open for trading."""
        return self.status == MarketStatus.OPEN and self.get_current_session() is not None
    
    def update_market_data(self, data: MarketData) -> None:
        """Update market data."""
        self.current_data = data
        
        if data.last_price is not None:
            self.price_history.append(data.last_price)
        
        if data.volume > 0:
            self.volume_history.append(data.volume)
        
        # Limit history size
        max_history = 1000
        if len(self.price_history) > max_history:
            self.price_history = self.price_history[-max_history:]
        if len(self.volume_history) > max_history:
            self.volume_history = self.volume_history[-max_history:]
    
    def get_market_depth(self, levels: int = 5) -> Dict[str, Any]:
        """Get market depth information."""
        return self.order_book.get_market_depth(levels)
    
    def get_spread(self) -> Optional[float]:
        """Get current bid-ask spread."""
        return self.order_book.get_spread()
    
    def validate_order(self, order: Order) -> List[str]:
        """Validate order parameters for this market."""
        errors = []
        
        # Check market is open
        if not self.is_market_open():
            errors.append(f"Market {self.name} is not open for trading")
        
        # Check price limits
        if order.price > 0:
            if order.price > self.max_price:
                errors.append(f"Price {order.price} exceeds maximum {self.max_price}")
            if order.price < self.min_price:
                errors.append(f"Price {order.price} below minimum {self.min_price}")
            
            # Check tick size
            if (order.price % self.tick_size) != 0:
                errors.append(f"Price {order.price} not aligned to tick size {self.tick_size}")
        
        # Check quantity
        if (order.quantity % self.lot_size) != 0:
            errors.append(f"Quantity {order.quantity} not aligned to lot size {self.lot_size}")
        
        return errors


class DayAheadMarket(Market):
    """Day-ahead energy market with auction clearing."""
    
    def __init__(self, name: str = "day_ahead"):
        """Initialize day-ahead market."""
        super().__init__(name, MarketType.DAY_AHEAD)
        
        # Day-ahead specific configuration
        self.auction_time = "12:00"  # Daily auction time
        self.delivery_periods = 24   # 24 hourly periods
        self.gate_closure_minutes = 60  # Gate closes 60 minutes before auction
        
        # Bid/offer tracking
        self.supply_bids: Dict[int, List[Tuple[float, float]]] = {}  # period -> [(price, quantity)]
        self.demand_bids: Dict[int, List[Tuple[float, float]]] = {}
        
        # Clearing results
        self.clearing_prices: Dict[int, float] = {}  # period -> price
        self.cleared_volumes: Dict[int, float] = {}  # period -> volume
    
    def execute_order(self, order: Order) -> Dict[str, Any]:
        """Execute order in day-ahead market (add to auction)."""
        try:
            # Validate order
            validation_errors = self.validate_order(order)
            if validation_errors:
                return {
                    "success": False,
                    "reason": "; ".join(validation_errors)
                }
            
            # Add to appropriate bid list
            delivery_period = order.metadata.get("delivery_period", 1)
            
            if order.side == "buy":
                if delivery_period not in self.demand_bids:
                    self.demand_bids[delivery_period] = []
                self.demand_bids[delivery_period].append((order.price, order.quantity))
            else:  # sell
                if delivery_period not in self.supply_bids:
                    self.supply_bids[delivery_period] = []
                self.supply_bids[delivery_period].append((order.price, order.quantity))
            
            order.status = OrderStatus.PENDING
            
            return {
                "success": True,
                "order_id": order.id,
                "status": "pending_auction",
                "delivery_period": delivery_period
            }
            
        except Exception as e:
            self.logger.error(f"Failed to execute order {order.id}: {e}")
            return {
                "success": False,
                "reason": str(e)
            }
    
    def calculate_clearing_price(self) -> Optional[float]:
        """Calculate clearing price for current period."""
        # This is a simplified clearing algorithm
        # In practice, this would be much more sophisticated
        
        current_period = datetime.now().hour + 1
        
        if (current_period not in self.supply_bids or 
            current_period not in self.demand_bids):
            return None
        
        # Sort supply bids (ascending price) and demand bids (descending price)
        supply = sorted(self.supply_bids[current_period])
        demand = sorted(self.demand_bids[current_period], reverse=True)
        
        # Find intersection
        supply_cumulative = 0
        demand_cumulative = 0
        
        for i, (supply_price, supply_qty) in enumerate(supply):
            supply_cumulative += supply_qty
            
            for j, (demand_price, demand_qty) in enumerate(demand):
                if j == 0:
                    demand_cumulative = demand_qty
                else:
                    demand_cumulative += demand_qty
                
                if supply_cumulative >= demand_cumulative and supply_price <= demand_price:
                    # Found clearing point
                    clearing_price = (supply_price + demand_price) / 2
                    self.clearing_prices[current_period] = clearing_price
                    self.cleared_volumes[current_period] = min(supply_cumulative, demand_cumulative)
                    return clearing_price
        
        return None
    
    def run_auction(self, period: int) -> Dict[str, Any]:
        """Run auction for specific delivery period."""
        self.logger.info(f"Running auction for period {period}")
        
        clearing_price = self.calculate_clearing_price()
        
        if clearing_price is None:
            return {
                "success": False,
                "reason": "No clearing price found"
            }
        
        # Clear orders at clearing price
        cleared_volume = self.cleared_volumes.get(period, 0)
        
        return {
            "success": True,
            "period": period,
            "clearing_price": clearing_price,
            "cleared_volume": cleared_volume,
            "timestamp": datetime.now()
        }


class RealTimeMarket(Market):
    """Real-time energy market with continuous trading."""
    
    def __init__(self, name: str = "real_time"):
        """Initialize real-time market."""
        super().__init__(name, MarketType.REAL_TIME)
        
        # Real-time specific configuration
        self.settlement_interval = 5  # 5-minute settlements
        self.price_volatility_limit = 0.1  # 10% price movement limit
        self.imbalance_tolerance = 0.05  # 5% imbalance tolerance
        
        # Real-time tracking
        self.system_demand = 0.0
        self.system_supply = 0.0
        self.system_frequency = 60.0  # Hz
        self.reserve_margin = 0.0
    
    def execute_order(self, order: Order) -> Dict[str, Any]:
        """Execute order in real-time market."""
        try:
            # Validate order
            validation_errors = self.validate_order(order)
            if validation_errors:
                return {
                    "success": False,
                    "reason": "; ".join(validation_errors)
                }
            
            # Try to match order immediately
            matches = self.order_book.match_order(order)
            
            if matches:
                # Order was matched
                total_filled = sum(match["quantity"] for match in matches)
                avg_price = sum(match["price"] * match["quantity"] for match in matches) / total_filled
                
                return {
                    "success": True,
                    "quantity": total_filled,
                    "price": avg_price,
                    "fees": total_filled * self.transaction_fee,
                    "matches": len(matches)
                }
            else:
                # Add to order book if not fully filled
                if order.remaining_quantity > 0:
                    self.order_book.add_order(order)
                
                return {
                    "success": True,
                    "quantity": order.filled_quantity,
                    "price": order.average_price,
                    "fees": order.filled_quantity * self.transaction_fee,
                    "status": "partial" if order.filled_quantity > 0 else "pending"
                }
            
        except Exception as e:
            self.logger.error(f"Failed to execute order {order.id}: {e}")
            return {
                "success": False,
                "reason": str(e)
            }
    
    def calculate_clearing_price(self) -> Optional[float]:
        """Calculate current market clearing price."""
        mid_price = self.order_book.get_mid_price()
        
        if mid_price is not None:
            # Adjust for system conditions
            frequency_adjustment = (self.system_frequency - 60.0) * 0.01  # $0.01/Hz deviation
            imbalance_adjustment = (self.system_supply - self.system_demand) * 0.001
            
            adjusted_price = mid_price + frequency_adjustment + imbalance_adjustment
            return max(self.min_price, min(self.max_price, adjusted_price))
        
        return None
    
    def update_system_conditions(self, demand: float, supply: float, frequency: float) -> None:
        """Update real-time system conditions."""
        self.system_demand = demand
        self.system_supply = supply
        self.system_frequency = frequency
        self.reserve_margin = (supply - demand) / demand if demand > 0 else 0
        
        # Update market data
        self.current_data.system_lambda = self.calculate_clearing_price()
        self.current_data.total_demand = demand
        self.current_data.total_supply = supply


class AncillaryServicesMarket(Market):
    """Ancillary services market for grid stability services."""
    
    def __init__(self, name: str = "ancillary_services"):
        """Initialize ancillary services market."""
        super().__init__(name, MarketType.ANCILLARY_SERVICES)
        
        # Service types
        self.service_types = {
            "frequency_response": {"min_duration": 1, "max_duration": 30},  # seconds
            "spinning_reserve": {"min_duration": 10, "max_duration": 600},  # seconds
            "non_spinning_reserve": {"min_duration": 600, "max_duration": 3600},  # seconds
            "voltage_support": {"min_duration": 1, "max_duration": 3600},
            "black_start": {"min_duration": 3600, "max_duration": 86400}
        }
        
        # Service requirements
        self.service_requirements: Dict[str, float] = {
            "frequency_response": 100.0,  # MW
            "spinning_reserve": 500.0,
            "non_spinning_reserve": 1000.0,
            "voltage_support": 200.0,
            "black_start": 50.0
        }
        
        # Awarded capacity
        self.awarded_capacity: Dict[str, Dict[str, float]] = {}
    
    def execute_order(self, order: Order) -> Dict[str, Any]:
        """Execute ancillary services order."""
        try:
            service_type = order.metadata.get("service_type", "frequency_response")
            
            if service_type not in self.service_types:
                return {
                    "success": False,
                    "reason": f"Unknown service type: {service_type}"
                }
            
            # Validate service-specific requirements
            duration = order.metadata.get("duration", 0)
            service_config = self.service_types[service_type]
            
            if not (service_config["min_duration"] <= duration <= service_config["max_duration"]):
                return {
                    "success": False,
                    "reason": f"Duration {duration}s not valid for {service_type}"
                }
            
            # Check if capacity is needed
            required_capacity = self.service_requirements.get(service_type, 0)
            current_awarded = self.awarded_capacity.get(service_type, {})
            total_awarded = sum(current_awarded.values())
            
            if total_awarded >= required_capacity:
                return {
                    "success": False,
                    "reason": f"Sufficient {service_type} capacity already awarded"
                }
            
            # Award capacity (simplified - in practice this would be an auction)
            participant_id = order.metadata.get("participant_id", "unknown")
            
            if service_type not in self.awarded_capacity:
                self.awarded_capacity[service_type] = {}
            
            self.awarded_capacity[service_type][participant_id] = order.quantity
            
            # Calculate payment (simplified)
            capacity_price = order.price  # $/MW/hour
            energy_price = order.metadata.get("energy_price", 0.0)  # $/MWh when called
            
            return {
                "success": True,
                "quantity": order.quantity,
                "price": capacity_price,
                "service_type": service_type,
                "capacity_payment": capacity_price * order.quantity,
                "energy_price": energy_price
            }
            
        except Exception as e:
            self.logger.error(f"Failed to execute ancillary services order {order.id}: {e}")
            return {
                "success": False,
                "reason": str(e)
            }
    
    def calculate_clearing_price(self) -> Optional[float]:
        """Calculate clearing price for ancillary services."""
        # Simplified - would typically be service-specific
        return 50.0  # $/MW/hour base price


class BilateralMarket(Market):
    """Bilateral contract market for direct trading."""
    
    def __init__(self, name: str = "bilateral"):
        """Initialize bilateral market."""
        super().__init__(name, MarketType.BILATERAL)
        
        # Contract tracking
        self.active_contracts: Dict[str, Dict[str, Any]] = {}
        self.contract_templates: Dict[str, Dict[str, Any]] = {}
        
        # Default contract terms
        self.default_terms = {
            "settlement_period": "monthly",
            "payment_terms": "net_30",
            "force_majeure": True,
            "price_escalation": 0.02  # 2% annual
        }
    
    def execute_order(self, order: Order) -> Dict[str, Any]:
        """Execute bilateral contract order."""
        try:
            contract_type = order.metadata.get("contract_type", "standard")
            counterparty = order.metadata.get("counterparty")
            
            if not counterparty:
                return {
                    "success": False,
                    "reason": "Counterparty required for bilateral contracts"
                }
            
            # Create contract
            contract_id = f"BIL_{order.id}_{counterparty}"
            
            contract = {
                "id": contract_id,
                "buyer": counterparty if order.side == "sell" else order.metadata.get("participant_id"),
                "seller": order.metadata.get("participant_id") if order.side == "sell" else counterparty,
                "quantity": order.quantity,
                "price": order.price,
                "start_date": order.metadata.get("start_date", datetime.now()),
                "end_date": order.metadata.get("end_date", datetime.now() + timedelta(days=30)),
                "delivery_point": order.metadata.get("delivery_point", "default"),
                "terms": {**self.default_terms, **order.metadata.get("custom_terms", {})}
            }
            
            self.active_contracts[contract_id] = contract
            
            return {
                "success": True,
                "contract_id": contract_id,
                "quantity": order.quantity,
                "price": order.price,
                "counterparty": counterparty,
                "settlement_date": contract["end_date"]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to execute bilateral order {order.id}: {e}")
            return {
                "success": False,
                "reason": str(e)
            }
    
    def calculate_clearing_price(self) -> Optional[float]:
        """Calculate average bilateral contract price."""
        if not self.active_contracts:
            return None
        
        total_value = 0.0
        total_quantity = 0.0
        
        for contract in self.active_contracts.values():
            total_value += contract["price"] * contract["quantity"]
            total_quantity += contract["quantity"]
        
        return total_value / total_quantity if total_quantity > 0 else None
    
    def get_contract(self, contract_id: str) -> Optional[Dict[str, Any]]:
        """Get contract details."""
        return self.active_contracts.get(contract_id)
    
    def settle_contract(self, contract_id: str) -> Dict[str, Any]:
        """Settle a bilateral contract."""
        contract = self.active_contracts.get(contract_id)
        
        if not contract:
            return {
                "success": False,
                "reason": "Contract not found"
            }
        
        # Calculate settlement amount
        settlement_amount = contract["quantity"] * contract["price"]
        
        # Remove from active contracts
        del self.active_contracts[contract_id]
        
        return {
            "success": True,
            "contract_id": contract_id,
            "settlement_amount": settlement_amount,
            "settlement_date": datetime.now()
        }


def create_market(market_type: str, name: str = None, **kwargs) -> Market:
    """
    Factory function to create markets.
    
    Args:
        market_type: Type of market to create
        name: Market name (optional)
        **kwargs: Additional market configuration
        
    Returns:
        Market instance
        
    Example:
        >>> market = create_market("day_ahead", "CAISO_DA")
        >>> market = create_market("real_time", "PJM_RT")
    """
    market_type = market_type.lower()
    
    if market_type == "day_ahead":
        return DayAheadMarket(name or "day_ahead")
    elif market_type == "real_time":
        return RealTimeMarket(name or "real_time")
    elif market_type == "ancillary_services":
        return AncillaryServicesMarket(name or "ancillary_services")
    elif market_type == "bilateral":
        return BilateralMarket(name or "bilateral")
    else:
        raise ValueError(f"Unknown market type: {market_type}")
