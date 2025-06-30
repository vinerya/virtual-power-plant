"""
Order management system for VPP trading operations.

This module provides comprehensive order types, order book management,
and execution tracking for energy market trading.
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    ICEBERG = "iceberg"
    FILL_OR_KILL = "fill_or_kill"
    IMMEDIATE_OR_CANCEL = "immediate_or_cancel"


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order(ABC):
    """Base class for all order types."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    market: str = ""
    side: str = ""  # "buy" or "sell"
    quantity: float = 0.0
    price: float = 0.0
    order_type: OrderType = OrderType.MARKET
    status: OrderStatus = OrderStatus.PENDING
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Execution tracking
    filled_quantity: float = 0.0
    remaining_quantity: float = field(init=False)
    average_price: float = 0.0
    
    # Order parameters
    time_in_force: str = "GTC"  # Good Till Cancelled
    expire_time: Optional[datetime] = None
    client_order_id: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize computed fields."""
        self.remaining_quantity = self.quantity
    
    @abstractmethod
    def is_executable(self, market_price: float) -> bool:
        """Check if order can be executed at given market price."""
        pass
    
    def update_fill(self, fill_quantity: float, fill_price: float) -> None:
        """Update order with partial or full fill."""
        self.filled_quantity += fill_quantity
        self.remaining_quantity = self.quantity - self.filled_quantity
        
        # Update average price
        if self.filled_quantity > 0:
            total_value = (self.average_price * (self.filled_quantity - fill_quantity) + 
                          fill_price * fill_quantity)
            self.average_price = total_value / self.filled_quantity
        
        # Update status
        if self.remaining_quantity <= 0:
            self.status = OrderStatus.FILLED
        elif self.filled_quantity > 0:
            self.status = OrderStatus.PARTIAL
    
    def cancel(self) -> bool:
        """Cancel the order if possible."""
        if self.status in [OrderStatus.PENDING, OrderStatus.PARTIAL]:
            self.status = OrderStatus.CANCELLED
            return True
        return False
    
    def is_active(self) -> bool:
        """Check if order is still active."""
        return self.status in [OrderStatus.PENDING, OrderStatus.PARTIAL]
    
    def is_expired(self) -> bool:
        """Check if order has expired."""
        if self.expire_time and datetime.now() > self.expire_time:
            return True
        return False


class MarketOrder(Order):
    """Market order - executes immediately at best available price."""
    
    def __init__(self, market: str, side: str, quantity: float, **kwargs):
        """Initialize market order."""
        super().__init__(
            market=market,
            side=side,
            quantity=quantity,
            price=0.0,  # Market orders don't have a price
            order_type=OrderType.MARKET,
            **kwargs
        )
    
    def is_executable(self, market_price: float) -> bool:
        """Market orders are always executable."""
        return True


class LimitOrder(Order):
    """Limit order - executes only at specified price or better."""
    
    def __init__(self, market: str, side: str, quantity: float, price: float, **kwargs):
        """Initialize limit order."""
        super().__init__(
            market=market,
            side=side,
            quantity=quantity,
            price=price,
            order_type=OrderType.LIMIT,
            **kwargs
        )
    
    def is_executable(self, market_price: float) -> bool:
        """Check if limit order can be executed."""
        if self.side == "buy":
            return market_price <= self.price
        else:  # sell
            return market_price >= self.price


class StopOrder(Order):
    """Stop order - becomes market order when stop price is reached."""
    
    def __init__(self, market: str, side: str, quantity: float, stop_price: float, **kwargs):
        """Initialize stop order."""
        super().__init__(
            market=market,
            side=side,
            quantity=quantity,
            price=stop_price,
            order_type=OrderType.STOP,
            **kwargs
        )
        self.stop_price = stop_price
        self.triggered = False
    
    def is_executable(self, market_price: float) -> bool:
        """Check if stop order should be triggered."""
        if not self.triggered:
            if self.side == "buy":
                # Buy stop: trigger when price goes above stop price
                if market_price >= self.stop_price:
                    self.triggered = True
                    return True
            else:  # sell
                # Sell stop: trigger when price goes below stop price
                if market_price <= self.stop_price:
                    self.triggered = True
                    return True
            return False
        return True  # Already triggered, execute as market order


class StopLimitOrder(Order):
    """Stop-limit order - becomes limit order when stop price is reached."""
    
    def __init__(self, market: str, side: str, quantity: float, 
                 stop_price: float, limit_price: float, **kwargs):
        """Initialize stop-limit order."""
        super().__init__(
            market=market,
            side=side,
            quantity=quantity,
            price=limit_price,
            order_type=OrderType.STOP_LIMIT,
            **kwargs
        )
        self.stop_price = stop_price
        self.limit_price = limit_price
        self.triggered = False
    
    def is_executable(self, market_price: float) -> bool:
        """Check if stop-limit order can be executed."""
        if not self.triggered:
            if self.side == "buy":
                if market_price >= self.stop_price:
                    self.triggered = True
            else:  # sell
                if market_price <= self.stop_price:
                    self.triggered = True
            return False
        
        # Once triggered, behave like limit order
        if self.side == "buy":
            return market_price <= self.limit_price
        else:  # sell
            return market_price >= self.limit_price


class IcebergOrder(Order):
    """Iceberg order - only shows small portion of total quantity."""
    
    def __init__(self, market: str, side: str, quantity: float, price: float,
                 visible_quantity: float, **kwargs):
        """Initialize iceberg order."""
        super().__init__(
            market=market,
            side=side,
            quantity=quantity,
            price=price,
            order_type=OrderType.ICEBERG,
            **kwargs
        )
        self.visible_quantity = visible_quantity
        self.hidden_quantity = quantity - visible_quantity
    
    def is_executable(self, market_price: float) -> bool:
        """Check if iceberg order can be executed."""
        if self.side == "buy":
            return market_price <= self.price
        else:  # sell
            return market_price >= self.price
    
    def refresh_visible_quantity(self) -> None:
        """Refresh visible quantity after partial fill."""
        if self.hidden_quantity > 0:
            new_visible = min(self.hidden_quantity, self.visible_quantity)
            self.hidden_quantity -= new_visible
            # Reset visible portion for next execution


class FillOrKillOrder(Order):
    """Fill-or-Kill order - must be filled completely or cancelled."""
    
    def __init__(self, market: str, side: str, quantity: float, price: float, **kwargs):
        """Initialize FOK order."""
        super().__init__(
            market=market,
            side=side,
            quantity=quantity,
            price=price,
            order_type=OrderType.FILL_OR_KILL,
            time_in_force="FOK",
            **kwargs
        )
    
    def is_executable(self, market_price: float) -> bool:
        """FOK orders must be completely fillable."""
        if self.side == "buy":
            return market_price <= self.price
        else:  # sell
            return market_price >= self.price
    
    def update_fill(self, fill_quantity: float, fill_price: float) -> None:
        """FOK orders must be filled completely or cancelled."""
        if fill_quantity < self.remaining_quantity:
            # Partial fill not allowed - cancel order
            self.status = OrderStatus.CANCELLED
        else:
            super().update_fill(fill_quantity, fill_price)


class ImmediateOrCancelOrder(Order):
    """Immediate-or-Cancel order - fill what's possible, cancel the rest."""
    
    def __init__(self, market: str, side: str, quantity: float, price: float, **kwargs):
        """Initialize IOC order."""
        super().__init__(
            market=market,
            side=side,
            quantity=quantity,
            price=price,
            order_type=OrderType.IMMEDIATE_OR_CANCEL,
            time_in_force="IOC",
            **kwargs
        )
    
    def is_executable(self, market_price: float) -> bool:
        """IOC orders execute immediately if possible."""
        if self.side == "buy":
            return market_price <= self.price
        else:  # sell
            return market_price >= self.price


@dataclass
class OrderBookLevel:
    """Single level in order book."""
    price: float
    quantity: float
    order_count: int = 0
    orders: List[Order] = field(default_factory=list)


class OrderBook:
    """Order book for a specific market."""
    
    def __init__(self, market: str):
        """Initialize order book."""
        self.market = market
        self.bids: Dict[float, OrderBookLevel] = {}  # Buy orders
        self.asks: Dict[float, OrderBookLevel] = {}  # Sell orders
        self.last_update: datetime = datetime.now()
    
    def add_order(self, order: Order) -> None:
        """Add order to the book."""
        if order.order_type == OrderType.MARKET:
            return  # Market orders don't go in the book
        
        book_side = self.bids if order.side == "buy" else self.asks
        price = order.price
        
        if price not in book_side:
            book_side[price] = OrderBookLevel(price=price, quantity=0)
        
        level = book_side[price]
        level.quantity += order.remaining_quantity
        level.order_count += 1
        level.orders.append(order)
        
        self.last_update = datetime.now()
    
    def remove_order(self, order: Order) -> None:
        """Remove order from the book."""
        book_side = self.bids if order.side == "buy" else self.asks
        price = order.price
        
        if price in book_side:
            level = book_side[price]
            if order in level.orders:
                level.orders.remove(order)
                level.quantity -= order.remaining_quantity
                level.order_count -= 1
                
                if level.order_count == 0:
                    del book_side[price]
        
        self.last_update = datetime.now()
    
    def update_order(self, order: Order, old_quantity: float) -> None:
        """Update order quantity in the book."""
        book_side = self.bids if order.side == "buy" else self.asks
        price = order.price
        
        if price in book_side:
            level = book_side[price]
            quantity_change = order.remaining_quantity - old_quantity
            level.quantity += quantity_change
        
        self.last_update = datetime.now()
    
    def get_best_bid(self) -> Optional[float]:
        """Get best bid price."""
        if self.bids:
            return max(self.bids.keys())
        return None
    
    def get_best_ask(self) -> Optional[float]:
        """Get best ask price."""
        if self.asks:
            return min(self.asks.keys())
        return None
    
    def get_spread(self) -> Optional[float]:
        """Get bid-ask spread."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid is not None and best_ask is not None:
            return best_ask - best_bid
        return None
    
    def get_mid_price(self) -> Optional[float]:
        """Get mid-market price."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2
        return None
    
    def get_market_depth(self, levels: int = 5) -> Dict[str, List[Dict[str, float]]]:
        """Get market depth (top N levels)."""
        # Sort bids (highest first) and asks (lowest first)
        sorted_bids = sorted(self.bids.items(), key=lambda x: x[0], reverse=True)
        sorted_asks = sorted(self.asks.items(), key=lambda x: x[0])
        
        bid_levels = []
        for price, level in sorted_bids[:levels]:
            bid_levels.append({
                "price": price,
                "quantity": level.quantity,
                "orders": level.order_count
            })
        
        ask_levels = []
        for price, level in sorted_asks[:levels]:
            ask_levels.append({
                "price": price,
                "quantity": level.quantity,
                "orders": level.order_count
            })
        
        return {
            "bids": bid_levels,
            "asks": ask_levels,
            "timestamp": self.last_update
        }
    
    def match_order(self, incoming_order: Order) -> List[Dict[str, Any]]:
        """Match incoming order against the book."""
        matches = []
        
        if incoming_order.side == "buy":
            # Match against asks (sell orders)
            sorted_asks = sorted(self.asks.items(), key=lambda x: x[0])
            
            for price, level in sorted_asks:
                if not incoming_order.is_executable(price):
                    break
                
                remaining_to_fill = incoming_order.remaining_quantity
                if remaining_to_fill <= 0:
                    break
                
                # Match against orders at this level
                for book_order in level.orders[:]:  # Copy list to avoid modification issues
                    if remaining_to_fill <= 0:
                        break
                    
                    fill_quantity = min(remaining_to_fill, book_order.remaining_quantity)
                    
                    matches.append({
                        "incoming_order": incoming_order,
                        "book_order": book_order,
                        "price": price,
                        "quantity": fill_quantity
                    })
                    
                    # Update orders
                    incoming_order.update_fill(fill_quantity, price)
                    book_order.update_fill(fill_quantity, price)
                    
                    # Remove filled order from book
                    if book_order.remaining_quantity <= 0:
                        self.remove_order(book_order)
                    
                    remaining_to_fill -= fill_quantity
        
        else:  # sell order
            # Match against bids (buy orders)
            sorted_bids = sorted(self.bids.items(), key=lambda x: x[0], reverse=True)
            
            for price, level in sorted_bids:
                if not incoming_order.is_executable(price):
                    break
                
                remaining_to_fill = incoming_order.remaining_quantity
                if remaining_to_fill <= 0:
                    break
                
                # Match against orders at this level
                for book_order in level.orders[:]:
                    if remaining_to_fill <= 0:
                        break
                    
                    fill_quantity = min(remaining_to_fill, book_order.remaining_quantity)
                    
                    matches.append({
                        "incoming_order": incoming_order,
                        "book_order": book_order,
                        "price": price,
                        "quantity": fill_quantity
                    })
                    
                    # Update orders
                    incoming_order.update_fill(fill_quantity, price)
                    book_order.update_fill(fill_quantity, price)
                    
                    # Remove filled order from book
                    if book_order.remaining_quantity <= 0:
                        self.remove_order(book_order)
                    
                    remaining_to_fill -= fill_quantity
        
        return matches
    
    def get_volume_at_price(self, price: float, side: str) -> float:
        """Get total volume available at a specific price."""
        book_side = self.bids if side == "buy" else self.asks
        level = book_side.get(price)
        return level.quantity if level else 0.0
    
    def get_total_volume(self, side: str) -> float:
        """Get total volume on one side of the book."""
        book_side = self.bids if side == "buy" else self.asks
        return sum(level.quantity for level in book_side.values())


def create_order(order_type: str, market: str, side: str, quantity: float, 
                price: float = None, **kwargs) -> Order:
    """
    Factory function to create orders of different types.
    
    Args:
        order_type: Type of order ("market", "limit", "stop", etc.)
        market: Market identifier
        side: Order side ("buy" or "sell")
        quantity: Order quantity
        price: Order price (required for most order types)
        **kwargs: Additional order parameters
        
    Returns:
        Order instance of the specified type
        
    Example:
        >>> order = create_order("limit", "day_ahead", "buy", 100, 0.12)
        >>> order = create_order("market", "real_time", "sell", 50)
    """
    order_type = order_type.lower()
    
    if order_type == "market":
        return MarketOrder(market, side, quantity, **kwargs)
    elif order_type == "limit":
        if price is None:
            raise ValueError("Limit orders require a price")
        return LimitOrder(market, side, quantity, price, **kwargs)
    elif order_type == "stop":
        if price is None:
            raise ValueError("Stop orders require a stop price")
        return StopOrder(market, side, quantity, price, **kwargs)
    elif order_type == "stop_limit":
        stop_price = kwargs.pop("stop_price", price)
        limit_price = kwargs.pop("limit_price", price)
        return StopLimitOrder(market, side, quantity, stop_price, limit_price, **kwargs)
    elif order_type == "iceberg":
        if price is None:
            raise ValueError("Iceberg orders require a price")
        visible_quantity = kwargs.pop("visible_quantity", quantity * 0.1)
        return IcebergOrder(market, side, quantity, price, visible_quantity, **kwargs)
    elif order_type == "fill_or_kill" or order_type == "fok":
        if price is None:
            raise ValueError("FOK orders require a price")
        return FillOrKillOrder(market, side, quantity, price, **kwargs)
    elif order_type == "immediate_or_cancel" or order_type == "ioc":
        if price is None:
            raise ValueError("IOC orders require a price")
        return ImmediateOrCancelOrder(market, side, quantity, price, **kwargs)
    else:
        raise ValueError(f"Unknown order type: {order_type}")


def validate_order_parameters(order_type: str, market: str, side: str, 
                            quantity: float, price: float = None, **kwargs) -> List[str]:
    """
    Validate order parameters.
    
    Args:
        order_type: Type of order
        market: Market identifier
        side: Order side
        quantity: Order quantity
        price: Order price
        **kwargs: Additional parameters
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    # Basic validation
    if not market:
        errors.append("Market is required")
    
    if side not in ["buy", "sell"]:
        errors.append("Side must be 'buy' or 'sell'")
    
    if quantity <= 0:
        errors.append("Quantity must be positive")
    
    # Order type specific validation
    if order_type in ["limit", "stop", "stop_limit", "iceberg", "fill_or_kill", "immediate_or_cancel"]:
        if price is None or price <= 0:
            errors.append(f"{order_type} orders require a positive price")
    
    if order_type == "stop_limit":
        stop_price = kwargs.get("stop_price")
        limit_price = kwargs.get("limit_price")
        if stop_price is None or stop_price <= 0:
            errors.append("Stop-limit orders require a positive stop price")
        if limit_price is None or limit_price <= 0:
            errors.append("Stop-limit orders require a positive limit price")
    
    if order_type == "iceberg":
        visible_quantity = kwargs.get("visible_quantity", quantity * 0.1)
        if visible_quantity <= 0 or visible_quantity > quantity:
            errors.append("Iceberg visible quantity must be positive and less than total quantity")
    
    return errors
