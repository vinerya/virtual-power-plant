"""Pydantic schemas for trading operations."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------

class OrderCreate(BaseModel):
    """Schema for submitting a new order."""

    order_type: str = Field(
        ...,
        description="market | limit | stop | stop_limit | iceberg | fok | ioc",
    )
    market: str = Field(..., min_length=1, description="Target market name")
    side: str = Field(..., pattern="^(buy|sell)$", description="buy or sell")
    quantity: float = Field(..., gt=0, description="Order quantity in kW or MWh")
    price: Optional[float] = Field(None, ge=0, description="Limit / stop price")
    stop_price: Optional[float] = Field(None, ge=0, description="Stop trigger price")
    limit_price: Optional[float] = Field(None, ge=0, description="Limit price for stop-limit")
    visible_quantity: Optional[float] = Field(None, gt=0, description="Visible qty for iceberg")
    time_in_force: str = Field("GTC", description="GTC | FOK | IOC | DAY")
    metadata: dict[str, Any] = Field(default_factory=dict)


class OrderResponse(BaseModel):
    """Schema returned when querying an order."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    order_type: str
    market: str
    side: str
    quantity: float
    price: float
    status: str
    filled_quantity: float = 0.0
    remaining_quantity: float = 0.0
    average_price: float = 0.0
    time_in_force: str = "GTC"
    created_at: datetime
    updated_at: Optional[datetime] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Trades
# ---------------------------------------------------------------------------

class TradeResponse(BaseModel):
    """Schema for a completed trade."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    order_id: str
    market: str
    side: str
    quantity: float
    price: float
    fees: float = 0.0
    timestamp: datetime
    strategy: Optional[str] = None
    realized_pnl: float = 0.0


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------

class PositionResponse(BaseModel):
    """Schema for a single market position."""

    market: str
    quantity: float
    average_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    notional_value: float = 0.0


class PortfolioResponse(BaseModel):
    """Schema for the full portfolio snapshot."""

    cash: float
    equity: float
    total_pnl: float
    max_drawdown: float
    positions: list[PositionResponse] = Field(default_factory=list)
    total_trades: int = 0
    last_updated: datetime


class MarketDataResponse(BaseModel):
    """Snapshot of market data for a single market."""

    market: str
    bid: Optional[float] = None
    ask: Optional[float] = None
    last_price: Optional[float] = None
    volume: float = 0.0
    timestamp: datetime
