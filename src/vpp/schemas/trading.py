"""Pydantic schemas for trading operations."""

from __future__ import annotations

from datetime import datetime  # noqa: TC003 -- pydantic resolves at runtime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_ORDER_TYPES = {
    "market", "limit", "stop", "stop_limit", "iceberg",
    "fok", "ioc", "fill_or_kill", "immediate_or_cancel",
}
_TIME_IN_FORCE = {"GTC", "DAY", "IOC", "FOK"}


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
    price: float | None = Field(None, ge=0, description="Limit / stop price")
    stop_price: float | None = Field(None, ge=0, description="Stop trigger price")
    limit_price: float | None = Field(None, ge=0, description="Limit price for stop-limit")
    visible_quantity: float | None = Field(None, gt=0, description="Visible qty for iceberg")
    time_in_force: str = Field("GTC", description="GTC | FOK | IOC | DAY")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("order_type")
    @classmethod
    def _check_order_type(cls, v: str) -> str:
        v = v.lower()
        if v not in _ORDER_TYPES:
            raise ValueError(f"order_type must be one of {sorted(_ORDER_TYPES)}")
        return v

    @field_validator("time_in_force")
    @classmethod
    def _check_tif(cls, v: str) -> str:
        v = v.upper()
        if v not in _TIME_IN_FORCE:
            raise ValueError(f"time_in_force must be one of {sorted(_TIME_IN_FORCE)}")
        return v


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
    updated_at: datetime | None = None
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
    strategy: str | None = None
    realized_pnl: float = 0.0


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------

class OrderSubmitResponse(OrderResponse):
    """Order as accepted by the venue plus any fills it produced immediately."""

    fills: list[TradeResponse] = Field(default_factory=list)


class PositionResponse(BaseModel):
    """Schema for a single market position."""

    market: str
    quantity: float
    average_price: float
    mark_price: float | None = None
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    notional_value: float = 0.0


class VolatilityEstimate(BaseModel):
    value: float = Field(..., description="Daily return standard deviation")
    source: str = Field(..., description="estimated | model_prior")
    samples: int = 0


class RiskLimitsResponse(BaseModel):
    max_position: float
    max_daily_loss: float
    max_drawdown: float
    var_limit: float
    concentration_limit: float


class RiskSummary(BaseModel):
    var_95_1d: float | None = None
    var_method: str = ""
    daily_pnl: float = 0.0
    concentrations: dict[str, float] = Field(default_factory=dict)
    breach: bool = False
    breaches: list[str] = Field(default_factory=list)
    volatility: dict[str, VolatilityEstimate] = Field(default_factory=dict)
    limits: RiskLimitsResponse | None = None


class PortfolioResponse(BaseModel):
    """Schema for the full portfolio snapshot.

    ``total_pnl`` = ``equity - initial_cash`` = realized + unrealized - fees.
    Drawdowns are fractions of peak equity.
    """

    cash: float
    equity: float
    total_pnl: float
    max_drawdown: float
    positions: list[PositionResponse] = Field(default_factory=list)
    total_trades: int = 0
    last_updated: datetime
    initial_cash: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    fees_paid: float = 0.0
    current_drawdown: float = 0.0
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    open_orders: int = 0
    risk: RiskSummary | None = None
    venue: str = "simulated"


class MarketDataResponse(BaseModel):
    """Snapshot of market data for a single market."""

    market: str
    bid: float | None = None
    ask: float | None = None
    last_price: float | None = None
    volume: float = 0.0
    timestamp: datetime


class MarketResponse(BaseModel):
    """A tradable (simulated) market with its latest quote."""

    market: str
    market_type: str
    status: str
    venue: str = "simulated"
    source: str = "simulated"
    currency: str = "USD"
    price_unit: str = "$/MWh"
    quantity_unit: str = "MWh"
    tick_size: float
    lot_size: float
    fee_per_unit: float
    last_price: float | None = None
    bid: float | None = None
    ask: float | None = None
    volume: float = 0.0
    timestamp: datetime | None = None
    depth: dict[str, list[tuple[float, float]]] | None = None


class TickResponse(BaseModel):
    timestamp: datetime
    markets: list[MarketResponse]
    fills: int = 0
    expired_orders: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

class StrategyInfo(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]
    min_markets: int = 1


class SyntheticDataSpec(BaseModel):
    periods: int = Field(168, ge=2, le=20_000, description="Number of bars")
    interval_minutes: float = Field(60.0, gt=0, le=1440)
    seed: int = 42


class BacktestRequest(BaseModel):
    """Backtest a strategy over supplied prices, or seeded synthetic prices."""

    params: dict[str, Any] = Field(default_factory=dict)
    prices: dict[str, list[float]] | None = Field(
        None, description="Per-market price series ($/MWh); equal lengths"
    )
    timestamps: list[datetime] | None = None
    interval_minutes: float = Field(60.0, gt=0, le=1440)
    synthetic: SyntheticDataSpec = Field(default_factory=SyntheticDataSpec)
    initial_cash: float = Field(100_000.0, gt=0)
    fee_per_unit: float = Field(0.07, ge=0)
    half_spread: float = Field(0.001, ge=0, lt=0.5)

    @model_validator(mode="after")
    def _check_prices(self) -> BacktestRequest:
        if self.prices is not None:
            if not self.prices:
                raise ValueError("prices must contain at least one market")
            lengths = {len(v) for v in self.prices.values()}
            if len(lengths) != 1:
                raise ValueError("all price series must have the same length")
            n = lengths.pop()
            if n < 2 or n > 20_000:
                raise ValueError("price series must have between 2 and 20000 points")
            if self.timestamps is not None and len(self.timestamps) != n:
                raise ValueError("timestamps must match the price series length")
        return self


class EquityPoint(BaseModel):
    timestamp: datetime
    equity: float


class BacktestTrade(BaseModel):
    timestamp: datetime
    market: str
    side: str
    quantity: float
    price: float
    fees: float
    realized_pnl: float


class BacktestResponse(BaseModel):
    strategy: str
    data_source: str = Field(..., description="provided | synthetic")
    markets: list[str]
    periods: int
    interval_minutes: float
    initial_cash: float
    final_equity: float
    total_pnl: float
    total_return: float
    realized_pnl: float
    unrealized_pnl: float
    fees: float
    sharpe_ratio: float
    max_drawdown: float
    num_trades: int
    win_rate: float | None = None
    final_positions: dict[str, float] = Field(default_factory=dict)
    equity_curve: list[EquityPoint] = Field(default_factory=list)
    trades: list[BacktestTrade] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


class StrategyRunRequest(BaseModel):
    params: dict[str, Any] = Field(default_factory=dict)
    dry_run: bool = Field(True, description="Only return signals; submit nothing")


class StrategySignalResult(BaseModel):
    market: str
    side: str
    order_type: str
    quantity: float
    price: float | None = None
    confidence: float = 0.0
    status: str
    order_id: str | None = None
    reasons: list[str] = Field(default_factory=list)


class StrategyRunResponse(BaseModel):
    strategy: str
    dry_run: bool
    signals: list[StrategySignalResult]
