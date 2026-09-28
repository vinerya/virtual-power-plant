"""Pydantic v2 schemas for API request/response validation."""

from .auth import (
    APIKeyCreate,
    APIKeyResponse,
    Token,
    TokenPayload,
    UserCreate,
    UserResponse,
)
from .optimization import (
    DispatchRequest,
    DispatchResponse,
    DistributedRequest,
    OptimizationRequest,
    OptimizationResponse,
    RealTimeRequest,
    StochasticRequest,
)
from .resources import (
    BatteryCreate,
    ResourceCreate,
    ResourceCreateRequest,
    ResourceMetrics,
    ResourceResponse,
    ResourceType,
    ResourceUpdate,
    SolarCreate,
    WindTurbineCreate,
)
from .trading import (
    MarketDataResponse,
    OrderCreate,
    OrderResponse,
    PortfolioResponse,
    PositionResponse,
    TradeResponse,
)

__all__ = [
    # Resources
    "ResourceCreate",
    "ResourceCreateRequest",
    "ResourceResponse",
    "ResourceUpdate",
    "BatteryCreate",
    "SolarCreate",
    "WindTurbineCreate",
    "ResourceMetrics",
    "ResourceType",
    # Optimization
    "DispatchRequest",
    "DispatchResponse",
    "OptimizationRequest",
    "OptimizationResponse",
    "StochasticRequest",
    "RealTimeRequest",
    "DistributedRequest",
    # Trading
    "OrderCreate",
    "OrderResponse",
    "TradeResponse",
    "PositionResponse",
    "PortfolioResponse",
    "MarketDataResponse",
    # Auth
    "UserCreate",
    "UserResponse",
    "Token",
    "TokenPayload",
    "APIKeyCreate",
    "APIKeyResponse",
]
