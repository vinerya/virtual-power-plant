"""Pydantic v2 schemas for API request/response validation."""

from .resources import (
    ResourceCreate,
    ResourceResponse,
    ResourceUpdate,
    BatteryCreate,
    BatteryResponse,
    SolarCreate,
    SolarResponse,
    WindTurbineCreate,
    WindTurbineResponse,
    ResourceMetrics,
    ResourceType,
)
from .optimization import (
    DispatchRequest,
    DispatchResponse,
    OptimizationRequest,
    OptimizationResponse,
    StochasticRequest,
    RealTimeRequest,
    DistributedRequest,
)
from .trading import (
    OrderCreate,
    OrderResponse,
    TradeResponse,
    PositionResponse,
    PortfolioResponse,
    MarketDataResponse,
)
from .auth import (
    UserCreate,
    UserResponse,
    Token,
    TokenPayload,
    APIKeyCreate,
    APIKeyResponse,
)

__all__ = [
    # Resources
    "ResourceCreate",
    "ResourceResponse",
    "ResourceUpdate",
    "BatteryCreate",
    "BatteryResponse",
    "SolarCreate",
    "SolarResponse",
    "WindTurbineCreate",
    "WindTurbineResponse",
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
