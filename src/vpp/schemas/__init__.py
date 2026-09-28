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
    BatteryResponse,
    ResourceCreate,
    ResourceMetrics,
    ResourceResponse,
    ResourceType,
    ResourceUpdate,
    SolarCreate,
    SolarResponse,
    WindTurbineCreate,
    WindTurbineResponse,
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
