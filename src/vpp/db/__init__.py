"""Database layer — async SQLAlchemy 2.0 with repository pattern."""

from .base import Base, TimestampMixin
from .engine import get_db, init_db, create_engine_from_settings
from .models import (
    ResourceModel,
    BatteryStateModel,
    OptimizationRunModel,
    OrderModel,
    TradeModel,
    UserModel,
    APIKeyModel,
    EventLogModel,
)
from .repositories import (
    ResourceRepository,
    OptimizationRepository,
    TradingRepository,
    UserRepository,
)

__all__ = [
    "Base",
    "TimestampMixin",
    "get_db",
    "init_db",
    "create_engine_from_settings",
    # Models
    "ResourceModel",
    "BatteryStateModel",
    "OptimizationRunModel",
    "OrderModel",
    "TradeModel",
    "UserModel",
    "APIKeyModel",
    "EventLogModel",
    # Repositories
    "ResourceRepository",
    "OptimizationRepository",
    "TradingRepository",
    "UserRepository",
]
