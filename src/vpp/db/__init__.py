"""Database layer — async SQLAlchemy 2.0 with repository pattern."""

from .base import Base, TimestampMixin
from .engine import create_engine_from_settings, get_db, init_db
from .models import (
    APIKeyModel,
    BatteryStateModel,
    EventLogModel,
    OptimizationRunModel,
    OrderModel,
    ResourceModel,
    TradeModel,
    UserModel,
)
from .repositories import (
    OptimizationRepository,
    ResourceRepository,
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
