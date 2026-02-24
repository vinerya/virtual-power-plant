"""SQLAlchemy ORM models for all persisted entities."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, TimestampMixin


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------

class ResourceModel(TimestampMixin, Base):
    """Persisted energy resource."""

    __tablename__ = "resources"

    name: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    resource_type: Mapped[str] = mapped_column(String(50), index=True)  # battery | solar | wind_turbine
    rated_power: Mapped[float] = mapped_column(Float)
    online: Mapped[bool] = mapped_column(Boolean, default=True)
    current_power: Mapped[float] = mapped_column(Float, default=0.0)
    efficiency: Mapped[float] = mapped_column(Float, default=0.95)
    config_json: Mapped[str] = mapped_column(Text, default="{}")  # type-specific params
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")

    # Relationships
    battery_states: Mapped[list["BatteryStateModel"]] = relationship(
        back_populates="resource", cascade="all, delete-orphan"
    )


class BatteryStateModel(TimestampMixin, Base):
    """Time-series battery state snapshots."""

    __tablename__ = "battery_states"
    __table_args__ = (
        Index("ix_battery_states_resource_ts", "resource_id", "timestamp"),
    )

    resource_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("resources.id", ondelete="CASCADE"), index=True
    )
    soc: Mapped[float] = mapped_column(Float)  # 0-100
    soh: Mapped[float] = mapped_column(Float, default=100.0)
    temperature: Mapped[float] = mapped_column(Float, default=25.0)
    voltage: Mapped[float] = mapped_column(Float, default=0.0)
    current: Mapped[float] = mapped_column(Float, default=0.0)
    power: Mapped[float] = mapped_column(Float, default=0.0)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    resource: Mapped["ResourceModel"] = relationship(back_populates="battery_states")


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------

class OptimizationRunModel(TimestampMixin, Base):
    """Record of a single optimization solve."""

    __tablename__ = "optimization_runs"
    __table_args__ = (
        Index("ix_opt_runs_type_ts", "problem_type", "created_at"),
    )

    problem_type: Mapped[str] = mapped_column(String(50), index=True)  # stochastic | realtime | distributed
    status: Mapped[str] = mapped_column(String(30))  # success | failed | timeout | fallback
    objective_value: Mapped[float] = mapped_column(Float, default=0.0)
    solve_time_ms: Mapped[float] = mapped_column(Float, default=0.0)
    solver: Mapped[str] = mapped_column(String(100), default="")
    fallback_used: Mapped[bool] = mapped_column(Boolean, default=False)
    solution_json: Mapped[str] = mapped_column(Text, default="{}")
    parameters_json: Mapped[str] = mapped_column(Text, default="{}")


# ---------------------------------------------------------------------------
# Trading
# ---------------------------------------------------------------------------

class OrderModel(TimestampMixin, Base):
    """Persisted trading order."""

    __tablename__ = "orders"
    __table_args__ = (
        Index("ix_orders_market_status", "market", "status"),
    )

    order_type: Mapped[str] = mapped_column(String(30))
    market: Mapped[str] = mapped_column(String(100), index=True)
    side: Mapped[str] = mapped_column(String(10))
    quantity: Mapped[float] = mapped_column(Float)
    price: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String(30), default="pending", index=True)
    filled_quantity: Mapped[float] = mapped_column(Float, default=0.0)
    remaining_quantity: Mapped[float] = mapped_column(Float, default=0.0)
    average_price: Mapped[float] = mapped_column(Float, default=0.0)
    time_in_force: Mapped[str] = mapped_column(String(10), default="GTC")
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")

    trades: Mapped[list["TradeModel"]] = relationship(
        back_populates="order", cascade="all, delete-orphan"
    )


class TradeModel(TimestampMixin, Base):
    """Persisted trade execution."""

    __tablename__ = "trades"
    __table_args__ = (
        Index("ix_trades_market_ts", "market", "created_at"),
    )

    order_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("orders.id", ondelete="CASCADE"), index=True
    )
    market: Mapped[str] = mapped_column(String(100), index=True)
    side: Mapped[str] = mapped_column(String(10))
    quantity: Mapped[float] = mapped_column(Float)
    price: Mapped[float] = mapped_column(Float)
    fees: Mapped[float] = mapped_column(Float, default=0.0)
    strategy: Mapped[str] = mapped_column(String(100), default="")
    realized_pnl: Mapped[float] = mapped_column(Float, default=0.0)

    order: Mapped["OrderModel"] = relationship(back_populates="trades")


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

class UserModel(TimestampMixin, Base):
    """Application user."""

    __tablename__ = "users"

    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(256))
    role: Mapped[str] = mapped_column(String(30), default="viewer")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


class APIKeyModel(TimestampMixin, Base):
    """API key for programmatic access."""

    __tablename__ = "api_keys"

    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(128))
    hashed_key: Mapped[str] = mapped_column(String(256), unique=True, index=True)
    role: Mapped[str] = mapped_column(String(30), default="viewer")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

class EventLogModel(TimestampMixin, Base):
    """Persisted event log for auditing and replay."""

    __tablename__ = "event_log"
    __table_args__ = (
        Index("ix_event_log_type_ts", "event_type", "created_at"),
    )

    event_type: Mapped[str] = mapped_column(String(50), index=True)
    resource_id: Mapped[str] = mapped_column(String(36), nullable=True, index=True)
    details_json: Mapped[str] = mapped_column(Text, default="{}")
    severity: Mapped[str] = mapped_column(String(20), default="info")
