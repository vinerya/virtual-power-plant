"""SQLAlchemy ORM models for all persisted entities."""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    Date,
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

    # M3 -- Battery degradation persistence.
    # NOTE: this project relies on `Base.metadata.create_all()` at startup
    # rather than alembic migrations.  When alembic is wired up later (M4+),
    # these four columns require an explicit migration on existing dev DBs:
    #   ALTER TABLE resources ADD COLUMN state_of_health FLOAT DEFAULT 1.0;
    #   ALTER TABLE resources ADD COLUMN cumulative_throughput_kwh FLOAT DEFAULT 0.0;
    #   ALTER TABLE resources ADD COLUMN last_degradation_update DATETIME NULL;
    #   ALTER TABLE resources ADD COLUMN chemistry VARCHAR(16) NULL;
    state_of_health: Mapped[float] = mapped_column(Float, default=1.0)
    cumulative_throughput_kwh: Mapped[float] = mapped_column(Float, default=0.0)
    last_degradation_update: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    chemistry: Mapped[str | None] = mapped_column(String(16), nullable=True)

    # M4 -- Nominal energy capacity (kWh) for accurate throughput accounting.
    # Nullable for non-battery resources.  When NULL on a battery the
    # DegradationUpdater falls back to a documented C/4 heuristic
    # (rated_power * 4.0).  Migration: 0003_add_nominal_energy.py.
    nominal_energy_kwh: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Relationships
    battery_states: Mapped[list["BatteryStateModel"]] = relationship(
        back_populates="resource", cascade="all, delete-orphan"
    )
    soh_samples: Mapped[list["BatterySOHSampleModel"]] = relationship(
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


class BatterySOHSampleModel(TimestampMixin, Base):
    """Time-series SOH/throughput samples for the dashboard history view."""

    __tablename__ = "battery_soh_samples"
    __table_args__ = (
        Index("ix_battery_soh_samples_resource_ts", "resource_id", "timestamp"),
    )

    resource_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("resources.id", ondelete="CASCADE"), index=True
    )
    state_of_health: Mapped[float] = mapped_column(Float)
    cumulative_throughput_kwh: Mapped[float] = mapped_column(Float)
    loss_fraction: Mapped[float] = mapped_column(Float, default=0.0)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    resource: Mapped["ResourceModel"] = relationship(back_populates="soh_samples")


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

class TariffRow(TimestampMixin, Base):
    """Persisted utility tariff (URDB-shaped JSON payload)."""

    __tablename__ = "tariffs"
    __table_args__ = (
        Index("ix_tariffs_utility", "utility"),
        Index("ix_tariffs_urdb_label", "urdb_label"),
    )

    name: Mapped[str] = mapped_column(String(255), index=True)
    utility: Mapped[str] = mapped_column(String(255), default="")
    urdb_label: Mapped[str | None] = mapped_column(String(128), nullable=True)
    urdb_json: Mapped[str] = mapped_column(Text, default="{}")  # JSON-serialized URDB payload
    effective_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


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
