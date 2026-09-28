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
    Integer,
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

    # M3 -- Battery degradation persistence.  Migration:
    # 0002_add_battery_degradation.py.  Any schema change in this module needs
    # a matching alembic revision; tests/test_alembic_drift.py enforces it.
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
# Tariffs
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


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

class AlertRuleModel(TimestampMixin, Base):
    """Persisted alert rule evaluated against live telemetry.

    ``metric`` names a numeric key of ``RESOURCE_UPDATED`` event data
    (e.g. ``soc``, ``temperature``, ``current_power_kw``).  ``resource_id``
    optionally scopes the rule to a single resource.
    """

    __tablename__ = "alert_rules"

    name: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    description: Mapped[str] = mapped_column(Text, default="")
    rule_type: Mapped[str] = mapped_column(String(32), default="threshold")
    metric: Mapped[str] = mapped_column(String(64), index=True)
    comparison: Mapped[str] = mapped_column(String(4), default=">")
    threshold: Mapped[float] = mapped_column(Float, default=0.0)
    severity: Mapped[str] = mapped_column(String(16), default="warning")
    rate_window_s: Mapped[float] = mapped_column(Float, default=60.0)
    rate_threshold: Mapped[float] = mapped_column(Float, default=0.0)
    z_score_threshold: Mapped[float] = mapped_column(Float, default=3.0)
    cooldown_s: Mapped[float] = mapped_column(Float, default=300.0)
    resource_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    auto_resolve: Mapped[bool] = mapped_column(Boolean, default=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)


class AlertModel(TimestampMixin, Base):
    """A fired alert and its operator lifecycle.

    ``status`` is one of ``active`` | ``acknowledged`` | ``snoozed`` |
    ``resolved``.  A ``snoozed`` alert whose ``snoozed_until`` has passed is
    reported as ``active`` again by the API (computed at read time).
    """

    __tablename__ = "alerts"
    __table_args__ = (
        Index("ix_alerts_status_fired", "status", "fired_at"),
        Index("ix_alerts_rule_source", "rule_id", "source"),
    )

    rule_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    rule_name: Mapped[str] = mapped_column(String(255), default="")
    severity: Mapped[str] = mapped_column(String(16), default="warning", index=True)
    status: Mapped[str] = mapped_column(String(16), default="active")
    title: Mapped[str] = mapped_column(String(255), default="")
    message: Mapped[str] = mapped_column(Text, default="")
    source: Mapped[str] = mapped_column(String(255), default="system", index=True)
    source_kind: Mapped[str] = mapped_column(String(32), default="system")
    metric: Mapped[str | None] = mapped_column(String(64), nullable=True)
    value: Mapped[float | None] = mapped_column(Float, nullable=True)
    threshold: Mapped[float | None] = mapped_column(Float, nullable=True)
    occurrences: Mapped[int] = mapped_column(Integer, default=1)
    fired_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    last_fired_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    acknowledged_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    snoozed_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    resolved_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")
