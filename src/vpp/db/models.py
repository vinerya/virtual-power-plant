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
    UniqueConstraint,
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
    resource_type: Mapped[str] = mapped_column(
        String(50), index=True
    )  # battery | solar | wind_turbine
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

    # Site membership (sites/customer portal). NULL = not assigned to any
    # site. Ownership is derived through the site (``sites.owner_id``).
    site_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("sites.id", ondelete="SET NULL"), nullable=True, index=True
    )

    # Relationships
    battery_states: Mapped[list[BatteryStateModel]] = relationship(
        back_populates="resource", cascade="all, delete-orphan"
    )
    soh_samples: Mapped[list[BatterySOHSampleModel]] = relationship(
        back_populates="resource", cascade="all, delete-orphan"
    )


class BatteryStateModel(TimestampMixin, Base):
    """Time-series battery state snapshots."""

    __tablename__ = "battery_states"
    __table_args__ = (Index("ix_battery_states_resource_ts", "resource_id", "timestamp"),)

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

    resource: Mapped[ResourceModel] = relationship(back_populates="battery_states")


class BatterySOHSampleModel(TimestampMixin, Base):
    """Time-series SOH/throughput samples for the dashboard history view."""

    __tablename__ = "battery_soh_samples"
    __table_args__ = (Index("ix_battery_soh_samples_resource_ts", "resource_id", "timestamp"),)

    resource_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("resources.id", ondelete="CASCADE"), index=True
    )
    state_of_health: Mapped[float] = mapped_column(Float)
    cumulative_throughput_kwh: Mapped[float] = mapped_column(Float)
    loss_fraction: Mapped[float] = mapped_column(Float, default=0.0)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    resource: Mapped[ResourceModel] = relationship(back_populates="soh_samples")


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------


class OptimizationRunModel(TimestampMixin, Base):
    """Record of a single optimization solve."""

    __tablename__ = "optimization_runs"
    __table_args__ = (Index("ix_opt_runs_type_ts", "problem_type", "created_at"),)

    problem_type: Mapped[str] = mapped_column(
        String(50), index=True
    )  # stochastic | realtime | distributed
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
    __table_args__ = (Index("ix_orders_market_status", "market", "status"),)

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

    trades: Mapped[list[TradeModel]] = relationship(
        back_populates="order", cascade="all, delete-orphan"
    )


class TradeModel(TimestampMixin, Base):
    """Persisted trade execution."""

    __tablename__ = "trades"
    __table_args__ = (Index("ix_trades_market_ts", "market", "created_at"),)

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

    order: Mapped[OrderModel] = relationship(back_populates="trades")


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
    #: Session generation. Embedded in every JWT as ``ver``; bumping it
    #: (password change, deactivation, role change, "log out everywhere")
    #: invalidates every token issued before. See vpp.auth.security.
    token_version: Mapped[int] = mapped_column(Integer, default=0)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


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
    #: First characters of the raw key (e.g. ``vpp_AbCd1234``), shown in
    #: listings so a key can be recognised; NULL for keys created before 0009.
    key_prefix: Mapped[str | None] = mapped_column(String(16), nullable=True)
    #: Updated at most once a minute per key (see get_api_key_user).
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class AuditLogModel(Base):
    """Append-only record of a security-relevant action (see :mod:`vpp.audit`).

    ``actor_id`` / ``actor_username`` are copies, not foreign keys, so the
    history survives deletion of the user. ``details_json`` never holds
    passwords, tokens or API keys (:func:`vpp.audit.sanitize_details`).
    Migration: 0012_audit_log.
    """

    __tablename__ = "audit_log"
    __table_args__ = (
        Index("ix_audit_log_actor_ts", "actor_id", "ts"),
        Index("ix_audit_log_action_ts", "action", "ts"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    actor_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    actor_username: Mapped[str | None] = mapped_column(String(64), nullable=True)
    action: Mapped[str] = mapped_column(String(64))
    target_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    target_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    client_ip: Mapped[str | None] = mapped_column(String(64), nullable=True)
    outcome: Mapped[str] = mapped_column(String(16), default="success")
    details_json: Mapped[str | None] = mapped_column(Text, nullable=True)


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
    __table_args__ = (Index("ix_event_log_type_ts", "event_type", "created_at"),)

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
    acknowledged_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    acknowledged_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    snoozed_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    resolved_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")


# ---------------------------------------------------------------------------
# Sites, customers, metering (sites map + customer portal)
# ---------------------------------------------------------------------------


class SiteModel(TimestampMixin, Base):
    """A physical premise grouping resources, optionally owned by a customer."""

    __tablename__ = "sites"

    name: Mapped[str] = mapped_column(String(255), index=True)
    lat: Mapped[float] = mapped_column(Float)
    lon: Mapped[float] = mapped_column(Float)
    region: Mapped[str | None] = mapped_column(String(128), nullable=True)
    address: Mapped[str | None] = mapped_column(String(512), nullable=True)
    # IANA zone used for billing-period boundaries and TOU classification.
    timezone: Mapped[str] = mapped_column(String(64), default="UTC")
    owner_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")


class CustomerProfileModel(TimestampMixin, Base):
    """Portal profile for a ``customer``-role user (1:1 with ``users``)."""

    __tablename__ = "customer_profiles"

    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), unique=True, index=True
    )
    name: Mapped[str] = mapped_column(String(255))
    email: Mapped[str | None] = mapped_column(String(320), nullable=True)
    address: Mapped[str | None] = mapped_column(String(512), nullable=True)
    tariff_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    baseline_kwh_per_month: Mapped[float | None] = mapped_column(Float, nullable=True)


class DRProgramModel(TimestampMixin, Base):
    """A demand-response program customers can enroll in."""

    __tablename__ = "dr_programs"

    name: Mapped[str] = mapped_column(String(255), unique=True)
    description: Mapped[str] = mapped_column(Text, default="")
    utility: Mapped[str | None] = mapped_column(String(255), nullable=True)
    incentive_per_event: Mapped[float | None] = mapped_column(Float, nullable=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True)


class ProgramEnrollmentModel(TimestampMixin, Base):
    """A customer's enrollment in a DR program (consent is recorded)."""

    __tablename__ = "program_enrollments"
    __table_args__ = (
        UniqueConstraint("user_id", "program_id", name="uq_program_enrollment_user_program"),
    )

    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    program_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("dr_programs.id", ondelete="CASCADE"), index=True
    )
    acknowledged_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class MeterReadingModel(TimestampMixin, Base):
    """Revenue-meter interval data for a site (grid import/export kWh)."""

    __tablename__ = "meter_readings"
    __table_args__ = (
        UniqueConstraint("site_id", "timestamp", name="uq_meter_readings_site_ts"),
        Index("ix_meter_readings_site_ts", "site_id", "timestamp"),
    )

    site_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("sites.id", ondelete="CASCADE"), index=True
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))  # interval start, UTC
    interval_minutes: Mapped[int] = mapped_column(Integer)
    import_kwh: Mapped[float] = mapped_column(Float, default=0.0)
    export_kwh: Mapped[float] = mapped_column(Float, default=0.0)


class ResourceTelemetryModel(TimestampMixin, Base):
    """Generic per-resource telemetry samples (power, optional SOC).

    Complements ``battery_states`` (MQTT battery telemetry) for resources
    whose live state arrives another way (Modbus polling, the telemetry
    ingest endpoint). Power sign convention matches ``Battery``: positive =
    consuming / charging, negative = discharging; generators report
    positive output.
    """

    __tablename__ = "resource_telemetry"
    __table_args__ = (Index("ix_resource_telemetry_resource_ts", "resource_id", "timestamp"),)

    resource_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("resources.id", ondelete="CASCADE"), index=True
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))  # UTC
    power_kw: Mapped[float] = mapped_column(Float)
    state_of_charge: Mapped[float | None] = mapped_column(Float, nullable=True)  # 0-1
    source: Mapped[str] = mapped_column(String(64), default="api")


class ConfigDocumentModel(TimestampMixin, Base):
    """Versioned VPPConfig documents applied via ``PUT /api/v1/config``.

    Append-only: the newest row is the live configuration, older rows are
    the audit trail.
    """

    __tablename__ = "config_documents"

    version: Mapped[int] = mapped_column(Integer, unique=True, index=True)  # 1, 2, 3, ...
    yaml: Mapped[str] = mapped_column(Text)
    hash: Mapped[str] = mapped_column(String(64))
    updated_by: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )


# ---------------------------------------------------------------------------
# V2G fleet + grid-protocol integration (migration 0007_v2g_protocols)
# ---------------------------------------------------------------------------


class V2GVehicleModel(TimestampMixin, Base):
    """A fleet EV (the V2G API's source of truth; ``id`` is the ``ev_id``).

    ``charge_point_id``/``connector_id`` bind the vehicle to an OCPP
    connector: explicitly via the API (``binding_source="manual"``), or
    automatically when a StartTransaction carries the vehicle's ``id_tag``
    (``binding_source="id_tag"``; released again when the connector goes
    back to ``Available``). One vehicle per connector.
    """

    __tablename__ = "v2g_vehicles"
    __table_args__ = (
        UniqueConstraint("charge_point_id", "connector_id", name="uq_v2g_vehicles_connector"),
    )

    name: Mapped[str] = mapped_column(String(255), default="")
    capacity_kwh: Mapped[float] = mapped_column(Float)
    current_soc: Mapped[float] = mapped_column(Float)  # 0-1
    min_soc: Mapped[float] = mapped_column(Float)
    target_soc: Mapped[float] = mapped_column(Float)
    max_charge_kw: Mapped[float] = mapped_column(Float)
    max_discharge_kw: Mapped[float] = mapped_column(Float)
    charge_efficiency: Mapped[float] = mapped_column(Float, default=0.92)
    discharge_efficiency: Mapped[float] = mapped_column(Float, default=0.92)
    degradation_cost_per_kwh: Mapped[float] = mapped_column(Float, default=0.02)
    v2g_capable: Mapped[bool] = mapped_column(Boolean, default=True)
    connection_state: Mapped[str] = mapped_column(String(32), default="disconnected")
    connected_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    departure_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    vehicle_make: Mapped[str] = mapped_column(String(128), default="")
    vehicle_model: Mapped[str] = mapped_column(String(128), default="")
    owner_id: Mapped[str] = mapped_column(String(64), default="")
    # OCPP binding
    id_tag: Mapped[str | None] = mapped_column(String(64), nullable=True, unique=True, index=True)
    charge_point_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    connector_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    binding_source: Mapped[str | None] = mapped_column(String(16), nullable=True)
    active_transaction_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    charger_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    # Live state (from OCPP MeterValues when bound, else the API)
    current_power_kw: Mapped[float] = mapped_column(Float, default=0.0)  # + charge / - discharge
    soc_source: Mapped[str] = mapped_column(String(16), default="api")
    soc_updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")


class V2GChargingSessionModel(TimestampMixin, Base):
    """One OCPP transaction (Start/StopTransaction), linked to a vehicle when known."""

    __tablename__ = "v2g_charging_sessions"

    vehicle_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("v2g_vehicles.id", ondelete="SET NULL"), nullable=True, index=True
    )
    charge_point_id: Mapped[str] = mapped_column(String(64), index=True)
    connector_id: Mapped[int] = mapped_column(Integer)
    transaction_id: Mapped[int] = mapped_column(Integer, index=True)
    id_tag: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="active")  # active | completed
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    stopped_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    meter_start_wh: Mapped[float | None] = mapped_column(Float, nullable=True)
    meter_stop_wh: Mapped[float | None] = mapped_column(Float, nullable=True)
    energy_kwh: Mapped[float | None] = mapped_column(Float, nullable=True)
    stop_reason: Mapped[str | None] = mapped_column(String(64), nullable=True)


class V2GScheduleModel(TimestampMixin, Base):
    """Audit record of a V2G schedule/dispatch and its per-charger delivery results."""

    __tablename__ = "v2g_schedules"

    kind: Mapped[str] = mapped_column(String(16), index=True)  # schedule | dispatch | dr
    method: Mapped[str] = mapped_column(String(32), default="")
    created_by: Mapped[str | None] = mapped_column(String(36), nullable=True)
    vehicle_count: Mapped[int] = mapped_column(Integer, default=0)
    total_cost: Mapped[float] = mapped_column(Float, default=0.0)
    total_revenue: Mapped[float] = mapped_column(Float, default=0.0)
    parameters_json: Mapped[str] = mapped_column(Text, default="{}")
    result_json: Mapped[str] = mapped_column(Text, default="{}")
    deliveries_json: Mapped[str] = mapped_column(Text, default="[]")


class DREventResponseModel(TimestampMixin, Base):
    """What the DR orchestrator did about a grid signal (append-only audit log).

    One row per decision: an OpenADR event received/opted, an IEEE 2030.5
    control observed, a dispatch run for an event window, or a release.
    """

    __tablename__ = "dr_event_responses"
    __table_args__ = (Index("ix_dr_event_responses_source", "protocol", "source_id"),)

    protocol: Mapped[str] = mapped_column(String(32))  # openadr | ieee2030_5
    source_id: Mapped[str] = mapped_column(String(255))  # event id / control mRID(s)
    revision: Mapped[int] = mapped_column(Integer, default=0)
    action: Mapped[str] = mapped_column(String(32), index=True)
    opt_type: Mapped[str | None] = mapped_column(String(16), nullable=True)
    signal_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    signal_level: Mapped[float | None] = mapped_column(Float, nullable=True)
    target_kw: Mapped[float | None] = mapped_column(Float, nullable=True)
    delivered_kw: Mapped[float | None] = mapped_column(Float, nullable=True)
    window_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    window_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    run_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("optimization_runs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    reason: Mapped[str] = mapped_column(Text, default="")
    details_json: Mapped[str] = mapped_column(Text, default="{}")


class V2GFlexibilityBidModel(TimestampMixin, Base):
    """A V2G flexibility bid (``POST /api/v1/v2g/bid``), shared by every API worker.

    A bid is *active* while ``available_until`` is in the future.
    """

    __tablename__ = "v2g_flexibility_bids"

    service: Mapped[str] = mapped_column(String(32), index=True)
    capacity_kw: Mapped[float] = mapped_column(Float)
    duration_hours: Mapped[float] = mapped_column(Float)
    price_per_kw: Mapped[float] = mapped_column(Float)
    available_from: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    available_until: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    fleet_id: Mapped[str] = mapped_column(String(64), default="")
    ev_ids_json: Mapped[str] = mapped_column(Text, default="[]")
    created_by: Mapped[str | None] = mapped_column(String(36), nullable=True)


# ---------------------------------------------------------------------------
# Multi-process coordination (vpp.cluster)
# ---------------------------------------------------------------------------


class ClusterLeaseModel(Base):
    """A named leadership lease: at most one API process holds each ``name``.

    Acquired/renewed atomically with ``UPDATE ... WHERE holder = :me OR
    expires_at < :now`` (see :mod:`vpp.cluster.lease`); times are UTC from
    the holders' clocks.
    """

    __tablename__ = "cluster_leases"

    name: Mapped[str] = mapped_column(String(64), primary_key=True)
    holder: Mapped[str] = mapped_column(String(128))
    acquired_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class ClusterCallModel(Base):
    """A call forwarded to the process holding lease ``target`` (see :mod:`vpp.cluster.rpc`).

    ``status``: pending -> running -> done | failed; a pending call the caller
    gave up on becomes ``cancelled``, one nobody claimed before
    ``deadline_at`` becomes ``expired``.
    """

    __tablename__ = "cluster_calls"
    __table_args__ = (Index("ix_cluster_calls_target_status", "target", "status"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    target: Mapped[str] = mapped_column(String(64))
    method: Mapped[str] = mapped_column(String(64))
    payload_json: Mapped[str] = mapped_column(Text, default="{}")
    status: Mapped[str] = mapped_column(String(16), default="pending")
    result_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    caller: Mapped[str] = mapped_column(String(128))
    executor: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    deadline_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ClusterEventModel(Base):
    """A WebSocket broadcast relayed to the other API workers (see :mod:`vpp.cluster.relay`).

    Short-lived: rows older than a few minutes are purged.
    """

    __tablename__ = "cluster_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    origin: Mapped[str] = mapped_column(String(128))
    channel: Mapped[str] = mapped_column(String(64))
    payload_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
