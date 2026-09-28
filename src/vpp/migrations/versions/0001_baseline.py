"""baseline -- pre-M3 schema

Captures the schema as it existed *before* battery degradation persistence
(M3) was introduced.  This exists primarily so that older deployments can
``alembic stamp 0001`` and then run ``alembic upgrade head`` to pick up
the M3 + M4 changes.

For fresh installations this revision is run from scratch and creates the
full baseline schema; subsequent migrations layer on the new columns.

Revision ID: 0001_baseline
Revises:
Create Date: 2025-11-01 00:00:00
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0001_baseline"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "resources",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("resource_type", sa.String(50), nullable=False),
        sa.Column("rated_power", sa.Float(), nullable=False),
        sa.Column("online", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("current_power", sa.Float(), nullable=False, server_default="0"),
        sa.Column("efficiency", sa.Float(), nullable=False, server_default="0.95"),
        sa.Column("config_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("metadata_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("name"),
    )
    op.create_index("ix_resources_name", "resources", ["name"], unique=True)
    op.create_index("ix_resources_resource_type", "resources", ["resource_type"])

    op.create_table(
        "battery_states",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("resource_id", sa.String(36), sa.ForeignKey("resources.id", ondelete="CASCADE"), nullable=False),
        sa.Column("soc", sa.Float(), nullable=False),
        sa.Column("soh", sa.Float(), nullable=False, server_default="100.0"),
        sa.Column("temperature", sa.Float(), nullable=False, server_default="25.0"),
        sa.Column("voltage", sa.Float(), nullable=False, server_default="0"),
        sa.Column("current", sa.Float(), nullable=False, server_default="0"),
        sa.Column("power", sa.Float(), nullable=False, server_default="0"),
        sa.Column("timestamp", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_battery_states_resource_id", "battery_states", ["resource_id"])
    op.create_index("ix_battery_states_resource_ts", "battery_states", ["resource_id", "timestamp"])

    op.create_table(
        "optimization_runs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("problem_type", sa.String(50), nullable=False),
        sa.Column("status", sa.String(30), nullable=False),
        sa.Column("objective_value", sa.Float(), nullable=False, server_default="0"),
        sa.Column("solve_time_ms", sa.Float(), nullable=False, server_default="0"),
        sa.Column("solver", sa.String(100), nullable=False, server_default=""),
        sa.Column("fallback_used", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("solution_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("parameters_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_optimization_runs_problem_type", "optimization_runs", ["problem_type"])
    op.create_index("ix_opt_runs_type_ts", "optimization_runs", ["problem_type", "created_at"])

    op.create_table(
        "orders",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("order_type", sa.String(30), nullable=False),
        sa.Column("market", sa.String(100), nullable=False),
        sa.Column("side", sa.String(10), nullable=False),
        sa.Column("quantity", sa.Float(), nullable=False),
        sa.Column("price", sa.Float(), nullable=False, server_default="0"),
        sa.Column("status", sa.String(30), nullable=False, server_default="pending"),
        sa.Column("filled_quantity", sa.Float(), nullable=False, server_default="0"),
        sa.Column("remaining_quantity", sa.Float(), nullable=False, server_default="0"),
        sa.Column("average_price", sa.Float(), nullable=False, server_default="0"),
        sa.Column("time_in_force", sa.String(10), nullable=False, server_default="GTC"),
        sa.Column("metadata_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_orders_market", "orders", ["market"])
    op.create_index("ix_orders_status", "orders", ["status"])
    op.create_index("ix_orders_market_status", "orders", ["market", "status"])

    op.create_table(
        "trades",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("order_id", sa.String(36), sa.ForeignKey("orders.id", ondelete="CASCADE"), nullable=False),
        sa.Column("market", sa.String(100), nullable=False),
        sa.Column("side", sa.String(10), nullable=False),
        sa.Column("quantity", sa.Float(), nullable=False),
        sa.Column("price", sa.Float(), nullable=False),
        sa.Column("fees", sa.Float(), nullable=False, server_default="0"),
        sa.Column("strategy", sa.String(100), nullable=False, server_default=""),
        sa.Column("realized_pnl", sa.Float(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_trades_order_id", "trades", ["order_id"])
    op.create_index("ix_trades_market", "trades", ["market"])
    op.create_index("ix_trades_market_ts", "trades", ["market", "created_at"])

    op.create_table(
        "users",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("username", sa.String(64), nullable=False),
        sa.Column("hashed_password", sa.String(256), nullable=False),
        sa.Column("role", sa.String(30), nullable=False, server_default="viewer"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("username"),
    )
    op.create_index("ix_users_username", "users", ["username"], unique=True)

    op.create_table(
        "api_keys",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("hashed_key", sa.String(256), nullable=False),
        sa.Column("role", sa.String(30), nullable=False, server_default="viewer"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("hashed_key"),
    )
    op.create_index("ix_api_keys_user_id", "api_keys", ["user_id"])
    op.create_index("ix_api_keys_hashed_key", "api_keys", ["hashed_key"], unique=True)

    op.create_table(
        "event_log",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("event_type", sa.String(50), nullable=False),
        sa.Column("resource_id", sa.String(36), nullable=True),
        sa.Column("details_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("severity", sa.String(20), nullable=False, server_default="info"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_event_log_event_type", "event_log", ["event_type"])
    op.create_index("ix_event_log_resource_id", "event_log", ["resource_id"])
    op.create_index("ix_event_log_type_ts", "event_log", ["event_type", "created_at"])


def downgrade() -> None:
    op.drop_table("event_log")
    op.drop_table("api_keys")
    op.drop_table("users")
    op.drop_table("trades")
    op.drop_table("orders")
    op.drop_table("optimization_runs")
    op.drop_table("battery_states")
    op.drop_table("resources")
