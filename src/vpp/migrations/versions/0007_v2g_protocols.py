"""add persistent V2G fleet, OCPP charging sessions, V2G schedules, DR responses

Moves the V2G fleet out of process memory (``v2g_vehicles``, including the
EV <-> OCPP charge point/connector binding), records OCPP transactions
(``v2g_charging_sessions``), schedule/dispatch delivery audits
(``v2g_schedules``) and the DR orchestrator's decisions
(``dr_event_responses``).

Revision ID: 0007_v2g_protocols
Revises: 0006_add_sites_customers
Create Date: 2026-09-28 12:00:00
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa
from alembic import op

if TYPE_CHECKING:
    from collections.abc import Sequence

revision: str = "0007_v2g_protocols"
down_revision: str | Sequence[str] | None = "0006_add_sites_customers"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _base() -> list[sa.Column]:
    return [
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    ]


def upgrade() -> None:
    op.create_table(
        "v2g_vehicles",
        *_base(),
        sa.Column("name", sa.String(255), nullable=False, server_default=""),
        sa.Column("capacity_kwh", sa.Float(), nullable=False),
        sa.Column("current_soc", sa.Float(), nullable=False),
        sa.Column("min_soc", sa.Float(), nullable=False),
        sa.Column("target_soc", sa.Float(), nullable=False),
        sa.Column("max_charge_kw", sa.Float(), nullable=False),
        sa.Column("max_discharge_kw", sa.Float(), nullable=False),
        sa.Column("charge_efficiency", sa.Float(), nullable=False, server_default="0.92"),
        sa.Column("discharge_efficiency", sa.Float(), nullable=False, server_default="0.92"),
        sa.Column("degradation_cost_per_kwh", sa.Float(), nullable=False, server_default="0.02"),
        sa.Column("v2g_capable", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column(
            "connection_state", sa.String(32), nullable=False, server_default="disconnected"
        ),
        sa.Column("connected_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("departure_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("vehicle_make", sa.String(128), nullable=False, server_default=""),
        sa.Column("vehicle_model", sa.String(128), nullable=False, server_default=""),
        sa.Column("owner_id", sa.String(64), nullable=False, server_default=""),
        sa.Column("id_tag", sa.String(64), nullable=True),
        sa.Column("charge_point_id", sa.String(64), nullable=True),
        sa.Column("connector_id", sa.Integer(), nullable=True),
        sa.Column("binding_source", sa.String(16), nullable=True),
        sa.Column("active_transaction_id", sa.Integer(), nullable=True),
        sa.Column("charger_status", sa.String(32), nullable=True),
        sa.Column("current_power_kw", sa.Float(), nullable=False, server_default="0"),
        sa.Column("soc_source", sa.String(16), nullable=False, server_default="api"),
        sa.Column("soc_updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("metadata_json", sa.Text(), nullable=False, server_default="{}"),
        sa.UniqueConstraint("charge_point_id", "connector_id", name="uq_v2g_vehicles_connector"),
    )
    op.create_index("ix_v2g_vehicles_id_tag", "v2g_vehicles", ["id_tag"], unique=True)
    op.create_index("ix_v2g_vehicles_charge_point_id", "v2g_vehicles", ["charge_point_id"])
    op.create_index(
        "ix_v2g_vehicles_active_transaction_id", "v2g_vehicles", ["active_transaction_id"]
    )

    op.create_table(
        "v2g_charging_sessions",
        *_base(),
        sa.Column(
            "vehicle_id",
            sa.String(36),
            sa.ForeignKey("v2g_vehicles.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("charge_point_id", sa.String(64), nullable=False),
        sa.Column("connector_id", sa.Integer(), nullable=False),
        sa.Column("transaction_id", sa.Integer(), nullable=False),
        sa.Column("id_tag", sa.String(64), nullable=True),
        sa.Column("status", sa.String(16), nullable=False, server_default="active"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("stopped_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("meter_start_wh", sa.Float(), nullable=True),
        sa.Column("meter_stop_wh", sa.Float(), nullable=True),
        sa.Column("energy_kwh", sa.Float(), nullable=True),
        sa.Column("stop_reason", sa.String(64), nullable=True),
    )
    op.create_index("ix_v2g_charging_sessions_vehicle_id", "v2g_charging_sessions", ["vehicle_id"])
    op.create_index(
        "ix_v2g_charging_sessions_charge_point_id", "v2g_charging_sessions", ["charge_point_id"]
    )
    op.create_index(
        "ix_v2g_charging_sessions_transaction_id", "v2g_charging_sessions", ["transaction_id"]
    )

    op.create_table(
        "v2g_schedules",
        *_base(),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("method", sa.String(32), nullable=False, server_default=""),
        sa.Column("created_by", sa.String(36), nullable=True),
        sa.Column("vehicle_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_cost", sa.Float(), nullable=False, server_default="0"),
        sa.Column("total_revenue", sa.Float(), nullable=False, server_default="0"),
        sa.Column("parameters_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("result_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("deliveries_json", sa.Text(), nullable=False, server_default="[]"),
    )
    op.create_index("ix_v2g_schedules_kind", "v2g_schedules", ["kind"])

    op.create_table(
        "dr_event_responses",
        *_base(),
        sa.Column("protocol", sa.String(32), nullable=False),
        sa.Column("source_id", sa.String(255), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("action", sa.String(32), nullable=False),
        sa.Column("opt_type", sa.String(16), nullable=True),
        sa.Column("signal_type", sa.String(64), nullable=True),
        sa.Column("signal_level", sa.Float(), nullable=True),
        sa.Column("target_kw", sa.Float(), nullable=True),
        sa.Column("delivered_kw", sa.Float(), nullable=True),
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=True),
        sa.Column("window_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "run_id",
            sa.String(36),
            sa.ForeignKey("optimization_runs.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("reason", sa.Text(), nullable=False, server_default=""),
        sa.Column("details_json", sa.Text(), nullable=False, server_default="{}"),
    )
    op.create_index(
        "ix_dr_event_responses_source", "dr_event_responses", ["protocol", "source_id"]
    )
    op.create_index("ix_dr_event_responses_action", "dr_event_responses", ["action"])
    op.create_index("ix_dr_event_responses_run_id", "dr_event_responses", ["run_id"])


def downgrade() -> None:
    op.drop_index("ix_dr_event_responses_run_id", table_name="dr_event_responses")
    op.drop_index("ix_dr_event_responses_action", table_name="dr_event_responses")
    op.drop_index("ix_dr_event_responses_source", table_name="dr_event_responses")
    op.drop_table("dr_event_responses")
    op.drop_index("ix_v2g_schedules_kind", table_name="v2g_schedules")
    op.drop_table("v2g_schedules")
    op.drop_index("ix_v2g_charging_sessions_transaction_id", table_name="v2g_charging_sessions")
    op.drop_index("ix_v2g_charging_sessions_charge_point_id", table_name="v2g_charging_sessions")
    op.drop_index("ix_v2g_charging_sessions_vehicle_id", table_name="v2g_charging_sessions")
    op.drop_table("v2g_charging_sessions")
    op.drop_index("ix_v2g_vehicles_active_transaction_id", table_name="v2g_vehicles")
    op.drop_index("ix_v2g_vehicles_charge_point_id", table_name="v2g_vehicles")
    op.drop_index("ix_v2g_vehicles_id_tag", table_name="v2g_vehicles")
    op.drop_table("v2g_vehicles")
