"""add alert_rules and alerts tables

Persistence for the alerting service: rules evaluated against live telemetry
(``AlertRuleModel``) and fired alerts with their operator lifecycle
(``AlertModel``).

Revision ID: 0005_add_alerts
Revises: 0004_add_tariffs
Create Date: 2026-09-28 00:00:00
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa
from alembic import op

if TYPE_CHECKING:
    from collections.abc import Sequence

revision: str = "0005_add_alerts"
down_revision: str | Sequence[str] | None = "0004_add_tariffs"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _timestamps() -> list[sa.Column]:
    return [
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    ]


def upgrade() -> None:
    op.create_table(
        "alert_rules",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column("rule_type", sa.String(32), nullable=False, server_default="threshold"),
        sa.Column("metric", sa.String(64), nullable=False),
        sa.Column("comparison", sa.String(4), nullable=False, server_default=">"),
        sa.Column("threshold", sa.Float(), nullable=False, server_default="0"),
        sa.Column("severity", sa.String(16), nullable=False, server_default="warning"),
        sa.Column("rate_window_s", sa.Float(), nullable=False, server_default="60"),
        sa.Column("rate_threshold", sa.Float(), nullable=False, server_default="0"),
        sa.Column("z_score_threshold", sa.Float(), nullable=False, server_default="3"),
        sa.Column("cooldown_s", sa.Float(), nullable=False, server_default="300"),
        sa.Column("resource_id", sa.String(36), nullable=True),
        sa.Column("auto_resolve", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        *_timestamps(),
    )
    op.create_index("ix_alert_rules_name", "alert_rules", ["name"], unique=True)
    op.create_index("ix_alert_rules_metric", "alert_rules", ["metric"])

    op.create_table(
        "alerts",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("rule_id", sa.String(36), nullable=True),
        sa.Column("rule_name", sa.String(255), nullable=False, server_default=""),
        sa.Column("severity", sa.String(16), nullable=False, server_default="warning"),
        sa.Column("status", sa.String(16), nullable=False, server_default="active"),
        sa.Column("title", sa.String(255), nullable=False, server_default=""),
        sa.Column("message", sa.Text(), nullable=False, server_default=""),
        sa.Column("source", sa.String(255), nullable=False, server_default="system"),
        sa.Column("source_kind", sa.String(32), nullable=False, server_default="system"),
        sa.Column("metric", sa.String(64), nullable=True),
        sa.Column("value", sa.Float(), nullable=True),
        sa.Column("threshold", sa.Float(), nullable=True),
        sa.Column("occurrences", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("fired_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_fired_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("acknowledged_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("acknowledged_by", sa.String(255), nullable=True),
        sa.Column("snoozed_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("resolved_by", sa.String(255), nullable=True),
        sa.Column("metadata_json", sa.Text(), nullable=False, server_default="{}"),
        *_timestamps(),
    )
    op.create_index("ix_alerts_severity", "alerts", ["severity"])
    op.create_index("ix_alerts_source", "alerts", ["source"])
    op.create_index("ix_alerts_fired_at", "alerts", ["fired_at"])
    op.create_index("ix_alerts_status_fired", "alerts", ["status", "fired_at"])
    op.create_index("ix_alerts_rule_source", "alerts", ["rule_id", "source"])


def downgrade() -> None:
    for name in (
        "ix_alerts_rule_source",
        "ix_alerts_status_fired",
        "ix_alerts_fired_at",
        "ix_alerts_source",
        "ix_alerts_severity",
    ):
        op.drop_index(name, table_name="alerts")
    op.drop_table("alerts")
    op.drop_index("ix_alert_rules_metric", table_name="alert_rules")
    op.drop_index("ix_alert_rules_name", table_name="alert_rules")
    op.drop_table("alert_rules")
