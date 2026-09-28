"""add sites, customer portal, metering, telemetry history and config documents

Tables backing the sites map, the customer portal (profiles, DR programs,
enrollments, revenue-meter intervals), generic per-resource telemetry
history, and versioned VPPConfig documents; plus ``resources.site_id``.

Revision ID: 0006_add_sites_customers
Revises: 0005_add_alerts
Create Date: 2026-09-28 00:00:00
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa
from alembic import op

if TYPE_CHECKING:
    from collections.abc import Sequence

revision: str = "0006_add_sites_customers"
down_revision: str | Sequence[str] | None = "0005_add_alerts"
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
        "sites",
        *_base(),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("lat", sa.Float(), nullable=False),
        sa.Column("lon", sa.Float(), nullable=False),
        sa.Column("region", sa.String(128), nullable=True),
        sa.Column("address", sa.String(512), nullable=True),
        sa.Column("timezone", sa.String(64), nullable=False, server_default="UTC"),
        sa.Column(
            "owner_id",
            sa.String(36),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("metadata_json", sa.Text(), nullable=False, server_default="{}"),
    )
    op.create_index("ix_sites_name", "sites", ["name"])
    op.create_index("ix_sites_owner_id", "sites", ["owner_id"])

    op.create_table(
        "customer_profiles",
        *_base(),
        sa.Column(
            "user_id",
            sa.String(36),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("email", sa.String(320), nullable=True),
        sa.Column("address", sa.String(512), nullable=True),
        sa.Column("tariff_id", sa.String(36), nullable=True),
        sa.Column("baseline_kwh_per_month", sa.Float(), nullable=True),
    )
    op.create_index(
        "ix_customer_profiles_user_id", "customer_profiles", ["user_id"], unique=True
    )

    op.create_table(
        "dr_programs",
        *_base(),
        sa.Column("name", sa.String(255), nullable=False, unique=True),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column("utility", sa.String(255), nullable=True),
        sa.Column("incentive_per_event", sa.Float(), nullable=True),
        sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.true()),
    )

    op.create_table(
        "program_enrollments",
        *_base(),
        sa.Column(
            "user_id",
            sa.String(36),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "program_id",
            sa.String(36),
            sa.ForeignKey("dr_programs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("acknowledged_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "user_id", "program_id", name="uq_program_enrollment_user_program"
        ),
    )
    op.create_index("ix_program_enrollments_user_id", "program_enrollments", ["user_id"])
    op.create_index(
        "ix_program_enrollments_program_id", "program_enrollments", ["program_id"]
    )

    op.create_table(
        "meter_readings",
        *_base(),
        sa.Column(
            "site_id",
            sa.String(36),
            sa.ForeignKey("sites.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("interval_minutes", sa.Integer(), nullable=False),
        sa.Column("import_kwh", sa.Float(), nullable=False, server_default="0"),
        sa.Column("export_kwh", sa.Float(), nullable=False, server_default="0"),
        sa.UniqueConstraint("site_id", "timestamp", name="uq_meter_readings_site_ts"),
    )
    op.create_index("ix_meter_readings_site_id", "meter_readings", ["site_id"])
    op.create_index("ix_meter_readings_site_ts", "meter_readings", ["site_id", "timestamp"])

    op.create_table(
        "resource_telemetry",
        *_base(),
        sa.Column(
            "resource_id",
            sa.String(36),
            sa.ForeignKey("resources.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("power_kw", sa.Float(), nullable=False),
        sa.Column("state_of_charge", sa.Float(), nullable=True),
        sa.Column("source", sa.String(64), nullable=False, server_default="api"),
    )
    op.create_index(
        "ix_resource_telemetry_resource_id", "resource_telemetry", ["resource_id"]
    )
    op.create_index(
        "ix_resource_telemetry_resource_ts", "resource_telemetry", ["resource_id", "timestamp"]
    )

    op.create_table(
        "config_documents",
        *_base(),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("yaml", sa.Text(), nullable=False),
        sa.Column("hash", sa.String(64), nullable=False),
        sa.Column(
            "updated_by",
            sa.String(36),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.create_index(
        "ix_config_documents_version", "config_documents", ["version"], unique=True
    )

    with op.batch_alter_table("resources") as batch:
        batch.add_column(sa.Column("site_id", sa.String(36), nullable=True))
        batch.create_foreign_key(
            "fk_resources_site_id_sites", "sites", ["site_id"], ["id"], ondelete="SET NULL"
        )
        batch.create_index("ix_resources_site_id", ["site_id"])


def downgrade() -> None:
    with op.batch_alter_table("resources") as batch:
        batch.drop_index("ix_resources_site_id")
        batch.drop_constraint("fk_resources_site_id_sites", type_="foreignkey")
        batch.drop_column("site_id")

    op.drop_index("ix_config_documents_version", table_name="config_documents")
    op.drop_table("config_documents")
    op.drop_index("ix_resource_telemetry_resource_ts", table_name="resource_telemetry")
    op.drop_index("ix_resource_telemetry_resource_id", table_name="resource_telemetry")
    op.drop_table("resource_telemetry")
    op.drop_index("ix_meter_readings_site_ts", table_name="meter_readings")
    op.drop_index("ix_meter_readings_site_id", table_name="meter_readings")
    op.drop_table("meter_readings")
    op.drop_index("ix_program_enrollments_program_id", table_name="program_enrollments")
    op.drop_index("ix_program_enrollments_user_id", table_name="program_enrollments")
    op.drop_table("program_enrollments")
    op.drop_table("dr_programs")
    op.drop_index("ix_customer_profiles_user_id", table_name="customer_profiles")
    op.drop_table("customer_profiles")
    op.drop_index("ix_sites_owner_id", table_name="sites")
    op.drop_index("ix_sites_name", table_name="sites")
    op.drop_table("sites")
