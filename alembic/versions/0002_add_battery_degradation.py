"""add battery degradation columns and SOH samples table (M3)

Adds the four columns introduced by Milestone 3 of the degradation track to
the ``resources`` table, plus the new ``battery_soh_samples`` time-series
table.  These were originally created via ``Base.metadata.create_all`` in
M3; this migration captures the same change for existing deployments.

Revision ID: 0002_add_battery_degradation
Revises: 0001_baseline
Create Date: 2025-11-15 00:00:00
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0002_add_battery_degradation"
down_revision: Union[str, Sequence[str], None] = "0001_baseline"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("resources") as batch:
        batch.add_column(
            sa.Column(
                "state_of_health", sa.Float(), nullable=False, server_default="1.0"
            )
        )
        batch.add_column(
            sa.Column(
                "cumulative_throughput_kwh",
                sa.Float(),
                nullable=False,
                server_default="0.0",
            )
        )
        batch.add_column(
            sa.Column(
                "last_degradation_update",
                sa.DateTime(timezone=True),
                nullable=True,
            )
        )
        batch.add_column(sa.Column("chemistry", sa.String(16), nullable=True))

    op.create_table(
        "battery_soh_samples",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "resource_id",
            sa.String(36),
            sa.ForeignKey("resources.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("state_of_health", sa.Float(), nullable=False),
        sa.Column("cumulative_throughput_kwh", sa.Float(), nullable=False),
        sa.Column("loss_fraction", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("timestamp", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index(
        "ix_battery_soh_samples_resource_id", "battery_soh_samples", ["resource_id"]
    )
    op.create_index(
        "ix_battery_soh_samples_resource_ts",
        "battery_soh_samples",
        ["resource_id", "timestamp"],
    )


def downgrade() -> None:
    op.drop_index("ix_battery_soh_samples_resource_ts", table_name="battery_soh_samples")
    op.drop_index("ix_battery_soh_samples_resource_id", table_name="battery_soh_samples")
    op.drop_table("battery_soh_samples")

    with op.batch_alter_table("resources") as batch:
        batch.drop_column("chemistry")
        batch.drop_column("last_degradation_update")
        batch.drop_column("cumulative_throughput_kwh")
        batch.drop_column("state_of_health")
