"""multi-worker coordination tables; persisted V2G flexibility bids

- ``v2g_flexibility_bids``: bids from ``POST /api/v1/v2g/bid``, previously
  held in memory per process.
- ``cluster_leases``: named leadership leases for singleton background work.
- ``cluster_calls``: calls forwarded to the process holding a lease.
- ``cluster_events``: WebSocket broadcasts relayed between API workers.

Revision ID: 0010_cluster_and_v2g_bids
Revises: 0009_user_management
Create Date: 2026-09-28 00:00:00
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa
from alembic import op

if TYPE_CHECKING:
    from collections.abc import Sequence

revision: str = "0010_cluster_and_v2g_bids"
down_revision: str | Sequence[str] | None = "0009_user_management"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "v2g_flexibility_bids",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("service", sa.String(32), nullable=False),
        sa.Column("capacity_kw", sa.Float(), nullable=False),
        sa.Column("duration_hours", sa.Float(), nullable=False),
        sa.Column("price_per_kw", sa.Float(), nullable=False),
        sa.Column("available_from", sa.DateTime(timezone=True), nullable=False),
        sa.Column("available_until", sa.DateTime(timezone=True), nullable=False),
        sa.Column("fleet_id", sa.String(64), nullable=False, server_default=""),
        sa.Column("ev_ids_json", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("created_by", sa.String(36), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.create_index("ix_v2g_flexibility_bids_service", "v2g_flexibility_bids", ["service"])
    op.create_index(
        "ix_v2g_flexibility_bids_available_until", "v2g_flexibility_bids", ["available_until"]
    )

    op.create_table(
        "cluster_leases",
        sa.Column("name", sa.String(64), primary_key=True),
        sa.Column("holder", sa.String(128), nullable=False),
        sa.Column("acquired_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "cluster_calls",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("target", sa.String(64), nullable=False),
        sa.Column("method", sa.String(64), nullable=False),
        sa.Column("payload_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("status", sa.String(16), nullable=False, server_default="pending"),
        sa.Column("result_json", sa.Text(), nullable=True),
        sa.Column("error_json", sa.Text(), nullable=True),
        sa.Column("caller", sa.String(128), nullable=False),
        sa.Column("executor", sa.String(128), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("deadline_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_cluster_calls_target_status", "cluster_calls", ["target", "status"])

    op.create_table(
        "cluster_events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("origin", sa.String(128), nullable=False),
        sa.Column("channel", sa.String(64), nullable=False),
        sa.Column("payload_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_cluster_events_created_at", "cluster_events", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_cluster_events_created_at", table_name="cluster_events")
    op.drop_table("cluster_events")
    op.drop_index("ix_cluster_calls_target_status", table_name="cluster_calls")
    op.drop_table("cluster_calls")
    op.drop_table("cluster_leases")
    op.drop_index("ix_v2g_flexibility_bids_available_until", table_name="v2g_flexibility_bids")
    op.drop_index("ix_v2g_flexibility_bids_service", table_name="v2g_flexibility_bids")
    op.drop_table("v2g_flexibility_bids")
