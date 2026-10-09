"""output caps held by the setpoint actuator

``output_caps``: one row per generation resource whose output the VPP
currently caps (a setpoint below its upper limit that the device accepted).
Loaded at startup so a restarted API still recognises its own curtailment.

Revision ID: 0013_output_caps
Revises: 0012_audit_log
Create Date: 2026-10-09 12:00:00
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa
from alembic import op

if TYPE_CHECKING:
    from collections.abc import Sequence

revision: str = "0013_output_caps"
down_revision: str | Sequence[str] | None = "0012_audit_log"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "output_caps",
        sa.Column("resource_id", sa.String(36), primary_key=True),
        sa.Column("cap_kw", sa.Float(), nullable=False),
        sa.Column("upper_limit_kw", sa.Float(), nullable=False),
        sa.Column("profile", sa.String(32), nullable=False),
        sa.Column("revert_timeout_s", sa.Float(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("output_caps")
