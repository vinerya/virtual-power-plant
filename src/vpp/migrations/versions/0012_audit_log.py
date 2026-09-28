"""audit log of security-relevant actions

``audit_log``: logins (success / failure), logouts, session and API-key
revocation, user administration, password changes and privileged control
actions (DR / device dispatch, market orders). Actor columns are copies,
not foreign keys, so the history survives user deletion.

Revision ID: 0012_audit_log
Revises: 0011_shared_rate_limits
Create Date: 2026-09-28 22:00:00
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa
from alembic import op

if TYPE_CHECKING:
    from collections.abc import Sequence

revision: str = "0012_audit_log"
down_revision: str | Sequence[str] | None = "0011_shared_rate_limits"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "audit_log",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("actor_id", sa.String(36), nullable=True),
        sa.Column("actor_username", sa.String(64), nullable=True),
        sa.Column("action", sa.String(64), nullable=False),
        sa.Column("target_type", sa.String(32), nullable=True),
        sa.Column("target_id", sa.String(128), nullable=True),
        sa.Column("client_ip", sa.String(64), nullable=True),
        sa.Column("outcome", sa.String(16), nullable=False, server_default="success"),
        sa.Column("details_json", sa.Text(), nullable=True),
    )
    op.create_index("ix_audit_log_ts", "audit_log", ["ts"])
    op.create_index("ix_audit_log_actor_ts", "audit_log", ["actor_id", "ts"])
    op.create_index("ix_audit_log_action_ts", "audit_log", ["action", "ts"])


def downgrade() -> None:
    op.drop_index("ix_audit_log_action_ts", table_name="audit_log")
    op.drop_index("ix_audit_log_actor_ts", table_name="audit_log")
    op.drop_index("ix_audit_log_ts", table_name="audit_log")
    op.drop_table("audit_log")
