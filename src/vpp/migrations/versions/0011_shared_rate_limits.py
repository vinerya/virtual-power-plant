"""shared rate-limit and login-throttle counters

- ``shared_rate_limits``: per-client-IP request counters and per-username
  failed-login counters shared by every API worker (used when
  ``VPP_RATE_LIMIT_BACKEND`` resolves to ``database``, the default with
  ``VPP_API_WORKERS > 1``).

Revision ID: 0011_shared_rate_limits
Revises: 0010_cluster_and_v2g_bids
Create Date: 2026-09-28 00:00:00
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa
from alembic import op

if TYPE_CHECKING:
    from collections.abc import Sequence

revision: str = "0011_shared_rate_limits"
down_revision: str | Sequence[str] | None = "0010_cluster_and_v2g_bids"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "shared_rate_limits",
        sa.Column("key", sa.String(255), primary_key=True),
        sa.Column("hits", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("prev_hits", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("window_start", sa.Float(), nullable=False),
        sa.Column("locked_until", sa.Float(), nullable=False, server_default="0"),
        sa.Column("expires_at", sa.Float(), nullable=False),
    )
    op.create_index("ix_shared_rate_limits_expires_at", "shared_rate_limits", ["expires_at"])


def downgrade() -> None:
    op.drop_index("ix_shared_rate_limits_expires_at", table_name="shared_rate_limits")
    op.drop_table("shared_rate_limits")
