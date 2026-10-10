"""partial index for the newest state of charge per resource

``ix_resource_telemetry_resource_soc_ts`` on ``resource_telemetry
(resource_id, timestamp, state_of_charge) WHERE state_of_charge IS NOT
NULL``: lets the "newest SOC per battery" lookup behind
``GET /api/v1/resources`` seek
straight to the latest sample that carries a state of charge instead of
walking back over every power-only sample (found with ``vpp loadtest``,
see docs/performance.md). It includes ``state_of_charge`` so it covers the
lookup: SQLite then prefers it even without ``ANALYZE`` statistics. Index
only; the downgrade drops it without losing data.

Revision ID: 0014_telemetry_soc_index
Revises: 0013_output_caps
Create Date: 2026-10-10 12:00:00
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa
from alembic import op

if TYPE_CHECKING:
    from collections.abc import Sequence

revision: str = "0014_telemetry_soc_index"
down_revision: str | Sequence[str] | None = "0013_output_caps"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_WHERE = sa.text("state_of_charge IS NOT NULL")


def upgrade() -> None:
    op.create_index(
        "ix_resource_telemetry_resource_soc_ts",
        "resource_telemetry",
        ["resource_id", "timestamp", "state_of_charge"],
        sqlite_where=_WHERE,
        postgresql_where=_WHERE,
    )


def downgrade() -> None:
    op.drop_index("ix_resource_telemetry_resource_soc_ts", table_name="resource_telemetry")
