"""add tariffs table; align timestamp nullability with the ORM models

Two pieces of model/migration drift are closed here:

1. ``TariffRow`` (``tariffs``) was added to :mod:`vpp.db.models` without a
   migration, so deployments running with ``VPP_USE_ALEMBIC=1`` never got
   the table and every tariff endpoint failed.  This revision creates it
   exactly as the model defines it, including its indexes.

2. ``TimestampMixin.created_at`` / ``updated_at`` (and the ``timestamp``
   columns on ``battery_states`` / ``battery_soh_samples``) are ``NOT NULL``
   in the ORM but were created nullable by 0001/0002.  Existing NULLs (which
   should not exist, since every column has a ``now()`` server default) are
   backfilled before the constraint is tightened.

``tests/test_alembic_drift.py`` asserts that ``alembic upgrade head``
produces a schema identical to ``Base.metadata``, so future drift fails CI.

Revision ID: 0004_add_tariffs
Revises: 0003_add_nominal_energy
Create Date: 2026-09-28 00:00:00
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa
from alembic import op

if TYPE_CHECKING:
    from collections.abc import Sequence

revision: str = "0004_add_tariffs"
down_revision: str | Sequence[str] | None = "0003_add_nominal_energy"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# Tables whose TimestampMixin columns were created nullable by earlier revisions.
_TIMESTAMP_TABLES: tuple[str, ...] = (
    "resources",
    "battery_states",
    "battery_soh_samples",
    "optimization_runs",
    "orders",
    "trades",
    "users",
    "api_keys",
    "event_log",
)

# Extra ``timestamp`` columns that the ORM declares NOT NULL.
_EXTRA_TIMESTAMP_COLUMNS: dict[str, tuple[str, ...]] = {
    "battery_states": ("timestamp",),
    "battery_soh_samples": ("timestamp",),
}


def _columns_for(table: str) -> tuple[str, ...]:
    return (*_EXTRA_TIMESTAMP_COLUMNS.get(table, ()), "created_at", "updated_at")


def upgrade() -> None:
    op.create_table(
        "tariffs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("utility", sa.String(255), nullable=False, server_default=""),
        sa.Column("urdb_label", sa.String(128), nullable=True),
        sa.Column("urdb_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("effective_date", sa.Date(), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_tariffs_name", "tariffs", ["name"])
    op.create_index("ix_tariffs_utility", "tariffs", ["utility"])
    op.create_index("ix_tariffs_urdb_label", "tariffs", ["urdb_label"])

    for table in _TIMESTAMP_TABLES:
        for column in _columns_for(table):
            op.execute(
                sa.text(
                    f"UPDATE {table} SET {column} = CURRENT_TIMESTAMP "
                    f"WHERE {column} IS NULL"
                )
            )
        with op.batch_alter_table(table) as batch:
            for column in _columns_for(table):
                batch.alter_column(
                    column,
                    existing_type=sa.DateTime(timezone=True),
                    nullable=False,
                )


def downgrade() -> None:
    for table in reversed(_TIMESTAMP_TABLES):
        with op.batch_alter_table(table) as batch:
            for column in _columns_for(table):
                batch.alter_column(
                    column,
                    existing_type=sa.DateTime(timezone=True),
                    nullable=True,
                )

    op.drop_index("ix_tariffs_urdb_label", table_name="tariffs")
    op.drop_index("ix_tariffs_utility", table_name="tariffs")
    op.drop_index("ix_tariffs_name", table_name="tariffs")
    op.drop_table("tariffs")
