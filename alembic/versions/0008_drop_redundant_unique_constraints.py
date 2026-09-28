"""drop redundant UNIQUE constraints duplicated by unique indexes (PostgreSQL)

The baseline revision declared both ``sa.UniqueConstraint(col)`` *and*
``op.create_index(..., unique=True)`` for ``resources.name``,
``users.username`` and ``api_keys.hashed_key``.  The ORM models
(``unique=True, index=True``) only produce the unique index, so on
PostgreSQL every such column carried two unique B-tree indexes and
``alembic.autogenerate.compare_metadata`` reported drift.  Uniqueness is
still enforced by the ``ix_*`` unique indexes.

SQLite is left untouched: its unnamed table-level constraints cannot be
dropped by name, are not reflected as drift, and are harmless there.

Revision ID: 0008_drop_redundant_uniques
Revises: 0007_v2g_protocols
Create Date: 2026-09-28 18:00:00
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from alembic import op

if TYPE_CHECKING:
    from collections.abc import Sequence

revision: str = "0008_drop_redundant_uniques"
down_revision: str | Sequence[str] | None = "0007_v2g_protocols"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# (table, column, PostgreSQL's default name for the baseline's unnamed constraint)
_REDUNDANT_UNIQUES = [
    ("resources", "name", "resources_name_key"),
    ("users", "username", "users_username_key"),
    ("api_keys", "hashed_key", "api_keys_hashed_key_key"),
]


def upgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return
    for table, _column, constraint in _REDUNDANT_UNIQUES:
        # IF EXISTS: databases created by ``create_all`` never had them.
        op.execute(f'ALTER TABLE {table} DROP CONSTRAINT IF EXISTS "{constraint}"')


def downgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return
    for table, column, constraint in _REDUNDANT_UNIQUES:
        op.create_unique_constraint(constraint, table, [column])
