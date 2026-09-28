"""user management: session versioning, last login, API-key metadata

* ``users.token_version`` -- embedded in JWTs as ``ver`` and bumped on
  password change / deactivation / role change / "log out everywhere",
  which revokes every previously issued token.
* ``users.last_login_at``
* ``api_keys.key_prefix`` -- first characters of the raw key for display
  (NULL for keys created before this revision).
* ``api_keys.last_used_at``

Revision ID: 0009_user_management
Revises: 0008_drop_redundant_uniques
Create Date: 2026-09-28 20:00:00
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa
from alembic import op

if TYPE_CHECKING:
    from collections.abc import Sequence

revision: str = "0009_user_management"
down_revision: str | Sequence[str] | None = "0008_drop_redundant_uniques"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("users") as batch:
        batch.add_column(
            sa.Column("token_version", sa.Integer(), nullable=False, server_default="0")
        )
        batch.add_column(sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True))
    with op.batch_alter_table("api_keys") as batch:
        batch.add_column(sa.Column("key_prefix", sa.String(16), nullable=True))
        batch.add_column(sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("api_keys") as batch:
        batch.drop_column("last_used_at")
        batch.drop_column("key_prefix")
    with op.batch_alter_table("users") as batch:
        batch.drop_column("last_login_at")
        batch.drop_column("token_version")
