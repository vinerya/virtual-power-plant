"""add nominal_energy_kwh column on resources (M4)

M3 used ``rated_power`` (kW) as a proxy for nominal energy capacity (kWh)
when computing throughput.  M4 introduces an explicit ``nominal_energy_kwh``
column to fix the unit mismatch.  The column is nullable so non-battery
resources (and pre-M4 fixtures) carry NULL; the ``DegradationUpdater`` falls
back to a documented C/4 heuristic (rated_power * 4.0) when missing.

Revision ID: 0003_add_nominal_energy
Revises: 0002_add_battery_degradation
Create Date: 2025-12-01 00:00:00
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0003_add_nominal_energy"
down_revision: Union[str, Sequence[str], None] = "0002_add_battery_degradation"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("resources") as batch:
        batch.add_column(sa.Column("nominal_energy_kwh", sa.Float(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("resources") as batch:
        batch.drop_column("nominal_energy_kwh")
