"""latest_soc: newest state of charge per resource, found with index seeks.

Regression for a hot spot found with ``vpp loadtest`` (docs/performance.md):
the lookup behind ``GET /api/v1/resources`` used ``GROUP BY max(timestamp)``
over every telemetry sample of the listed resources, so the endpoint slowed
down linearly as telemetry accumulated.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import event, text

from vpp.db.models import BatteryStateModel, ResourceModel, ResourceTelemetryModel
from vpp.portal.telemetry import latest_soc

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


async def _battery(session) -> str:
    rid = str(uuid.uuid4())
    session.add(ResourceModel(id=rid, name=f"soc-{rid}", resource_type="battery", rated_power=5.0))
    await session.flush()
    return rid


@pytest.mark.asyncio
async def test_newest_sample_wins_across_both_tables(db_session):
    # a: MQTT sample newest; b: telemetry SOC newest; c: power-only; d: nothing.
    a, b, c, d = [await _battery(db_session) for _ in range(4)]
    db_session.add_all(
        [
            BatteryStateModel(resource_id=a, soc=80.0, timestamp=T0 + timedelta(minutes=2)),
            ResourceTelemetryModel(
                resource_id=a, timestamp=T0, power_kw=1.0, state_of_charge=0.1, source="t"
            ),
            BatteryStateModel(resource_id=b, soc=0.3, timestamp=T0),
            ResourceTelemetryModel(
                resource_id=b,
                timestamp=T0 + timedelta(minutes=1),
                power_kw=1.0,
                state_of_charge=0.6,
                source="t",
            ),
            # A newer power-only sample does not hide the older SOC.
            ResourceTelemetryModel(
                resource_id=b,
                timestamp=T0 + timedelta(minutes=5),
                power_kw=2.0,
                state_of_charge=None,
                source="t",
            ),
            ResourceTelemetryModel(
                resource_id=c, timestamp=T0, power_kw=1.0, state_of_charge=None, source="t"
            ),
        ]
    )
    await db_session.flush()
    socs = await latest_soc(db_session, [a, b, c, d, "no-such-resource"])
    assert socs == {a: pytest.approx(0.8), b: pytest.approx(0.6)}


@pytest.mark.asyncio
async def test_lookup_seeks_indexes_instead_of_aggregating(db_session):
    rid = await _battery(db_session)
    statements: list[str] = []
    engine = db_session.get_bind()

    def capture(conn, cursor, statement, params, context, executemany):
        statements.append(statement)

    event.listen(engine, "before_cursor_execute", capture)
    try:
        await latest_soc(db_session, [rid])
    finally:
        event.remove(engine, "before_cursor_execute", capture)
    [sql] = statements
    assert "GROUP BY" not in sql.upper()
    assert sql.upper().count("LIMIT") == 4

    if engine.dialect.name == "sqlite":
        # Without ANALYZE statistics SQLite must still pick the covering
        # partial index for the SOC lookup (not walk power-only rows).
        plan = (
            await db_session.execute(
                text(
                    "EXPLAIN QUERY PLAN SELECT timestamp, state_of_charge "
                    "FROM resource_telemetry WHERE resource_id = :r "
                    "AND state_of_charge IS NOT NULL ORDER BY timestamp DESC LIMIT 1"
                ),
                {"r": rid},
            )
        ).all()
        detail = " ".join(str(row[-1]) for row in plan)
        assert "COVERING INDEX ix_resource_telemetry_resource_soc_ts" in detail
