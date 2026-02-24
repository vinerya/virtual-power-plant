"""Tests for the database layer — repositories and models."""

import pytest

from vpp.db.repositories import (
    ResourceRepository,
    OptimizationRepository,
    TradingRepository,
    UserRepository,
    EventLogRepository,
)
from vpp.auth.security import get_password_hash


@pytest.mark.asyncio
async def test_resource_crud(db_session):
    # Create
    res = await ResourceRepository.create(
        db_session, name="db-test-bat", resource_type="battery", rated_power=25.0
    )
    assert res.id is not None
    assert res.name == "db-test-bat"

    # Read
    fetched = await ResourceRepository.get_by_id(db_session, res.id)
    assert fetched is not None
    assert fetched.name == "db-test-bat"

    # Update
    updated = await ResourceRepository.update(db_session, res.id, rated_power=30.0)
    assert updated.rated_power == 30.0

    # Delete
    deleted = await ResourceRepository.delete(db_session, res.id)
    assert deleted is True
    assert await ResourceRepository.get_by_id(db_session, res.id) is None


@pytest.mark.asyncio
async def test_optimization_record(db_session):
    run = await OptimizationRepository.record_run(
        db_session,
        problem_type="stochastic",
        status="success",
        objective_value=42.5,
        solve_time_ms=12.3,
    )
    assert run.id is not None

    runs = await OptimizationRepository.list_runs(db_session, limit=5)
    assert len(runs) >= 1

    stats = await OptimizationRepository.get_stats(db_session)
    assert stats["total_runs"] >= 1


@pytest.mark.asyncio
async def test_trading_order_crud(db_session):
    order = await TradingRepository.create_order(
        db_session,
        order_type="limit",
        market="day_ahead",
        side="buy",
        quantity=100.0,
        price=0.12,
        remaining_quantity=100.0,
    )
    assert order.id is not None
    assert order.status == "pending"

    fetched = await TradingRepository.get_order(db_session, order.id)
    assert fetched is not None

    updated = await TradingRepository.update_order_status(db_session, order.id, "filled")
    assert updated.status == "filled"


@pytest.mark.asyncio
async def test_event_log(db_session):
    evt = await EventLogRepository.log(
        db_session,
        event_type="resource_added",
        details={"name": "test"},
        severity="info",
    )
    assert evt.id is not None

    events = await EventLogRepository.query(db_session, event_type="resource_added")
    assert len(events) >= 1
