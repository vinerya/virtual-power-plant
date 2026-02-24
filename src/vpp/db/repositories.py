"""Repository pattern — thin async CRUD wrappers around SQLAlchemy queries."""

from __future__ import annotations

import json
from typing import Any, Optional

from sqlalchemy import select, func, delete
from sqlalchemy.ext.asyncio import AsyncSession

from .models import (
    ResourceModel,
    BatteryStateModel,
    OptimizationRunModel,
    OrderModel,
    TradeModel,
    UserModel,
    APIKeyModel,
    EventLogModel,
)


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------

class ResourceRepository:
    """CRUD for energy resources."""

    @staticmethod
    async def create(session: AsyncSession, *, name: str, resource_type: str,
                     rated_power: float, config: dict | None = None,
                     metadata: dict | None = None) -> ResourceModel:
        obj = ResourceModel(
            name=name,
            resource_type=resource_type,
            rated_power=rated_power,
            config_json=json.dumps(config or {}),
            metadata_json=json.dumps(metadata or {}),
        )
        session.add(obj)
        await session.flush()
        return obj

    @staticmethod
    async def get_by_id(session: AsyncSession, resource_id: str) -> Optional[ResourceModel]:
        return await session.get(ResourceModel, resource_id)

    @staticmethod
    async def get_by_name(session: AsyncSession, name: str) -> Optional[ResourceModel]:
        result = await session.execute(
            select(ResourceModel).where(ResourceModel.name == name)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def list_all(session: AsyncSession, *, skip: int = 0,
                       limit: int = 100, resource_type: str | None = None) -> list[ResourceModel]:
        stmt = select(ResourceModel).offset(skip).limit(limit).order_by(ResourceModel.created_at.desc())
        if resource_type:
            stmt = stmt.where(ResourceModel.resource_type == resource_type)
        result = await session.execute(stmt)
        return list(result.scalars().all())

    @staticmethod
    async def update(session: AsyncSession, resource_id: str,
                     **fields: Any) -> Optional[ResourceModel]:
        obj = await session.get(ResourceModel, resource_id)
        if obj is None:
            return None
        for key, value in fields.items():
            if hasattr(obj, key) and value is not None:
                setattr(obj, key, value)
        await session.flush()
        return obj

    @staticmethod
    async def delete(session: AsyncSession, resource_id: str) -> bool:
        obj = await session.get(ResourceModel, resource_id)
        if obj is None:
            return False
        await session.delete(obj)
        await session.flush()
        return True

    @staticmethod
    async def count(session: AsyncSession) -> int:
        result = await session.execute(select(func.count(ResourceModel.id)))
        return result.scalar_one()


# ---------------------------------------------------------------------------
# Battery State
# ---------------------------------------------------------------------------

class BatteryStateRepository:
    """Time-series battery state snapshots."""

    @staticmethod
    async def record(session: AsyncSession, resource_id: str, *,
                     soc: float, soh: float = 100.0, temperature: float = 25.0,
                     voltage: float = 0.0, current: float = 0.0,
                     power: float = 0.0) -> BatteryStateModel:
        obj = BatteryStateModel(
            resource_id=resource_id, soc=soc, soh=soh,
            temperature=temperature, voltage=voltage,
            current=current, power=power,
        )
        session.add(obj)
        await session.flush()
        return obj

    @staticmethod
    async def get_latest(session: AsyncSession, resource_id: str,
                         limit: int = 100) -> list[BatteryStateModel]:
        result = await session.execute(
            select(BatteryStateModel)
            .where(BatteryStateModel.resource_id == resource_id)
            .order_by(BatteryStateModel.timestamp.desc())
            .limit(limit)
        )
        return list(result.scalars().all())


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------

class OptimizationRepository:
    """CRUD for optimization run records."""

    @staticmethod
    async def record_run(session: AsyncSession, *, problem_type: str, status: str,
                         objective_value: float = 0.0, solve_time_ms: float = 0.0,
                         solver: str = "", fallback_used: bool = False,
                         solution: dict | None = None,
                         parameters: dict | None = None) -> OptimizationRunModel:
        obj = OptimizationRunModel(
            problem_type=problem_type,
            status=status,
            objective_value=objective_value,
            solve_time_ms=solve_time_ms,
            solver=solver,
            fallback_used=fallback_used,
            solution_json=json.dumps(solution or {}),
            parameters_json=json.dumps(parameters or {}),
        )
        session.add(obj)
        await session.flush()
        return obj

    @staticmethod
    async def list_runs(session: AsyncSession, *, skip: int = 0,
                        limit: int = 50, problem_type: str | None = None) -> list[OptimizationRunModel]:
        stmt = (
            select(OptimizationRunModel)
            .offset(skip).limit(limit)
            .order_by(OptimizationRunModel.created_at.desc())
        )
        if problem_type:
            stmt = stmt.where(OptimizationRunModel.problem_type == problem_type)
        result = await session.execute(stmt)
        return list(result.scalars().all())

    @staticmethod
    async def get_stats(session: AsyncSession) -> dict[str, Any]:
        total = await session.execute(select(func.count(OptimizationRunModel.id)))
        avg_time = await session.execute(select(func.avg(OptimizationRunModel.solve_time_ms)))
        fallbacks = await session.execute(
            select(func.count(OptimizationRunModel.id))
            .where(OptimizationRunModel.fallback_used.is_(True))
        )
        return {
            "total_runs": total.scalar_one(),
            "avg_solve_time_ms": avg_time.scalar_one() or 0.0,
            "fallback_count": fallbacks.scalar_one(),
        }


# ---------------------------------------------------------------------------
# Trading
# ---------------------------------------------------------------------------

class TradingRepository:
    """CRUD for orders and trades."""

    # --- Orders ---

    @staticmethod
    async def create_order(session: AsyncSession, **fields: Any) -> OrderModel:
        metadata = fields.pop("metadata", {})
        obj = OrderModel(**fields, metadata_json=json.dumps(metadata))
        session.add(obj)
        await session.flush()
        return obj

    @staticmethod
    async def get_order(session: AsyncSession, order_id: str) -> Optional[OrderModel]:
        return await session.get(OrderModel, order_id)

    @staticmethod
    async def list_orders(session: AsyncSession, *, skip: int = 0,
                          limit: int = 50, market: str | None = None,
                          status: str | None = None) -> list[OrderModel]:
        stmt = select(OrderModel).offset(skip).limit(limit).order_by(OrderModel.created_at.desc())
        if market:
            stmt = stmt.where(OrderModel.market == market)
        if status:
            stmt = stmt.where(OrderModel.status == status)
        result = await session.execute(stmt)
        return list(result.scalars().all())

    @staticmethod
    async def update_order_status(session: AsyncSession, order_id: str,
                                  status: str, **extra: Any) -> Optional[OrderModel]:
        obj = await session.get(OrderModel, order_id)
        if obj is None:
            return None
        obj.status = status
        for k, v in extra.items():
            if hasattr(obj, k):
                setattr(obj, k, v)
        await session.flush()
        return obj

    # --- Trades ---

    @staticmethod
    async def record_trade(session: AsyncSession, **fields: Any) -> TradeModel:
        obj = TradeModel(**fields)
        session.add(obj)
        await session.flush()
        return obj

    @staticmethod
    async def list_trades(session: AsyncSession, *, skip: int = 0,
                          limit: int = 50, market: str | None = None) -> list[TradeModel]:
        stmt = select(TradeModel).offset(skip).limit(limit).order_by(TradeModel.created_at.desc())
        if market:
            stmt = stmt.where(TradeModel.market == market)
        result = await session.execute(stmt)
        return list(result.scalars().all())


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

class UserRepository:
    """CRUD for users and API keys."""

    @staticmethod
    async def create_user(session: AsyncSession, *, username: str,
                          hashed_password: str, role: str = "viewer") -> UserModel:
        obj = UserModel(username=username, hashed_password=hashed_password, role=role)
        session.add(obj)
        await session.flush()
        return obj

    @staticmethod
    async def get_by_username(session: AsyncSession, username: str) -> Optional[UserModel]:
        result = await session.execute(
            select(UserModel).where(UserModel.username == username)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_by_id(session: AsyncSession, user_id: str) -> Optional[UserModel]:
        return await session.get(UserModel, user_id)

    @staticmethod
    async def create_api_key(session: AsyncSession, *, user_id: str,
                             name: str, hashed_key: str,
                             role: str = "viewer") -> APIKeyModel:
        obj = APIKeyModel(user_id=user_id, name=name, hashed_key=hashed_key, role=role)
        session.add(obj)
        await session.flush()
        return obj

    @staticmethod
    async def get_api_key_by_hash(session: AsyncSession, hashed_key: str) -> Optional[APIKeyModel]:
        result = await session.execute(
            select(APIKeyModel).where(APIKeyModel.hashed_key == hashed_key)
        )
        return result.scalar_one_or_none()


# ---------------------------------------------------------------------------
# Event log
# ---------------------------------------------------------------------------

class EventLogRepository:
    """Append-only event log."""

    @staticmethod
    async def log(session: AsyncSession, *, event_type: str,
                  details: dict | None = None, resource_id: str | None = None,
                  severity: str = "info") -> EventLogModel:
        obj = EventLogModel(
            event_type=event_type,
            details_json=json.dumps(details or {}),
            resource_id=resource_id,
            severity=severity,
        )
        session.add(obj)
        await session.flush()
        return obj

    @staticmethod
    async def query(session: AsyncSession, *, event_type: str | None = None,
                    resource_id: str | None = None, limit: int = 100) -> list[EventLogModel]:
        stmt = select(EventLogModel).limit(limit).order_by(EventLogModel.created_at.desc())
        if event_type:
            stmt = stmt.where(EventLogModel.event_type == event_type)
        if resource_id:
            stmt = stmt.where(EventLogModel.resource_id == resource_id)
        result = await session.execute(stmt)
        return list(result.scalars().all())
