"""Trading routes — orders, trades, portfolio."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from vpp.auth.security import get_current_user
from vpp.db.engine import get_db
from vpp.db.models import UserModel
from vpp.db.repositories import TradingRepository
from vpp.schemas.trading import (
    OrderCreate,
    OrderResponse,
    PortfolioResponse,
    PositionResponse,
    TradeResponse,
)

router = APIRouter(prefix="/api/v1/trading", tags=["Trading"])


@router.post("/orders", response_model=OrderResponse, status_code=status.HTTP_201_CREATED)
async def submit_order(
    body: OrderCreate,
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(get_current_user),
):
    """Submit a new trading order."""
    order = await TradingRepository.create_order(
        session,
        order_type=body.order_type,
        market=body.market,
        side=body.side,
        quantity=body.quantity,
        price=body.price or 0.0,
        remaining_quantity=body.quantity,
        time_in_force=body.time_in_force,
        metadata=body.metadata,
    )
    return OrderResponse(
        id=order.id,
        order_type=order.order_type,
        market=order.market,
        side=order.side,
        quantity=order.quantity,
        price=order.price,
        status=order.status,
        filled_quantity=order.filled_quantity,
        remaining_quantity=order.remaining_quantity,
        average_price=order.average_price,
        time_in_force=order.time_in_force,
        created_at=order.created_at,
    )


@router.get("/orders", response_model=list[OrderResponse])
async def list_orders(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    market: Optional[str] = None,
    order_status: Optional[str] = None,
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(get_current_user),
):
    """List trading orders with optional filters."""
    orders = await TradingRepository.list_orders(
        session, skip=skip, limit=limit, market=market, status=order_status
    )
    return [
        OrderResponse(
            id=o.id, order_type=o.order_type, market=o.market, side=o.side,
            quantity=o.quantity, price=o.price, status=o.status,
            filled_quantity=o.filled_quantity, remaining_quantity=o.remaining_quantity,
            average_price=o.average_price, time_in_force=o.time_in_force,
            created_at=o.created_at,
        )
        for o in orders
    ]


@router.get("/orders/{order_id}", response_model=OrderResponse)
async def get_order(
    order_id: str,
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(get_current_user),
):
    """Get a single order by ID."""
    order = await TradingRepository.get_order(session, order_id)
    if order is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Order not found")
    return OrderResponse(
        id=order.id, order_type=order.order_type, market=order.market,
        side=order.side, quantity=order.quantity, price=order.price,
        status=order.status, filled_quantity=order.filled_quantity,
        remaining_quantity=order.remaining_quantity, average_price=order.average_price,
        time_in_force=order.time_in_force, created_at=order.created_at,
    )


@router.delete("/orders/{order_id}", status_code=status.HTTP_200_OK)
async def cancel_order(
    order_id: str,
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(get_current_user),
):
    """Cancel a pending order."""
    order = await TradingRepository.update_order_status(session, order_id, "cancelled")
    if order is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Order not found")
    return {"id": order.id, "status": "cancelled"}


@router.get("/trades", response_model=list[TradeResponse])
async def list_trades(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    market: Optional[str] = None,
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(get_current_user),
):
    """List executed trades."""
    trades = await TradingRepository.list_trades(session, skip=skip, limit=limit, market=market)
    return [
        TradeResponse(
            id=t.id, order_id=t.order_id, market=t.market, side=t.side,
            quantity=t.quantity, price=t.price, fees=t.fees,
            timestamp=t.created_at, strategy=t.strategy,
            realized_pnl=t.realized_pnl,
        )
        for t in trades
    ]


@router.get("/portfolio", response_model=PortfolioResponse)
async def get_portfolio(
    _user: UserModel = Depends(get_current_user),
):
    """Get current portfolio snapshot (in-memory from trading engine)."""
    return PortfolioResponse(
        cash=0.0,
        equity=0.0,
        total_pnl=0.0,
        max_drawdown=0.0,
        positions=[],
        total_trades=0,
        last_updated=datetime.utcnow(),
    )
