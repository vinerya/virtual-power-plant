"""Trading routes — orders, trades, portfolio, markets, strategies.

Orders execute on the process-wide :class:`~vpp.trading.service.TradingService`,
which runs a *simulated* continuous-matching venue (no orders leave the
process). Prices are $/MWh and quantities MWh. Placing/cancelling orders,
advancing the simulation and live strategy runs require the ``operator`` or
``admin`` role; everything else is readable by any authenticated user.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: TC002 -- FastAPI resolves at runtime

from vpp.auth.security import get_current_user, require_role
from vpp.db.engine import get_db
from vpp.db.models import OrderModel, TradeModel, UserModel  # noqa: TC001 -- FastAPI runtime
from vpp.db.repositories import TradingRepository
from vpp.schemas.auth import UserRole
from vpp.schemas.trading import (
    BacktestRequest,
    BacktestResponse,
    MarketResponse,
    OrderCreate,
    OrderResponse,
    OrderSubmitResponse,
    PortfolioResponse,
    StrategyInfo,
    StrategyRunRequest,
    StrategyRunResponse,
    TickResponse,
    TradeResponse,
)
from vpp.trading.service import TradingError, get_trading_service

router = APIRouter(prefix="/api/v1/trading", tags=["Trading"])

_trader = require_role(UserRole.ADMIN, UserRole.OPERATOR)

_BACKTEST_ASSUMPTIONS = [
    "Market-order signals fill in full at last price +/- half_spread.",
    "Limit-order signals fill in full at the limit when the bar's last price "
    "is at or through it; otherwise they are dropped (no resting between bars).",
    "fee_per_unit is charged on every filled unit; no market impact or liquidity cap.",
    "Sharpe is annualised from per-bar equity returns (365-day year); "
    "drawdown is peak-to-trough of marked-to-market equity.",
]


def _raise(exc: TradingError) -> None:
    raise HTTPException(status_code=exc.status_code, detail=exc.to_detail()) from exc


def _metadata(row: OrderModel) -> dict[str, Any]:
    try:
        data = json.loads(row.metadata_json or "{}")
    except (TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _order_response(row: OrderModel) -> OrderResponse:
    return OrderResponse(
        id=row.id, order_type=row.order_type, market=row.market, side=row.side,
        quantity=row.quantity, price=row.price, status=row.status,
        filled_quantity=row.filled_quantity, remaining_quantity=row.remaining_quantity,
        average_price=row.average_price, time_in_force=row.time_in_force,
        created_at=row.created_at or datetime.utcnow(), updated_at=row.updated_at,
        metadata=_metadata(row),
    )


def _trade_response(row: TradeModel) -> TradeResponse:
    return TradeResponse(
        id=row.id, order_id=row.order_id, market=row.market, side=row.side,
        quantity=row.quantity, price=row.price, fees=row.fees,
        timestamp=row.created_at or datetime.utcnow(), strategy=row.strategy or None,
        realized_pnl=row.realized_pnl,
    )


def _username(user: UserModel) -> str | None:
    return getattr(user, "username", None)


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------


@router.post("/orders", response_model=OrderSubmitResponse, status_code=status.HTTP_201_CREATED)
async def submit_order(
    body: OrderCreate,
    session: AsyncSession = Depends(get_db),
    user: UserModel = Depends(_trader),
):
    """Submit an order: validate, pre-trade risk check, execute, persist.

    Returns 201 with the resulting order (status ``filled``, ``partial``,
    ``pending`` if resting, or ``cancelled`` for an unfilled IOC/FOK/market
    remainder) and any immediate fills. Returns 422 with
    ``detail.code`` = ``order_invalid`` / ``unknown_market`` for bad orders,
    or ``risk_limit_breached`` (with ``detail.reasons`` and the persisted
    ``detail.order_id``) when a risk limit would be breached.
    """
    svc = get_trading_service()
    try:
        result = await svc.place_order(
            session,
            order_type=body.order_type, market=body.market, side=body.side,
            quantity=body.quantity, price=body.price, stop_price=body.stop_price,
            limit_price=body.limit_price, visible_quantity=body.visible_quantity,
            time_in_force=body.time_in_force, metadata=body.metadata,
            submitted_by=_username(user),
        )
    except TradingError as exc:
        _raise(exc)
    base = _order_response(result.order)
    return OrderSubmitResponse(
        **base.model_dump(), fills=[_trade_response(t) for t in result.trades]
    )


@router.get("/orders", response_model=list[OrderResponse])
async def list_orders(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    market: str | None = None,
    order_status: str | None = None,
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(get_current_user),
):
    """List trading orders (newest first) with optional filters."""
    orders = await TradingRepository.list_orders(
        session, skip=skip, limit=limit, market=market, status=order_status
    )
    return [_order_response(o) for o in orders]


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
    return _order_response(order)


async def _cancel(order_id: str, session: AsyncSession, user: UserModel) -> OrderResponse:
    try:
        row = await get_trading_service().cancel_order(
            session, order_id, cancelled_by=_username(user)
        )
    except TradingError as exc:
        _raise(exc)
    return _order_response(row)


@router.delete("/orders/{order_id}", response_model=OrderResponse)
async def cancel_order(
    order_id: str,
    session: AsyncSession = Depends(get_db),
    user: UserModel = Depends(_trader),
):
    """Cancel a resting (``pending``/``partial``) order. 409 if already final."""
    return await _cancel(order_id, session, user)


@router.post("/orders/{order_id}/cancel", response_model=OrderResponse)
async def cancel_order_post(
    order_id: str,
    session: AsyncSession = Depends(get_db),
    user: UserModel = Depends(_trader),
):
    """Alias of ``DELETE /orders/{order_id}`` for clients that cannot send DELETE."""
    return await _cancel(order_id, session, user)


# ---------------------------------------------------------------------------
# Trades / portfolio / markets
# ---------------------------------------------------------------------------


@router.get("/trades", response_model=list[TradeResponse])
async def list_trades(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    market: str | None = None,
    order_id: str | None = None,
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(get_current_user),
):
    """List persisted fills (newest first)."""
    trades = await TradingRepository.list_trades(
        session, skip=skip, limit=limit, market=market, order_id=order_id
    )
    return [_trade_response(t) for t in trades]


@router.get("/portfolio", response_model=PortfolioResponse)
async def get_portfolio(
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(get_current_user),
):
    """Positions, realized/unrealized P&L, exposure and risk metrics."""
    svc = get_trading_service()
    await svc.ensure_ready(session)
    return svc.portfolio_snapshot()


@router.get("/markets", response_model=list[MarketResponse])
async def list_markets(
    depth: int = Query(0, ge=0, le=20, description="Order-book levels to include"),
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(get_current_user),
):
    """Tradable (simulated) markets with their latest prices."""
    svc = get_trading_service()
    await svc.ensure_ready(session)
    return svc.markets(depth_levels=depth)


@router.post("/markets/tick", response_model=TickResponse)
async def advance_markets(
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(_trader),
):
    """Advance the simulated venue one step (also done periodically in the
    background): new prices, resting-order matching, ``market_data`` events."""
    return await get_trading_service().tick(session)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@router.get("/strategies", response_model=list[StrategyInfo])
async def list_strategies(_user: UserModel = Depends(get_current_user)):
    """Available strategies and their default parameters."""
    return get_trading_service().list_strategies()


@router.post("/strategies/{name}/backtest", response_model=BacktestResponse)
async def backtest_strategy(
    name: str,
    body: BacktestRequest,
    _user: UserModel = Depends(get_current_user),
):
    """Backtest a strategy over supplied prices or seeded synthetic prices."""
    svc = get_trading_service()
    try:
        result = svc.backtest(
            name,
            params=body.params,
            prices=body.prices,
            timestamps=body.timestamps,
            interval_minutes=(
                body.interval_minutes if body.prices is not None
                else body.synthetic.interval_minutes
            ),
            periods=body.synthetic.periods,
            seed=body.synthetic.seed,
            initial_cash=body.initial_cash,
            fee_per_unit=body.fee_per_unit,
            half_spread=body.half_spread,
        )
    except TradingError as exc:
        _raise(exc)
    payload = dict(result.__dict__)
    payload["data_source"] = "provided" if body.prices is not None else "synthetic"
    payload["assumptions"] = list(_BACKTEST_ASSUMPTIONS)
    return payload


@router.post("/strategies/{name}/run", response_model=StrategyRunResponse)
async def run_strategy(
    name: str,
    body: StrategyRunRequest,
    session: AsyncSession = Depends(get_db),
    user: UserModel = Depends(_trader),
):
    """Run a strategy once against current market data.

    With ``dry_run`` (the default) only the signals are returned; otherwise
    each signal is submitted as an order through the same validation and
    pre-trade risk checks as ``POST /orders``.
    """
    try:
        return await get_trading_service().run_strategy(
            session, name, params=body.params, dry_run=body.dry_run,
            submitted_by=_username(user),
        )
    except TradingError as exc:
        _raise(exc)
