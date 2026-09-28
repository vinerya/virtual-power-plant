"""Trading routes — orders, trades, portfolio, markets, strategies.

Orders execute on the process-wide :class:`~vpp.trading.service.TradingService`,
which runs a *simulated* continuous-matching venue (no orders leave the
process). Prices are $/MWh and quantities MWh. Placing/cancelling orders,
advancing the simulation and live strategy runs require the ``operator`` or
``admin`` role; everything else is readable by any authenticated user.

With several API workers the venue runs only in the process holding the
``trading-venue`` lease (:mod:`vpp.cluster`). Other workers forward venue
operations (submit/cancel, portfolio, markets, tick, strategy runs) to it
through the database and answer with its result; if no leader answers in
``VPP_CLUSTER_CALL_TIMEOUT_SECONDS`` they return 503 ``leader_unavailable``
(the order was not executed) or 504 ``leader_timeout`` (claimed, outcome
unknown -- check ``GET /orders``).
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, NoReturn

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from vpp import audit
from vpp.api.pagination import Page, page_params, paginate
from vpp.auth.security import get_current_user, require_role
from vpp.cluster.rpc import on_leader, register_handler
from vpp.cluster.topology import LEASE_TRADING
from vpp.db.engine import get_db
from vpp.db.models import OrderModel, TradeModel, UserModel
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


def _raise(exc: TradingError) -> NoReturn:
    raise HTTPException(status_code=exc.status_code, detail=exc.to_detail()) from exc


def _metadata(row: OrderModel) -> dict[str, Any]:
    try:
        data = json.loads(row.metadata_json or "{}")
    except (TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _order_response(row: OrderModel) -> OrderResponse:
    return OrderResponse(
        id=row.id,
        order_type=row.order_type,
        market=row.market,
        side=row.side,
        quantity=row.quantity,
        price=row.price,
        status=row.status,
        filled_quantity=row.filled_quantity,
        remaining_quantity=row.remaining_quantity,
        average_price=row.average_price,
        time_in_force=row.time_in_force,
        created_at=row.created_at or datetime.utcnow(),
        updated_at=row.updated_at,
        metadata=_metadata(row),
    )


def _trade_response(row: TradeModel) -> TradeResponse:
    return TradeResponse(
        id=row.id,
        order_id=row.order_id,
        market=row.market,
        side=row.side,
        quantity=row.quantity,
        price=row.price,
        fees=row.fees,
        timestamp=row.created_at or datetime.utcnow(),
        strategy=row.strategy or None,
        realized_pnl=row.realized_pnl,
    )


def _username(user: UserModel) -> str | None:
    return getattr(user, "username", None)


_list_page = page_params(default_limit=50, max_limit=200, legacy_skip=True)


def _field(result: Any, name: str) -> Any:
    """``name`` of a local (model) or forwarded (dict) venue result."""
    if isinstance(result, dict):
        return result.get(name)
    return getattr(result, name, None)


def _audit_trading_error(
    session: AsyncSession,
    request: Request,
    user: UserModel,
    action: str,
    exc: HTTPException,
    details: dict[str, Any],
) -> None:
    detail = exc.detail if isinstance(exc.detail, dict) else {"message": str(exc.detail)}
    audit.record(
        session,
        request,
        action,
        actor=user,
        target_type="order",
        target_id=detail.get("order_id"),
        outcome="denied" if detail.get("code") == "risk_limit_breached" else "failure",
        details={**details, "status_code": exc.status_code, "code": detail.get("code")},
        always=True,
    )


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------


@router.post("/orders", response_model=OrderSubmitResponse, status_code=status.HTTP_201_CREATED)
async def submit_order(
    body: OrderCreate,
    request: Request,
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
    username = _username(user)
    order_details = {
        "market": body.market,
        "side": body.side,
        "order_type": body.order_type,
        "quantity": body.quantity,
        "price": body.price,
    }
    try:
        result = await on_leader(
            LEASE_TRADING,
            "submit_order",
            {"body": body.model_dump(mode="json"), "username": username},
            lambda: _submit_order_local(session, body, username),
        )
    except HTTPException as exc:
        _audit_trading_error(session, request, user, "market.order_submit", exc, order_details)
        raise
    audit.record(
        session,
        request,
        "market.order_submit",
        actor=user,
        target_type="order",
        target_id=_field(result, "id"),
        details={**order_details, "status": _field(result, "status")},
    )
    return result


async def _submit_order_local(
    session: AsyncSession, body: OrderCreate, username: str | None
) -> OrderSubmitResponse:
    svc = get_trading_service()
    try:
        result = await svc.place_order(
            session,
            order_type=body.order_type,
            market=body.market,
            side=body.side,
            quantity=body.quantity,
            price=body.price,
            stop_price=body.stop_price,
            limit_price=body.limit_price,
            visible_quantity=body.visible_quantity,
            time_in_force=body.time_in_force,
            metadata=body.metadata,
            submitted_by=username,
        )
    except TradingError as exc:
        _raise(exc)
    base = _order_response(result.order)
    return OrderSubmitResponse(
        **base.model_dump(), fills=[_trade_response(t) for t in result.trades]
    )


@router.get("/orders", response_model=list[OrderResponse])
async def list_orders(
    response: Response,
    page: Page = Depends(_list_page),
    market: str | None = None,
    order_status: str | None = None,
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(get_current_user),
):
    """List trading orders (newest first) with optional filters.

    Paginated (``limit`` / ``offset``, ``skip`` is an alias); the total is in
    ``X-Total-Count``.
    """
    stmt = TradingRepository.orders_query(market=market, status=order_status)
    orders = await paginate(session, response, stmt, page)
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


async def _cancel(order_id: str, session: AsyncSession, user: UserModel, request: Request) -> Any:
    username = _username(user)
    try:
        result = await on_leader(
            LEASE_TRADING,
            "cancel_order",
            {"order_id": order_id, "username": username},
            lambda: _cancel_local(session, order_id, username),
        )
    except HTTPException as exc:
        _audit_trading_error(
            session, request, user, "market.order_cancel", exc, {"order_id": order_id}
        )
        raise
    audit.record(
        session,
        request,
        "market.order_cancel",
        actor=user,
        target_type="order",
        target_id=order_id,
        details={"status": _field(result, "status")},
    )
    return result


async def _cancel_local(
    session: AsyncSession, order_id: str, username: str | None
) -> OrderResponse:
    try:
        row = await get_trading_service().cancel_order(session, order_id, cancelled_by=username)
    except TradingError as exc:
        _raise(exc)
    return _order_response(row)


@router.delete("/orders/{order_id}", response_model=OrderResponse)
async def cancel_order(
    order_id: str,
    request: Request,
    session: AsyncSession = Depends(get_db),
    user: UserModel = Depends(_trader),
):
    """Cancel a resting (``pending``/``partial``) order. 409 if already final."""
    return await _cancel(order_id, session, user, request)


@router.post("/orders/{order_id}/cancel", response_model=OrderResponse)
async def cancel_order_post(
    order_id: str,
    request: Request,
    session: AsyncSession = Depends(get_db),
    user: UserModel = Depends(_trader),
):
    """Alias of ``DELETE /orders/{order_id}`` for clients that cannot send DELETE."""
    return await _cancel(order_id, session, user, request)


# ---------------------------------------------------------------------------
# Trades / portfolio / markets
# ---------------------------------------------------------------------------


@router.get("/trades", response_model=list[TradeResponse])
async def list_trades(
    response: Response,
    page: Page = Depends(_list_page),
    market: str | None = None,
    order_id: str | None = None,
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(get_current_user),
):
    """List persisted fills (newest first).

    Paginated (``limit`` / ``offset``, ``skip`` is an alias); the total is in
    ``X-Total-Count``.
    """
    stmt = TradingRepository.trades_query(market=market, order_id=order_id)
    trades = await paginate(session, response, stmt, page)
    return [_trade_response(t) for t in trades]


@router.get("/portfolio", response_model=PortfolioResponse)
async def get_portfolio(
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(get_current_user),
):
    """Positions, realized/unrealized P&L, exposure and risk metrics."""
    return await on_leader(LEASE_TRADING, "portfolio", {}, lambda: _portfolio_local(session))


async def _portfolio_local(session: AsyncSession) -> dict[str, Any]:
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
    return await on_leader(
        LEASE_TRADING, "markets", {"depth": depth}, lambda: _markets_local(session, depth)
    )


async def _markets_local(session: AsyncSession, depth: int) -> list[dict[str, Any]]:
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
    return await on_leader(LEASE_TRADING, "tick", {}, lambda: get_trading_service().tick(session))


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
                body.interval_minutes
                if body.prices is not None
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
    request: Request,
    session: AsyncSession = Depends(get_db),
    user: UserModel = Depends(_trader),
):
    """Run a strategy once against current market data.

    With ``dry_run`` (the default) only the signals are returned; otherwise
    each signal is submitted as an order through the same validation and
    pre-trade risk checks as ``POST /orders``.
    """
    username = _username(user)
    result = await on_leader(
        LEASE_TRADING,
        "run_strategy",
        {"name": name, "body": body.model_dump(mode="json"), "username": username},
        lambda: _run_strategy_local(session, name, body, username),
    )
    if not body.dry_run:  # a dry run places no orders
        signals = _field(result, "signals")
        audit.record(
            session,
            request,
            "market.strategy_run",
            actor=user,
            target_type="strategy",
            target_id=name,
            details={"signals": len(signals) if isinstance(signals, list) else None},
        )
    return result


async def _run_strategy_local(
    session: AsyncSession, name: str, body: StrategyRunRequest, username: str | None
) -> dict[str, Any]:
    try:
        return await get_trading_service().run_strategy(
            session,
            name,
            params=body.params,
            dry_run=body.dry_run,
            submitted_by=username,
        )
    except TradingError as exc:
        _raise(exc)


# ---------------------------------------------------------------------------
# Forwarded venue operations (run on the trading-venue lease holder)
# ---------------------------------------------------------------------------


async def _h_submit_order(payload: dict[str, Any], session: AsyncSession) -> Any:
    body = OrderCreate.model_validate(payload["body"])
    return await _submit_order_local(session, body, payload.get("username"))


async def _h_cancel_order(payload: dict[str, Any], session: AsyncSession) -> Any:
    return await _cancel_local(session, payload["order_id"], payload.get("username"))


async def _h_portfolio(payload: dict[str, Any], session: AsyncSession) -> Any:
    return await _portfolio_local(session)


async def _h_markets(payload: dict[str, Any], session: AsyncSession) -> Any:
    return await _markets_local(session, int(payload.get("depth", 0)))


async def _h_tick(payload: dict[str, Any], session: AsyncSession) -> Any:
    return await get_trading_service().tick(session)


async def _h_run_strategy(payload: dict[str, Any], session: AsyncSession) -> Any:
    body = StrategyRunRequest.model_validate(payload["body"])
    return await _run_strategy_local(session, payload["name"], body, payload.get("username"))


for _method, _handler in (
    ("submit_order", _h_submit_order),
    ("cancel_order", _h_cancel_order),
    ("portfolio", _h_portfolio),
    ("markets", _h_markets),
    ("tick", _h_tick),
    ("run_strategy", _h_run_strategy),
):
    register_handler(LEASE_TRADING, _method, _handler)
