"""Application-level trading service.

One :class:`TradingService` per process owns the :class:`TradingEngine`
(markets, risk manager, portfolio), the :class:`SimulatedExchange` that
executes orders, and the glue that persists order/fill state to the
``orders``/``trades`` tables and publishes trading events to the EventBus.

Honesty notes
-------------
* The venue is **simulated** (see :mod:`vpp.trading.simulation`): no order
  leaves this process. Every market-data payload carries
  ``"source": "simulated"``.
* State is per process. The in-memory portfolio is rebuilt from the
  persisted ``trades`` table (and resting orders from ``orders``) on first
  use, so it survives restarts, but multiple API workers would each run
  their own independent simulated venue.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from itertools import pairwise
from typing import TYPE_CHECKING, Any

import numpy as np
from sqlalchemy import select

from vpp.db.models import OrderModel, TradeModel
from vpp.db.repositories import TradingRepository
from vpp.events import Event, EventType, get_event_bus

from .backtest import (
    STRATEGY_SPECS,
    BacktestResult,
    build_strategy,
    generate_synthetic_prices,
    run_backtest,
)
from .core import RiskLimits, TradingEngine
from .data import SimulatedDataProvider
from .markets import DayAheadMarket, Market, MarketStatus, RealTimeMarket
from .orders import Order, OrderStatus, OrderType, create_order, validate_order_parameters
from .portfolio import Portfolio, Trade
from .simulation import OrderRejected, SimulatedExchange

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.ext.asyncio import AsyncSession

    from .simulation import Fill

logger = logging.getLogger(__name__)

ORDER_TYPES = {
    "market", "limit", "stop", "stop_limit", "iceberg",
    "fok", "ioc", "fill_or_kill", "immediate_or_cancel",
}
TIME_IN_FORCE = {"GTC", "DAY", "IOC", "FOK"}
_ALIASES = {"fok": "fill_or_kill", "ioc": "immediate_or_cancel"}
_MIN_RETURNS_FOR_ESTIMATE = 20


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class TradingError(Exception):
    """Base class; carries an HTTP-ish status code and machine-readable code."""

    status_code = 422

    def __init__(self, message: str, *, code: str, reasons: list[str] | None = None,
                 order_id: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.reasons = reasons or [message]
        self.order_id = order_id

    def to_detail(self) -> dict[str, Any]:
        detail: dict[str, Any] = {"code": self.code, "message": self.message, "reasons": self.reasons}
        if self.order_id:
            detail["order_id"] = self.order_id
        return detail


class OrderRejectedError(TradingError):
    status_code = 422


class OrderNotFoundError(TradingError):
    status_code = 404


class OrderNotCancellableError(TradingError):
    status_code = 409


class StrategyNotFoundError(TradingError):
    status_code = 404


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class TradingServiceConfig:
    """Simulation + risk configuration. Prices are $/MWh, quantities MWh."""

    initial_cash: float = 100_000.0
    base_prices: dict[str, float] = field(
        default_factory=lambda: {"day_ahead": 45.0, "real_time": 50.0}
    )
    volatility: float = 0.35  # daily log-price volatility of the simulation
    mean_reversion: float = 3.0  # per day
    seasonal_amplitude: float = 0.25
    base_volume: float = 100.0  # scales synthetic book depth (MWh)
    seed: int | None = None
    tick_size: float = 0.01
    lot_size: float = 0.1
    max_price: float = 3000.0
    transaction_fee: float = 0.05  # $/MWh
    market_fee: float = 0.02  # $/MWh
    risk_limits: RiskLimits = field(default_factory=lambda: RiskLimits(
        max_position=50.0,
        max_daily_loss=5_000.0,
        max_drawdown=0.2,
        var_limit=10_000.0,
        concentration_limit=0.8,
    ))
    equity_curve_points: int = 5_000
    default_correlation: float = 0.5


def _to_local_naive(ts: datetime | None) -> datetime:
    """DB timestamps (UTC, possibly naive from SQLite) -> naive local time,
    the clock the trading package uses."""
    if ts is None:
        return datetime.now()
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone().replace(tzinfo=None)


def _iso(ts: datetime | None) -> str | None:
    return ts.isoformat() if ts is not None else None


def _round_to(value: float, step: float) -> float:
    return round(round(value / step) * step, 10)


def _load_metadata(row: OrderModel) -> dict[str, Any]:
    try:
        data = json.loads(row.metadata_json or "{}")
    except (TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


# ---------------------------------------------------------------------------
# Order construction
# ---------------------------------------------------------------------------


def build_order(
    *,
    order_type: str,
    market: str,
    side: str,
    quantity: float,
    price: float | None = None,
    stop_price: float | None = None,
    limit_price: float | None = None,
    visible_quantity: float | None = None,
    time_in_force: str = "GTC",
    metadata: dict[str, Any] | None = None,
) -> Order:
    """Translate API order fields into a :mod:`vpp.trading.orders` object.

    Raises ``ValueError`` with every validation problem found.
    """
    kind = (order_type or "").lower()
    tif = (time_in_force or "GTC").upper()
    if kind not in ORDER_TYPES:
        raise ValueError(f"Unknown order_type '{order_type}'. Allowed: {sorted(ORDER_TYPES)}")
    if tif not in TIME_IN_FORCE:
        raise ValueError(f"Unknown time_in_force '{time_in_force}'. Allowed: {sorted(TIME_IN_FORCE)}")
    kind = _ALIASES.get(kind, kind)
    # A limit order with an immediate time-in-force *is* an IOC/FOK order.
    if kind == "limit" and tif == "IOC":
        kind = "immediate_or_cancel"
    elif kind == "limit" and tif == "FOK":
        kind = "fill_or_kill"
    if kind == "stop" and price is None:
        price = stop_price
    if kind == "stop_limit" and limit_price is None:
        limit_price = price

    extra: dict[str, Any] = {}
    if kind == "stop_limit":
        extra = {"stop_price": stop_price, "limit_price": limit_price}
    elif kind == "iceberg" and visible_quantity is not None:
        extra = {"visible_quantity": visible_quantity}

    errors = validate_order_parameters(kind, market, side, quantity, price, **extra)
    if kind == "market" and price not in (None, 0, 0.0):
        errors.append("Market orders must not carry a price")
    if errors:
        raise ValueError("; ".join(errors))

    kwargs: dict[str, Any] = {"metadata": dict(metadata or {}), **extra}
    if kind not in ("fill_or_kill", "immediate_or_cancel"):
        kwargs["time_in_force"] = "IOC" if kind == "market" else tif
    return create_order(kind, market, side, quantity, price, **kwargs)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


@dataclass
class OrderResult:
    """Outcome of a submission: the persisted row plus the fills it produced."""

    order: OrderModel
    trades: list[TradeModel]


class TradingService:
    """Process-wide trading state + persistence/event glue."""

    def __init__(self, config: TradingServiceConfig | None = None) -> None:
        self.config = config or TradingServiceConfig()
        cfg = self.config
        self.engine = TradingEngine({"risk_limits": asdict(cfg.risk_limits)})
        self.engine.portfolio_manager.portfolio = Portfolio(initial_cash=cfg.initial_cash)
        for market in (DayAheadMarket("day_ahead"), RealTimeMarket("real_time")):
            self._configure_market(market)
            self.engine.add_market(market)
        for name in cfg.base_prices:
            if name not in self.engine.markets:
                market = RealTimeMarket(name)
                self._configure_market(market)
                self.engine.add_market(market)
        self.provider = SimulatedDataProvider({
            "markets": list(self.engine.markets),
            "base_prices": dict(cfg.base_prices),
            "volatility": cfg.volatility,
            "mean_reversion": cfg.mean_reversion,
            "seasonal_amplitude": cfg.seasonal_amplitude,
            "base_volume": cfg.base_volume,
            "seed": cfg.seed if cfg.seed is not None else int(uuid.uuid4().int % 2**31),
        })
        self.exchange = SimulatedExchange(self.engine.markets.values(), self.provider)
        self.started_at = datetime.now()
        self._lock = asyncio.Lock()
        self._hydrated = False
        self._order_strategy: dict[str, str] = {}
        self._update_equity(self.started_at)

    # -- setup -------------------------------------------------------------

    def _configure_market(self, market: Market) -> None:
        cfg = self.config
        market.tick_size = cfg.tick_size
        market.lot_size = cfg.lot_size
        market.max_price = cfg.max_price
        market.min_price = 0.0
        market.transaction_fee = cfg.transaction_fee
        market.market_fee = cfg.market_fee
        market.status = MarketStatus.OPEN  # no sessions -> continuous

    @property
    def portfolio(self) -> Portfolio:
        return self.engine.portfolio_manager.portfolio

    @property
    def risk_limits(self) -> RiskLimits:
        return self.engine.risk_manager.limits

    def _reset_state(self) -> None:
        self.engine.portfolio_manager.portfolio = Portfolio(initial_cash=self.config.initial_cash)
        self.engine.portfolio_manager.trades = []
        self.exchange.open_orders.clear()
        self._order_strategy.clear()
        self._hydrated = False

    async def ensure_ready(self, session: AsyncSession) -> None:
        """Rebuild in-memory state from the database on first use."""
        if self._hydrated:
            return
        async with self._lock:
            await self._hydrate(session)

    async def _hydrate(self, session: AsyncSession) -> None:
        if self._hydrated:
            return
        self._reset_state()
        trades = (await session.execute(
            select(TradeModel).order_by(TradeModel.created_at, TradeModel.id)
        )).scalars().all()
        for row in trades:
            trade = Trade(
                id=row.id, order_id=row.order_id, market=row.market, side=row.side,
                quantity=row.quantity, price=row.price, fees=row.fees or 0.0,
                timestamp=_to_local_naive(row.created_at), strategy=row.strategy or None,
            )
            # Direct replay (PortfolioManager.add_trade logs every trade at INFO).
            self.engine.portfolio_manager.trades.append(trade)
            self.portfolio.add_trade(trade)

        open_rows = (await session.execute(
            select(OrderModel)
            .where(OrderModel.status.in_(["pending", "partial"]))
            .order_by(OrderModel.created_at)
        )).scalars().all()
        for row in open_rows:
            meta = _load_metadata(row)
            try:
                order = build_order(
                    order_type=row.order_type, market=row.market, side=row.side,
                    quantity=row.quantity, price=row.price or None,
                    stop_price=meta.get("stop_price"), limit_price=meta.get("limit_price"),
                    visible_quantity=meta.get("visible_quantity"),
                    time_in_force=row.time_in_force or "GTC", metadata=meta,
                )
                self.exchange.validate(order)
            except (ValueError, OrderRejected) as exc:
                reason = f"Could not restore resting order: {exc}"
                meta["reject_reasons"] = [reason]
                row.status = OrderStatus.REJECTED.value
                row.metadata_json = json.dumps(meta)
                logger.warning("Order %s: %s", row.id, reason)
                continue
            order.id = row.id
            order.timestamp = _to_local_naive(row.created_at)
            order.filled_quantity = row.filled_quantity or 0.0
            order.remaining_quantity = (
                row.remaining_quantity if row.remaining_quantity is not None
                else row.quantity - order.filled_quantity
            )
            order.average_price = row.average_price or 0.0
            order.status = OrderStatus(row.status)
            if meta.get("triggered") and hasattr(order, "triggered"):
                order.triggered = True
            if order.order_type in (OrderType.MARKET, OrderType.FILL_OR_KILL,
                                    OrderType.IMMEDIATE_OR_CANCEL):
                # An immediate order can never legitimately be left resting.
                row.status = OrderStatus.CANCELLED.value
                continue
            self.exchange.restore(order)
            if meta.get("strategy"):
                self._order_strategy[order.id] = meta["strategy"]
        await session.flush()
        await session.commit()
        self._hydrated = True
        self._update_equity(datetime.now())
        logger.info(
            "Trading service hydrated: %d trades replayed, %d resting orders restored",
            len(trades), len(self.exchange.open_orders),
        )

    # -- analytics ---------------------------------------------------------

    def _update_equity(self, now: datetime) -> None:
        self.portfolio.update_equity_curve(
            self.exchange.last_prices(), timestamp=now,
            max_points=self.config.equity_curve_points,
        )

    def _normalized_returns(self) -> dict[str, dict[datetime, float]]:
        """Log returns scaled to a 1-day horizon, keyed by timestamp."""
        out: dict[str, dict[datetime, float]] = {}
        for market in self.engine.markets:
            history = self.provider.price_history.get(market, [])
            series: dict[datetime, float] = {}
            for prev, cur in pairwise(history):
                dt_days = (cur.timestamp - prev.timestamp).total_seconds() / 86400.0
                if dt_days <= 0 or prev.price <= 0 or cur.price <= 0:
                    continue
                series[cur.timestamp] = math.log(cur.price / prev.price) / math.sqrt(dt_days)
            out[market] = series
        return out

    def daily_volatility(self) -> dict[str, dict[str, Any]]:
        """Per-market daily return volatility and where it came from."""
        result: dict[str, dict[str, Any]] = {}
        for market, series in self._normalized_returns().items():
            values = list(series.values())
            if len(values) >= _MIN_RETURNS_FOR_ESTIMATE:
                result[market] = {"value": float(np.std(values, ddof=1)),
                                  "source": "estimated", "samples": len(values)}
            else:
                result[market] = {"value": self.config.volatility,
                                  "source": "model_prior", "samples": len(values)}
        return result

    def correlations(self) -> dict[str, dict[str, float]] | None:
        returns = self._normalized_returns()
        markets = list(returns)
        if len(markets) < 2:
            return None
        common = set.intersection(*(set(r) for r in returns.values()))
        if len(common) < _MIN_RETURNS_FOR_ESTIMATE:
            return None  # parametric_var falls back to default_correlation
        stamps = sorted(common)
        matrix = np.corrcoef([[returns[m][t] for t in stamps] for m in markets])
        return {
            a: {b: float(matrix[i][j]) for j, b in enumerate(markets) if i != j}
            for i, a in enumerate(markets)
        }

    def _vol_values(self) -> dict[str, float]:
        return {m: v["value"] for m, v in self.daily_volatility().items()}

    def _open_quantity(self, market: str, side: str) -> float:
        signed = 0.0
        for order in self.exchange.open_orders.values():
            if order.market == market and order.side == side:
                signed += order.remaining_quantity if side == "buy" else -order.remaining_quantity
        return signed

    def pre_trade_check(self, order: Order) -> list[str]:
        prices = self.exchange.last_prices()
        return self.engine.risk_manager.evaluate_order(
            order,
            self.portfolio,
            market_prices=prices,
            reference_price=prices.get(order.market),
            daily_volatility=self._vol_values(),
            correlations=self.correlations(),
            open_order_quantity=self._open_quantity(order.market, order.side),
        )

    # -- persistence helpers -------------------------------------------------

    @staticmethod
    def _order_metadata(order: Order) -> dict[str, Any]:
        meta = dict(order.metadata or {})
        for attr in ("stop_price", "limit_price", "visible_quantity"):
            if hasattr(order, attr):
                meta[attr] = getattr(order, attr)
        if getattr(order, "triggered", False):
            meta["triggered"] = True
        return meta

    async def _persist_new_order(self, session: AsyncSession, order: Order) -> OrderModel:
        return await TradingRepository.create_order(
            session,
            id=order.id,
            order_type=order.order_type.value,
            market=order.market,
            side=order.side,
            quantity=order.quantity,
            price=order.price or 0.0,
            status=order.status.value,
            filled_quantity=order.filled_quantity,
            remaining_quantity=max(order.remaining_quantity, 0.0),
            average_price=order.average_price,
            time_in_force=order.time_in_force,
            metadata=self._order_metadata(order),
        )

    async def _sync_order_row(self, session: AsyncSession, order: Order,
                              extra_meta: dict[str, Any] | None = None) -> OrderModel | None:
        meta = self._order_metadata(order)
        meta.update(extra_meta or {})
        return await TradingRepository.update_order_status(
            session, order.id, order.status.value,
            filled_quantity=order.filled_quantity,
            remaining_quantity=max(order.remaining_quantity, 0.0),
            average_price=order.average_price,
            metadata_json=json.dumps(meta),
        )

    async def _book_fills(self, session: AsyncSession, fills: Sequence[Fill],
                          events: list[Event]) -> list[TradeModel]:
        rows: list[TradeModel] = []
        for fill in fills:
            strategy = self._order_strategy.get(fill.order_id)
            trade = Trade(
                order_id=fill.order_id, market=fill.market, side=fill.side,
                quantity=fill.quantity, price=fill.price, timestamp=fill.timestamp,
                fees=fill.fee, strategy=strategy, execution_venue="simulated",
            )
            self.engine.portfolio_manager.add_trade(trade)
            self.engine.metrics["trades_executed"] += 1
            self.engine.metrics["total_volume"] += trade.quantity
            row = await TradingRepository.record_trade(
                session, id=trade.id, order_id=trade.order_id, market=trade.market,
                side=trade.side, quantity=trade.quantity, price=trade.price,
                fees=trade.fees, strategy=strategy or "", realized_pnl=trade.realized_pnl,
            )
            rows.append(row)
            events.append(Event(
                event_type=EventType.TRADE_EXECUTED,
                source="trading.exchange",
                data={
                    "trade_id": trade.id, "order_id": trade.order_id,
                    "market": trade.market, "side": trade.side,
                    "quantity": trade.quantity, "quantity_mwh": trade.quantity,
                    "price": trade.price,
                    "fees": trade.fees, "realized_pnl": trade.realized_pnl,
                    # Portfolio P&L (equity - initial cash) right after this fill.
                    "total_pnl": self.portfolio.get_equity(self.exchange.last_prices())
                    - self.portfolio.initial_cash,
                    "liquidity": fill.liquidity, "strategy": strategy,
                    "timestamp": _iso(trade.timestamp), "venue": "simulated",
                },
            ))
        return rows

    @staticmethod
    def _order_event(order: Order, event_type: EventType, source: str,
                     reasons: list[str] | None = None) -> Event:
        data = {
            "order_id": order.id, "market": order.market, "side": order.side,
            "order_type": order.order_type.value, "quantity": order.quantity,
            "price": order.price, "status": order.status.value,
            "filled_quantity": order.filled_quantity,
            "remaining_quantity": max(order.remaining_quantity, 0.0),
            "average_price": order.average_price,
        }
        if reasons:
            data["reasons"] = reasons
        return Event(
            event_type=event_type, source=source, data=data,
            severity="warning" if event_type == EventType.ORDER_REJECTED else "info",
        )

    def _status_events(self, order: Order, source: str) -> list[Event]:
        if order.status == OrderStatus.FILLED:
            return [self._order_event(order, EventType.ORDER_FILLED, source)]
        if order.status in (OrderStatus.CANCELLED, OrderStatus.EXPIRED):
            return [self._order_event(order, EventType.ORDER_CANCELLED, source)]
        return []

    async def _publish(self, events: Sequence[Event]) -> None:
        bus = get_event_bus()
        for event in events:
            try:
                await bus.publish(event)
            except Exception:
                logger.exception("Failed to publish %s", event.event_type.value)

    # -- commands ----------------------------------------------------------

    async def place_order(
        self,
        session: AsyncSession,
        *,
        order_type: str,
        market: str,
        side: str,
        quantity: float,
        price: float | None = None,
        stop_price: float | None = None,
        limit_price: float | None = None,
        visible_quantity: float | None = None,
        time_in_force: str = "GTC",
        metadata: dict[str, Any] | None = None,
        submitted_by: str | None = None,
        strategy: str | None = None,
    ) -> OrderResult:
        """Validate, risk-check, execute and persist one order.

        Raises :class:`OrderRejectedError` (422) for malformed orders, orders
        the venue refuses, and orders that breach pre-trade risk limits. Risk
        rejections are persisted with status ``rejected`` for the audit trail.
        """
        meta = dict(metadata or {})
        if submitted_by:
            meta["submitted_by"] = submitted_by
        if strategy:
            meta["strategy"] = strategy
        try:
            order = build_order(
                order_type=order_type, market=market, side=side, quantity=quantity,
                price=price, stop_price=stop_price, limit_price=limit_price,
                visible_quantity=visible_quantity, time_in_force=time_in_force,
                metadata=meta,
            )
        except (ValueError, TypeError) as exc:
            raise OrderRejectedError(str(exc), code="order_invalid") from exc

        events: list[Event] = []
        async with self._lock:
            await self._hydrate(session)
            try:
                self.exchange.validate(order)
            except OrderRejected as exc:
                raise OrderRejectedError(str(exc), code=exc.code, reasons=exc.reasons) from exc

            reasons = self.pre_trade_check(order)
            if reasons:
                order.status = OrderStatus.REJECTED
                order.metadata["reject_reasons"] = reasons
                try:
                    await self._persist_new_order(session, order)
                    await session.commit()
                except Exception:
                    await session.rollback()
                    raise
                rejected = self._order_event(order, EventType.ORDER_REJECTED,
                                             "trading.risk", reasons)
            else:
                rejected = None

            if rejected is None:
                now = datetime.now()
                try:
                    if strategy:
                        self._order_strategy[order.id] = strategy
                    fills = self.exchange.submit(order, now)
                    row = await self._persist_new_order(session, order)
                    trades = await self._book_fills(session, fills, events)
                    await session.commit()
                    # Load server-generated columns (timestamps) while we
                    # still hold the session, so callers never lazy-load.
                    for obj in (row, *trades):
                        await session.refresh(obj)
                except Exception:
                    # In-memory state may now be ahead of the database:
                    # rebuild from the DB on next use rather than diverge.
                    await session.rollback()
                    self.exchange.open_orders.pop(order.id, None)
                    self._hydrated = False
                    raise
                self._update_equity(now)
                events.insert(0, self._order_event(order, EventType.ORDER_SUBMITTED, "trading.orders"))
                events.extend(self._status_events(order, "trading.orders"))

        if rejected is not None:
            await self._publish([rejected])
            raise OrderRejectedError(
                "Order rejected by pre-trade risk checks",
                code="risk_limit_breached", reasons=reasons, order_id=order.id,
            )
        await self._publish(events)
        return OrderResult(order=row, trades=trades)

    async def cancel_order(self, session: AsyncSession, order_id: str,
                           cancelled_by: str | None = None) -> OrderModel:
        events: list[Event] = []
        async with self._lock:
            await self._hydrate(session)
            row = await TradingRepository.get_order(session, order_id)
            if row is None:
                raise OrderNotFoundError("Order not found", code="order_not_found")
            if row.status not in (OrderStatus.PENDING.value, OrderStatus.PARTIAL.value):
                raise OrderNotCancellableError(
                    f"Order is {row.status} and can no longer be cancelled",
                    code="order_not_cancellable", order_id=order_id,
                )
            order = self.exchange.cancel(order_id)
            meta = _load_metadata(row)
            if cancelled_by:
                meta["cancelled_by"] = cancelled_by
            row.status = OrderStatus.CANCELLED.value
            row.metadata_json = json.dumps(meta)
            await session.flush()
            await session.commit()
            await session.refresh(row)
            if order is not None:
                events.append(self._order_event(order, EventType.ORDER_CANCELLED, "trading.orders"))
            else:
                events.append(Event(
                    event_type=EventType.ORDER_CANCELLED, source="trading.orders",
                    data={"order_id": row.id, "market": row.market, "side": row.side,
                          "status": row.status},
                ))
        await self._publish(events)
        return row

    async def tick(self, session: AsyncSession, now: datetime | None = None) -> dict[str, Any]:
        """Advance the simulated venue one step and persist anything that filled."""
        events: list[Event] = []
        async with self._lock:
            await self._hydrate(session)
            result = self.exchange.tick(now or datetime.now())
            try:
                trades = await self._book_fills(session, result.fills, events)
                for order in result.touched_orders:
                    await self._sync_order_row(session, order)
                    events.extend(self._status_events(order, "trading.exchange"))
                for update in result.expired:
                    await self._sync_order_row(session, update.order, {"expired_reason": update.reason})
                    events.extend(self._status_events(update.order, "trading.exchange"))
                await session.commit()
            except Exception:
                await session.rollback()
                self._hydrated = False
                raise
            self._update_equity(result.timestamp)
            market_events = [
                Event(event_type=EventType.MARKET_DATA, source="trading.simulation",
                      data=self.market_snapshot(name))
                for name in result.market_data
            ]
        await self._publish(market_events + events)
        return {
            "timestamp": result.timestamp,
            "markets": [self.market_snapshot(name) for name in result.market_data],
            "fills": len(trades),
            "expired_orders": [u.order.id for u in result.expired],
        }

    # -- queries -----------------------------------------------------------

    def market_snapshot(self, name: str, depth_levels: int = 0) -> dict[str, Any]:
        market = self.engine.markets[name]
        data = self.exchange.snapshot(name)
        snap: dict[str, Any] = {
            "market": name,
            "market_type": market.market_type.value,
            "status": "open" if market.is_market_open() else market.status.value,
            "venue": "simulated",
            "source": "simulated",
            "currency": "USD",
            "price_unit": "$/MWh",
            "quantity_unit": "MWh",
            "tick_size": market.tick_size,
            "lot_size": market.lot_size,
            "fee_per_unit": market.transaction_fee + market.market_fee,
            "last_price": data.last_price if data else None,
            "bid": data.bid_price if data else None,
            "ask": data.ask_price if data else None,
            "volume": data.volume if data else 0.0,
            "timestamp": _iso(data.timestamp) if data else None,
        }
        if depth_levels:
            snap["depth"] = self.exchange.depth(name, depth_levels)
        return snap

    def markets(self, depth_levels: int = 0) -> list[dict[str, Any]]:
        return [self.market_snapshot(name, depth_levels) for name in self.engine.markets]

    def portfolio_snapshot(self) -> dict[str, Any]:
        prices = self.exchange.last_prices()
        portfolio = self.portfolio
        risk = self.engine.risk_manager.assess_portfolio(
            portfolio, prices, self._vol_values(), self.correlations(),
        )
        positions = []
        for market, position in sorted(portfolio.positions.items()):
            mark = prices.get(market, position.average_price)
            positions.append({
                "market": market,
                "quantity": position.quantity,
                "average_price": position.average_price,
                "mark_price": mark,
                "unrealized_pnl": position.calculate_unrealized_pnl(mark),
                "realized_pnl": position.realized_pnl,
                "notional_value": position.get_notional_value(mark),
            })
        equity = portfolio.get_equity(prices)
        realized = portfolio.calculate_realized_pnl()
        unrealized = portfolio.calculate_unrealized_pnl(prices)
        limits = self.risk_limits
        return {
            "cash": portfolio.cash,
            "equity": equity,
            "initial_cash": portfolio.initial_cash,
            "total_pnl": equity - portfolio.initial_cash,
            "realized_pnl": realized,
            "unrealized_pnl": unrealized,
            "fees_paid": portfolio.total_fees,
            "max_drawdown": risk["max_drawdown"],
            "current_drawdown": risk["current_drawdown"],
            "gross_exposure": risk["gross_exposure"],
            "net_exposure": risk["net_exposure"],
            "positions": positions,
            "total_trades": len(portfolio.trades),
            "open_orders": len(self.exchange.open_orders),
            "risk": {
                "var_95_1d": risk["var_95_1d"],
                "var_method": "parametric (delta-normal), 1-day horizon, 95% one-sided",
                "daily_pnl": risk["daily_pnl"],
                "concentrations": risk["concentrations"],
                "breach": risk["breach"],
                "breaches": risk["breaches"],
                "volatility": self.daily_volatility(),
                "limits": {
                    "max_position": limits.max_position,
                    "max_daily_loss": limits.max_daily_loss,
                    "max_drawdown": limits.max_drawdown,
                    "var_limit": limits.var_limit,
                    "concentration_limit": limits.concentration_limit,
                },
            },
            "venue": "simulated",
            "last_updated": datetime.now(timezone.utc),
        }

    # -- strategies ----------------------------------------------------------

    @staticmethod
    def list_strategies() -> list[dict[str, Any]]:
        return [
            {"name": spec.name, "description": spec.description,
             "parameters": dict(spec.parameters), "min_markets": spec.min_markets}
            for spec in STRATEGY_SPECS.values()
        ]

    @staticmethod
    def backtest(
        name: str,
        *,
        params: dict[str, Any] | None = None,
        prices: dict[str, list[float]] | None = None,
        timestamps: list[datetime] | None = None,
        interval_minutes: float = 60.0,
        periods: int = 168,
        seed: int = 42,
        initial_cash: float = 100_000.0,
        fee_per_unit: float = 0.07,
        half_spread: float = 0.001,
    ) -> BacktestResult:
        if name not in STRATEGY_SPECS:
            raise StrategyNotFoundError(f"Unknown strategy '{name}'", code="strategy_not_found")
        spec = STRATEGY_SPECS[name]
        try:
            strategy = build_strategy(name, params)
            if prices is None:
                markets = spec.parameters.get("markets") or ["day_ahead", "real_time"]
                timestamps, prices = generate_synthetic_prices(
                    markets, periods=periods, interval_minutes=interval_minutes, seed=seed,
                )
            elif len(prices) < spec.min_markets:
                raise ValueError(f"Strategy '{name}' needs price series for at least "
                                 f"{spec.min_markets} markets")
            return run_backtest(
                strategy, prices, timestamps=timestamps, interval_minutes=interval_minutes,
                initial_cash=initial_cash, fee_per_unit=fee_per_unit, half_spread=half_spread,
            )
        except (ValueError, TypeError) as exc:
            raise TradingError(str(exc), code="backtest_invalid") from exc

    async def run_strategy(
        self,
        session: AsyncSession,
        name: str,
        *,
        params: dict[str, Any] | None = None,
        dry_run: bool = False,
        submitted_by: str | None = None,
    ) -> dict[str, Any]:
        """Generate signals from live (simulated) market data; optionally trade them."""
        if name not in STRATEGY_SPECS:
            raise StrategyNotFoundError(f"Unknown strategy '{name}'", code="strategy_not_found")
        try:
            strategy = build_strategy(name, params)
        except (ValueError, TypeError) as exc:
            raise TradingError(str(exc), code="strategy_invalid") from exc

        async with self._lock:
            await self._hydrate(session)
            history = getattr(strategy, "price_history", None)
            if isinstance(history, dict):
                # Prime stateful strategies with the venue's recent prices
                # (all but the latest point, which generate_signals adds).
                for market, points in self.provider.price_history.items():
                    history[market] = [(p.timestamp, p.price) for p in points[:-1]]
            signals = strategy.generate_signals(dict(self.exchange.latest), self.portfolio)

        results: list[dict[str, Any]] = []
        for signal in signals:
            market = signal["market"]
            market_obj = self.engine.markets.get(market)
            lot = market_obj.lot_size if market_obj else self.config.lot_size
            tick = market_obj.tick_size if market_obj else self.config.tick_size
            quantity = math.floor(float(signal["quantity"]) / lot + 1e-9) * lot
            order_type = "market" if signal.get("order_type") == "market" else "limit"
            price = None if order_type == "market" else _round_to(float(signal["price"]), tick)
            entry: dict[str, Any] = {
                "market": market, "side": signal["action"], "order_type": order_type,
                "quantity": round(quantity, 10), "price": price,
                "confidence": float(signal.get("confidence", 0.0)),
            }
            if dry_run or quantity <= 0:
                entry["status"] = "not_submitted" if dry_run else "skipped"
                results.append(entry)
                continue
            try:
                outcome = await self.place_order(
                    session, order_type=order_type, market=market, side=signal["action"],
                    quantity=round(quantity, 10), price=price,
                    submitted_by=submitted_by, strategy=name,
                    metadata={"signal_confidence": entry["confidence"]},
                )
                entry.update(order_id=outcome.order.id, status=outcome.order.status)
            except OrderRejectedError as exc:
                entry.update(order_id=exc.order_id, status="rejected", reasons=exc.reasons)
            results.append(entry)
        return {"strategy": name, "dry_run": dry_run, "signals": results}


# ---------------------------------------------------------------------------
# Process-wide instance
# ---------------------------------------------------------------------------

_service: TradingService | None = None


def get_trading_service() -> TradingService:
    """Return the process-wide trading service, creating it lazily."""
    global _service
    if _service is None:
        _service = TradingService()
    return _service


def reset_trading_service(config: TradingServiceConfig | None = None) -> TradingService:
    """Replace the process-wide service with a fresh one (tests, reconfiguration)."""
    global _service
    _service = TradingService(config)
    return _service


async def run_market_data_loop(interval_seconds: float) -> None:
    """Background task: tick the simulated venue and publish market data."""
    from vpp.db.engine import get_session_factory

    factory = get_session_factory()
    while True:
        # Sleep first: the service already holds a fresh snapshot from
        # construction, and short-lived app lifespans need not touch the DB.
        await asyncio.sleep(interval_seconds)
        try:
            async with factory() as session:
                await get_trading_service().tick(session)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Trading market-data tick failed")
