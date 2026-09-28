"""Simulated continuous-matching exchange for the VPP trading stack.

There is no connection to a real ISO/market operator here. This module is an
explicit, deterministic-when-seeded *simulation* of a continuous energy
market venue so that orders placed through the API have realistic life
cycles (resting, partial fills, cancels, expiry) instead of sitting in
``pending`` forever.

Liquidity model
---------------
On every :meth:`SimulatedExchange.tick` each market's book is rebuilt from
the synthetic depth produced by :class:`~vpp.trading.data.SimulatedDataProvider`
(``bid_levels`` / ``ask_levels``), rounded to the market's tick and lot
sizes. The platform's own orders execute only against that synthetic
liquidity -- they never match each other (no self-trading) -- and consume it
until the next tick rebuilds the book.

* Market / IOC orders sweep the book; any unfilled remainder is cancelled.
* FOK orders fill completely or not at all.
* Limit and iceberg orders take what is marketable and rest the remainder;
  resting orders are re-matched on every tick.
* Stop / stop-limit orders trigger when the best opposite price touches the
  stop price (buy: ask >= stop, sell: bid <= stop).
* ``DAY`` time-in-force orders expire when the (local) date rolls over.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from .orders import (
    LimitOrder,
    Order,
    OrderBook,
    OrderStatus,
    OrderType,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .data import MarketData, SimulatedDataProvider
    from .markets import Market

logger = logging.getLogger("trading.simulation")

_NON_RESTING = {OrderType.MARKET, OrderType.FILL_OR_KILL, OrderType.IMMEDIATE_OR_CANCEL}


@dataclass
class Fill:
    """One execution of a platform order against simulated liquidity."""

    order_id: str
    market: str
    side: str
    quantity: float
    price: float
    fee: float
    timestamp: datetime
    liquidity: str  # "taker" (on submission) or "maker" (resting order, later tick)


@dataclass
class OrderUpdate:
    """Terminal / non-fill status change produced by a tick (e.g. expiry)."""

    order: Order
    reason: str


@dataclass
class TickResult:
    """Everything that changed on one simulation tick."""

    timestamp: datetime
    market_data: dict[str, MarketData] = field(default_factory=dict)
    fills: list[Fill] = field(default_factory=list)
    touched_orders: list[Order] = field(default_factory=list)
    expired: list[OrderUpdate] = field(default_factory=list)


class OrderRejected(Exception):
    """Raised when the venue refuses an order (validation, unknown market)."""

    def __init__(self, reasons: list[str], code: str = "order_invalid") -> None:
        super().__init__("; ".join(reasons))
        self.reasons = list(reasons)
        self.code = code


def _round_down(value: float, step: float) -> float:
    return math.floor(value / step + 1e-9) * step


def _round_up(value: float, step: float) -> float:
    return math.ceil(value / step - 1e-9) * step


class SimulatedExchange:
    """Continuous-matching venue simulation over a set of :class:`Market` objects."""

    def __init__(
        self,
        markets: Iterable[Market],
        provider: SimulatedDataProvider,
        *,
        start: datetime | None = None,
    ) -> None:
        self.markets: dict[str, Market] = {m.name: m for m in markets}
        self.provider = provider
        self.provider.markets = list(self.markets)
        if not self.provider.is_connected:
            self.provider.connect()
        self.open_orders: dict[str, Order] = {}
        self.latest: dict[str, MarketData] = {}
        self.last_tick: datetime | None = None
        self.tick(start or datetime.now())

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    def _rebuild_book(self, market: Market, data: MarketData) -> None:
        book = OrderBook(market.name)
        for side, levels in (("buy", data.bid_levels), ("sell", data.ask_levels)):
            for price, quantity in levels:
                if side == "buy":
                    price = _round_down(price, market.tick_size)
                else:
                    price = _round_up(price, market.tick_size)
                quantity = _round_down(quantity, market.lot_size)
                if price <= 0 or quantity <= 0:
                    continue
                book.add_order(
                    LimitOrder(
                        market=market.name,
                        side=side,
                        quantity=quantity,
                        price=round(price, 10),
                        metadata={"liquidity_provider": True},
                    )
                )
        market.order_book = book

    def tick(self, timestamp: datetime | None = None) -> TickResult:
        """Advance prices, refresh liquidity, expire and re-match resting orders."""
        now = timestamp or datetime.now()
        result = TickResult(timestamp=now)
        for name, market in self.markets.items():
            data = self.provider.generate(name, now)
            if data is None:
                continue
            self._rebuild_book(market, data)
            best_bid = market.order_book.get_best_bid()
            best_ask = market.order_book.get_best_ask()
            data.bid_price = best_bid
            data.ask_price = best_ask
            data.last_price = round(data.last_price, 6) if data.last_price is not None else None
            market.update_market_data(data)
            self.latest[name] = data
            result.market_data[name] = data
        self.last_tick = now

        for order in list(self.open_orders.values()):
            if order.time_in_force == "DAY" and order.timestamp.date() < now.date():
                order.status = OrderStatus.EXPIRED
                self.open_orders.pop(order.id, None)
                result.expired.append(OrderUpdate(order=order, reason="DAY order expired"))
                continue
            fills = self._match(order, liquidity="maker", now=now)
            if fills:
                result.fills.extend(fills)
                result.touched_orders.append(order)
            if not order.is_active():
                self.open_orders.pop(order.id, None)
        return result

    def snapshot(self, market: str) -> MarketData | None:
        return self.latest.get(market)

    def last_prices(self) -> dict[str, float]:
        return {
            name: data.last_price
            for name, data in self.latest.items()
            if data.last_price is not None
        }

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def _match(self, order: Order, *, liquidity: str, now: datetime) -> list[Fill]:
        market = self.markets[order.market]
        fee_rate = market.transaction_fee + market.market_fee
        fills: list[Fill] = []
        for match in market.order_book.match_order(order):
            quantity = match["quantity"]
            fills.append(
                Fill(
                    order_id=order.id,
                    market=order.market,
                    side=order.side,
                    quantity=quantity,
                    price=match["price"],
                    fee=quantity * fee_rate,
                    timestamp=now,
                    liquidity=liquidity,
                )
            )
        # Keep the displayed quote consistent with consumed liquidity.
        data = self.latest.get(order.market)
        if fills and data is not None:
            data.bid_price = market.order_book.get_best_bid()
            data.ask_price = market.order_book.get_best_ask()
        return fills

    def validate(self, order: Order) -> None:
        market = self.markets.get(order.market)
        if market is None:
            raise OrderRejected(
                [f"Unknown market '{order.market}'. Available: {sorted(self.markets)}"],
                code="unknown_market",
            )
        errors = market.validate_order(order)
        if errors:
            raise OrderRejected(errors)

    def submit(self, order: Order, now: datetime | None = None) -> list[Fill]:
        """Validate and execute *order*; rests any eligible remainder.

        Raises :class:`OrderRejected` if the venue refuses the order.
        """
        self.validate(order)
        now = now or datetime.now()
        fills = self._match(order, liquidity="taker", now=now)
        if order.remaining_quantity > 1e-12:
            if order.order_type in _NON_RESTING:
                # Unfilled remainder of an immediate order is cancelled
                # (for FOK that is the whole order: filled nothing).
                order.status = OrderStatus.CANCELLED
            else:
                self.open_orders[order.id] = order
        return fills

    def restore(self, order: Order) -> None:
        """Re-register an already-accepted resting order (e.g. after restart)."""
        if order.market not in self.markets:
            raise OrderRejected([f"Unknown market '{order.market}'"], code="unknown_market")
        if order.is_active() and order.order_type not in _NON_RESTING:
            self.open_orders[order.id] = order

    def cancel(self, order_id: str) -> Order | None:
        order = self.open_orders.pop(order_id, None)
        if order is not None:
            order.cancel()
        return order

    def depth(self, market: str, levels: int = 5) -> dict[str, list[tuple[float, float]]]:
        book = self.markets[market].order_book
        depth = book.get_market_depth(levels)
        return {
            "bids": [(lvl["price"], lvl["quantity"]) for lvl in depth["bids"]],
            "asks": [(lvl["price"], lvl["quantity"]) for lvl in depth["asks"]],
        }
