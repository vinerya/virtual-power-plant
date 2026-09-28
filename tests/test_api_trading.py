"""Tests for the /api/v1/trading endpoints (backed by TradingService)."""

from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import AsyncClient  # noqa: TC002
from sqlalchemy import delete

from vpp.api.websocket import event_to_channel
from vpp.auth.security import create_access_token, get_password_hash
from vpp.db.models import OrderModel, TradeModel
from vpp.db.repositories import UserRepository
from vpp.events import EventType, get_event_bus
from vpp.trading.core import RiskLimits
from vpp.trading.service import (
    TradingServiceConfig,
    get_trading_service,
    reset_trading_service,
)

BASE = "/api/v1/trading"


def _config(**overrides) -> TradingServiceConfig:
    cfg = TradingServiceConfig(seed=7, base_volume=20.0)
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


@pytest_asyncio.fixture
async def trading(db_session):
    """Empty orders/trades tables and a fresh, seeded trading service."""
    await db_session.execute(delete(TradeModel))
    await db_session.execute(delete(OrderModel))
    await db_session.commit()
    svc = reset_trading_service(_config())
    yield svc
    await db_session.execute(delete(TradeModel))
    await db_session.execute(delete(OrderModel))
    await db_session.commit()
    reset_trading_service(_config())


@pytest_asyncio.fixture
async def operator_headers(db_session) -> dict[str, str]:
    user = await UserRepository.get_by_username(db_session, "testoperator")
    if user is None:
        user = await UserRepository.create_user(
            db_session,
            username="testoperator",
            hashed_password=get_password_hash("operatorpassword123"),
            role="operator",
        )
        await db_session.commit()
    token = create_access_token({"sub": user.id, "username": user.username, "role": user.role})
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def captured_events():
    events = []

    async def _capture(event):
        events.append(event)

    bus = get_event_bus()
    sub_id = bus.subscribe(_capture)
    yield events
    bus.unsubscribe(sub_id)


async def _markets(client, headers, depth=0):
    resp = await client.get(f"{BASE}/markets?depth={depth}", headers=headers)
    assert resp.status_code == 200
    return {m["market"]: m for m in resp.json()}


async def _order(client, headers, **body):
    return await client.post(f"{BASE}/orders", json=body, headers=headers)


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_market_buy_fills_and_persists(
    client: AsyncClient, operator_headers, trading, captured_events
):
    ask = (await _markets(client, operator_headers))["real_time"]["ask"]
    resp = await _order(
        client,
        operator_headers,
        order_type="market",
        market="real_time",
        side="buy",
        quantity=10.0,
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["status"] == "filled"
    assert body["filled_quantity"] == pytest.approx(10.0)
    assert body["remaining_quantity"] == pytest.approx(0.0)
    assert body["average_price"] >= ask
    assert body["metadata"]["submitted_by"] == "testoperator"
    assert body["fills"] and sum(f["quantity"] for f in body["fills"]) == pytest.approx(10.0)
    notional = sum(f["quantity"] * f["price"] for f in body["fills"])
    assert body["average_price"] == pytest.approx(notional / 10.0)

    trades = (
        await client.get(f"{BASE}/trades?order_id={body['id']}", headers=operator_headers)
    ).json()
    assert {t["id"] for t in trades} == {f["id"] for f in body["fills"]}
    fees = sum(t["fees"] for t in trades)
    assert fees == pytest.approx(10.0 * 0.07)

    stored = (await client.get(f"{BASE}/orders/{body['id']}", headers=operator_headers)).json()
    assert stored["status"] == "filled"

    types = [e.event_type for e in captured_events]
    assert EventType.ORDER_SUBMITTED in types
    assert EventType.ORDER_FILLED in types
    assert types.count(EventType.TRADE_EXECUTED) == len(body["fills"])
    trade_event = next(e for e in captured_events if e.event_type == EventType.TRADE_EXECUTED)
    assert trade_event.data["venue"] == "simulated"
    # Keys consumed by vpp.metrics.observe_event
    assert trade_event.data["quantity_mwh"] == trade_event.data["quantity"]
    assert trade_event.data["market"] == "real_time" and trade_event.data["side"] == "buy"
    assert isinstance(trade_event.data["total_pnl"], float)
    for e in captured_events:
        if e.event_type in (EventType.ORDER_SUBMITTED, EventType.ORDER_FILLED):
            assert e.data["market"] == "real_time" and e.data["side"] == "buy"
    assert event_to_channel(trade_event.event_type) == "market_data"


@pytest.mark.asyncio
async def test_portfolio_reflects_fills(client: AsyncClient, operator_headers, trading):
    resp = await _order(
        client,
        operator_headers,
        order_type="market",
        market="real_time",
        side="buy",
        quantity=10.0,
    )
    fills = resp.json()["fills"]
    notional = sum(f["quantity"] * f["price"] for f in fills)
    fees = sum(f["fees"] for f in fills)

    pf = (await client.get(f"{BASE}/portfolio", headers=operator_headers)).json()
    assert pf["initial_cash"] == pytest.approx(100_000.0)
    assert pf["cash"] == pytest.approx(100_000.0 - notional - fees)
    assert pf["total_trades"] == len(fills)
    assert pf["fees_paid"] == pytest.approx(fees)
    pos = {p["market"]: p for p in pf["positions"]}["real_time"]
    assert pos["quantity"] == pytest.approx(10.0)
    assert pos["average_price"] == pytest.approx(notional / 10.0)
    mark = pos["mark_price"]
    assert pos["unrealized_pnl"] == pytest.approx((mark - notional / 10.0) * 10.0)
    assert pf["equity"] == pytest.approx(pf["cash"] + 10.0 * mark)
    assert pf["total_pnl"] == pytest.approx(pf["equity"] - pf["initial_cash"])
    assert pf["total_pnl"] == pytest.approx(pf["realized_pnl"] + pf["unrealized_pnl"] - fees)
    assert pf["gross_exposure"] == pytest.approx(10.0 * mark)
    risk = pf["risk"]
    assert risk["var_95_1d"] > 0
    assert risk["limits"]["max_position"] == 50.0
    assert risk["volatility"]["real_time"]["source"] in ("estimated", "model_prior")
    assert pf["venue"] == "simulated"

    # Closing half realizes P&L.
    close = await _order(
        client,
        operator_headers,
        order_type="market",
        market="real_time",
        side="sell",
        quantity=5.0,
    )
    assert close.status_code == 201
    pf2 = (await client.get(f"{BASE}/portfolio", headers=operator_headers)).json()
    pos2 = {p["market"]: p for p in pf2["positions"]}["real_time"]
    assert pos2["quantity"] == pytest.approx(5.0)
    assert pf2["realized_pnl"] != 0.0
    realized_from_trades = sum(t["realized_pnl"] for t in close.json()["fills"])
    assert pf2["realized_pnl"] == pytest.approx(realized_from_trades)


@pytest.mark.asyncio
async def test_passive_limit_rests_and_cancel(
    client: AsyncClient, operator_headers, trading, captured_events
):
    resp = await _order(
        client,
        operator_headers,
        order_type="limit",
        market="day_ahead",
        side="buy",
        quantity=10.0,
        price=1.00,
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["status"] == "pending" and body["fills"] == []
    assert get_trading_service().portfolio_snapshot()["open_orders"] == 1

    resp = await client.delete(f"{BASE}/orders/{body['id']}", headers=operator_headers)
    assert resp.status_code == 200
    assert resp.json()["status"] == "cancelled"
    assert resp.json()["metadata"]["cancelled_by"] == "testoperator"
    assert get_trading_service().portfolio_snapshot()["open_orders"] == 0
    assert EventType.ORDER_CANCELLED in [e.event_type for e in captured_events]

    again = await client.delete(f"{BASE}/orders/{body['id']}", headers=operator_headers)
    assert again.status_code == 409
    assert again.json()["detail"]["code"] == "order_not_cancellable"

    missing = await client.post(f"{BASE}/orders/does-not-exist/cancel", headers=operator_headers)
    assert missing.status_code == 404


@pytest.mark.asyncio
async def test_cancel_via_post_alias(client: AsyncClient, auth_headers, trading):
    resp = await _order(
        client,
        auth_headers,
        order_type="limit",
        market="real_time",
        side="sell",
        quantity=1.0,
        price=2500.00,
    )
    order_id = resp.json()["id"]
    resp = await client.post(f"{BASE}/orders/{order_id}/cancel", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["status"] == "cancelled"


@pytest.mark.asyncio
async def test_marketable_limit_fills_at_or_better(client: AsyncClient, operator_headers, trading):
    ask = (await _markets(client, operator_headers))["day_ahead"]["ask"]
    limit = round(ask + 5, 2)
    resp = await _order(
        client,
        operator_headers,
        order_type="limit",
        market="day_ahead",
        side="buy",
        quantity=5.0,
        price=limit,
    )
    body = resp.json()
    assert body["filled_quantity"] > 0
    assert all(f["price"] <= limit for f in body["fills"])


@pytest.mark.asyncio
async def test_resting_order_fills_on_tick(
    client: AsyncClient, operator_headers, trading, captured_events
):
    last = (await _markets(client, operator_headers))["real_time"]["last_price"]
    resp = await _order(
        client,
        operator_headers,
        order_type="limit",
        market="real_time",
        side="sell",
        quantity=2.0,
        price=round(last * 2, 2),
    )
    order_id = resp.json()["id"]
    assert resp.json()["status"] == "pending"

    trading.provider.base_prices["real_time"] *= 5  # force the market through the limit
    captured_events.clear()
    tick = await client.post(f"{BASE}/markets/tick", headers=operator_headers)
    assert tick.status_code == 200
    assert tick.json()["fills"] >= 1

    stored = (await client.get(f"{BASE}/orders/{order_id}", headers=operator_headers)).json()
    assert stored["status"] == "filled"
    trades = (
        await client.get(f"{BASE}/trades?order_id={order_id}", headers=operator_headers)
    ).json()
    assert sum(t["quantity"] for t in trades) == pytest.approx(2.0)
    assert all(t["price"] >= round(last * 2, 2) for t in trades)
    types = [e.event_type for e in captured_events]
    assert EventType.MARKET_DATA in types and EventType.ORDER_FILLED in types


@pytest.mark.asyncio
async def test_ioc_and_fok(client: AsyncClient, operator_headers, trading):
    depth = (await _markets(client, operator_headers, depth=5))["real_time"]["depth"]
    price, available = depth["asks"][0]
    qty = round(min(available + 1.0, 49.0), 1)
    assert qty > available

    fok = await _order(
        client,
        operator_headers,
        order_type="fok",
        market="real_time",
        side="buy",
        quantity=qty,
        price=price,
    )
    assert fok.status_code == 201
    assert fok.json()["status"] == "cancelled"
    assert fok.json()["filled_quantity"] == 0.0 and fok.json()["fills"] == []

    ioc = await _order(
        client,
        operator_headers,
        order_type="limit",
        time_in_force="IOC",
        market="real_time",
        side="buy",
        quantity=qty,
        price=price,
    )
    body = ioc.json()
    assert body["order_type"] == "immediate_or_cancel"
    assert body["status"] == "cancelled"
    assert body["filled_quantity"] == pytest.approx(available)


@pytest.mark.asyncio
async def test_stop_limit_rests_with_parameters(client: AsyncClient, operator_headers, trading):
    last = (await _markets(client, operator_headers))["real_time"]["last_price"]
    resp = await _order(
        client,
        operator_headers,
        order_type="stop_limit",
        market="real_time",
        side="buy",
        quantity=1.0,
        stop_price=round(last * 3, 2),
        limit_price=round(last * 3.5, 2),
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["status"] == "pending"
    assert body["metadata"]["stop_price"] == round(last * 3, 2)


# ---------------------------------------------------------------------------
# Validation and risk
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invalid_orders_rejected(client: AsyncClient, operator_headers, trading):
    resp = await _order(
        client, operator_headers, order_type="market", market="nowhere", side="buy", quantity=1.0
    )
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "unknown_market"

    resp = await _order(
        client, operator_headers, order_type="limit", market="real_time", side="buy", quantity=1.0
    )
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "order_invalid"

    resp = await _order(
        client,
        operator_headers,
        order_type="limit",
        market="real_time",
        side="buy",
        quantity=1.0,
        price=45.123,
    )
    assert resp.status_code == 422
    assert "tick size" in resp.json()["detail"]["message"]

    resp = await _order(
        client,
        operator_headers,
        order_type="limit",
        market="real_time",
        side="buy",
        quantity=1.05,
        price=45.0,
    )
    assert resp.status_code == 422
    assert "lot size" in resp.json()["detail"]["message"]

    resp = await _order(
        client,
        operator_headers,
        order_type="teleport",
        market="real_time",
        side="buy",
        quantity=1.0,
    )
    assert resp.status_code == 422  # schema validation

    resp = await _order(
        client,
        operator_headers,
        order_type="limit",
        market="real_time",
        side="buy",
        quantity=1.0,
        price=45.0,
        time_in_force="WEEK",
    )
    assert resp.status_code == 422

    # Nothing was persisted for malformed orders.
    orders = (await client.get(f"{BASE}/orders", headers=operator_headers)).json()
    assert orders == []


@pytest.mark.asyncio
async def test_position_limit_rejection_is_persisted(
    client: AsyncClient, operator_headers, trading, captured_events
):
    resp = await _order(
        client,
        operator_headers,
        order_type="market",
        market="real_time",
        side="buy",
        quantity=60.0,
    )
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["code"] == "risk_limit_breached"
    assert any("Position limit" in r for r in detail["reasons"])

    stored = (
        await client.get(f"{BASE}/orders/{detail['order_id']}", headers=operator_headers)
    ).json()
    assert stored["status"] == "rejected"
    assert stored["metadata"]["reject_reasons"] == detail["reasons"]
    assert (await client.get(f"{BASE}/trades", headers=operator_headers)).json() == []
    rejected = [e for e in captured_events if e.event_type == EventType.ORDER_REJECTED]
    assert rejected and rejected[0].data["reasons"] == detail["reasons"]
    assert event_to_channel(EventType.ORDER_REJECTED) == "market_data"


@pytest.mark.asyncio
async def test_resting_orders_count_toward_limits(client: AsyncClient, operator_headers, trading):
    first = await _order(
        client,
        operator_headers,
        order_type="limit",
        market="day_ahead",
        side="buy",
        quantity=30.0,
        price=1.00,
    )
    assert first.status_code == 201
    second = await _order(
        client,
        operator_headers,
        order_type="limit",
        market="day_ahead",
        side="buy",
        quantity=30.0,
        price=1.00,
    )
    assert second.status_code == 422
    assert "resting orders" in " ".join(second.json()["detail"]["reasons"])
    # The opposite side is unaffected.
    other = await _order(
        client,
        operator_headers,
        order_type="limit",
        market="day_ahead",
        side="sell",
        quantity=30.0,
        price=2500.00,
    )
    assert other.status_code == 201


@pytest.mark.asyncio
async def test_var_limit_blocks_risk_but_allows_reduction(
    client: AsyncClient, operator_headers, trading
):
    opened = await _order(
        client,
        operator_headers,
        order_type="market",
        market="real_time",
        side="buy",
        quantity=20.0,
    )
    assert opened.status_code == 201
    trading.risk_limits.var_limit = 1.0

    more = await _order(
        client, operator_headers, order_type="market", market="real_time", side="buy", quantity=1.0
    )
    assert more.status_code == 422
    assert any("VaR" in r for r in more.json()["detail"]["reasons"])

    pf = (await client.get(f"{BASE}/portfolio", headers=operator_headers)).json()
    assert pf["risk"]["breach"] is True

    reduce = await _order(
        client,
        operator_headers,
        order_type="market",
        market="real_time",
        side="sell",
        quantity=10.0,
    )
    assert reduce.status_code == 201
    assert reduce.json()["status"] == "filled"


# ---------------------------------------------------------------------------
# Roles
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_viewer_is_read_only(client: AsyncClient, viewer_headers, operator_headers, trading):
    resp = await _order(
        client, viewer_headers, order_type="market", market="real_time", side="buy", quantity=1.0
    )
    assert resp.status_code == 403

    resting = await _order(
        client,
        operator_headers,
        order_type="limit",
        market="day_ahead",
        side="buy",
        quantity=1.0,
        price=1.00,
    )
    order_id = resting.json()["id"]
    assert (
        await client.delete(f"{BASE}/orders/{order_id}", headers=viewer_headers)
    ).status_code == 403
    assert (
        await client.post(f"{BASE}/orders/{order_id}/cancel", headers=viewer_headers)
    ).status_code == 403
    assert (await client.post(f"{BASE}/markets/tick", headers=viewer_headers)).status_code == 403
    run = await client.post(f"{BASE}/strategies/arbitrage/run", json={}, headers=viewer_headers)
    assert run.status_code == 403

    for path in (
        "/orders",
        "/trades",
        "/portfolio",
        "/markets",
        "/strategies",
        f"/orders/{order_id}",
    ):
        assert (await client.get(f"{BASE}{path}", headers=viewer_headers)).status_code == 200, path
    bt = await client.post(
        f"{BASE}/strategies/momentum/backtest",
        json={"synthetic": {"periods": 48}},
        headers=viewer_headers,
    )
    assert bt.status_code == 200


@pytest.mark.asyncio
async def test_requires_authentication(client: AsyncClient, trading):
    assert (await client.get(f"{BASE}/portfolio")).status_code == 401
    resp = await client.post(
        f"{BASE}/orders",
        json={"order_type": "market", "market": "real_time", "side": "buy", "quantity": 1.0},
    )
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Markets / persistence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_markets_listing(client: AsyncClient, auth_headers, trading, captured_events):
    markets = await _markets(client, auth_headers, depth=3)
    assert set(markets) == {"day_ahead", "real_time"}
    for m in markets.values():
        assert m["last_price"] > 0
        assert m["bid"] < m["ask"]
        assert m["source"] == "simulated" and m["price_unit"] == "$/MWh"
        assert m["status"] == "open"
        assert len(m["depth"]["bids"]) <= 3 and m["depth"]["asks"]
    plain = (await client.get(f"{BASE}/markets", headers=auth_headers)).json()
    assert all(m["depth"] is None for m in plain)

    before = markets["real_time"]["timestamp"]
    tick = (await client.post(f"{BASE}/markets/tick", headers=auth_headers)).json()
    assert len(tick["markets"]) == 2
    after = (await _markets(client, auth_headers))["real_time"]["timestamp"]
    assert after != before
    md = [e for e in captured_events if e.event_type == EventType.MARKET_DATA]
    assert {e.data["market"] for e in md} == {"day_ahead", "real_time"}
    assert all(e.data["source"] == "simulated" for e in md)
    assert event_to_channel(EventType.MARKET_DATA) == "market_data"


@pytest.mark.asyncio
async def test_state_is_rebuilt_from_database(client: AsyncClient, operator_headers, trading):
    buy = await _order(
        client, operator_headers, order_type="market", market="real_time", side="buy", quantity=8.0
    )
    resting = await _order(
        client,
        operator_headers,
        order_type="limit",
        market="day_ahead",
        side="buy",
        quantity=3.0,
        price=1.00,
    )
    before = (await client.get(f"{BASE}/portfolio", headers=operator_headers)).json()

    # Simulate a process restart: a brand-new service instance.
    reset_trading_service(_config(seed=99))
    after = (await client.get(f"{BASE}/portfolio", headers=operator_headers)).json()
    assert after["cash"] == pytest.approx(before["cash"])
    assert after["total_trades"] == before["total_trades"] == len(buy.json()["fills"])
    assert after["realized_pnl"] == pytest.approx(before["realized_pnl"])
    pos = {p["market"]: p for p in after["positions"]}["real_time"]
    assert pos["quantity"] == pytest.approx(8.0)
    assert after["open_orders"] == 1

    cancel = await client.delete(f"{BASE}/orders/{resting.json()['id']}", headers=operator_headers)
    assert cancel.status_code == 200
    assert get_trading_service().exchange.open_orders == {}


@pytest.mark.asyncio
async def test_unrestorable_resting_order_is_rejected_on_hydration(
    client, operator_headers, trading, db_session
):
    from vpp.db.repositories import TradingRepository

    row = await TradingRepository.create_order(
        db_session,
        order_type="limit",
        market="legacy_market",
        side="buy",
        quantity=1.0,
        price=10.0,
        remaining_quantity=1.0,
    )
    await db_session.commit()
    reset_trading_service(_config())
    await client.get(f"{BASE}/portfolio", headers=operator_headers)
    stored = (await client.get(f"{BASE}/orders/{row.id}", headers=operator_headers)).json()
    assert stored["status"] == "rejected"
    assert "restore" in stored["metadata"]["reject_reasons"][0]


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_strategies(client: AsyncClient, auth_headers, trading):
    resp = await client.get(f"{BASE}/strategies", headers=auth_headers)
    names = {s["name"] for s in resp.json()}
    assert {"arbitrage", "momentum", "mean_reversion", "ml"} <= names
    arb = next(s for s in resp.json() if s["name"] == "arbitrage")
    assert arb["min_markets"] == 2 and "price_threshold" in arb["parameters"]


@pytest.mark.asyncio
async def test_backtest_synthetic(client: AsyncClient, auth_headers, trading):
    body = {"synthetic": {"periods": 96, "interval_minutes": 60, "seed": 3}}
    resp = await client.post(
        f"{BASE}/strategies/arbitrage/backtest", json=body, headers=auth_headers
    )
    assert resp.status_code == 200, resp.text
    out = resp.json()
    assert out["data_source"] == "synthetic"
    assert out["periods"] == 96
    for key in ("total_pnl", "sharpe_ratio", "max_drawdown", "num_trades", "equity_curve"):
        assert key in out
    assert out["total_pnl"] == pytest.approx(out["final_equity"] - out["initial_cash"])
    assert out["assumptions"]
    again = await client.post(
        f"{BASE}/strategies/arbitrage/backtest", json=body, headers=auth_headers
    )
    assert again.json()["total_pnl"] == out["total_pnl"]  # seeded -> reproducible


@pytest.mark.asyncio
async def test_backtest_provided_prices(client: AsyncClient, auth_headers, trading):
    prices = [50.0 + (i % 12) for i in range(60)]
    body = {
        "params": {"lookback_hours": 3, "momentum_threshold": 0.05},
        "prices": {"real_time": prices},
        "interval_minutes": 60,
        "fee_per_unit": 0.0,
    }
    resp = await client.post(
        f"{BASE}/strategies/momentum/backtest", json=body, headers=auth_headers
    )
    assert resp.status_code == 200, resp.text
    out = resp.json()
    assert out["data_source"] == "provided" and out["markets"] == ["real_time"]
    assert out["num_trades"] > 0
    assert out["fees"] == 0.0


@pytest.mark.asyncio
async def test_backtest_errors(client: AsyncClient, auth_headers, trading):
    resp = await client.post(f"{BASE}/strategies/nope/backtest", json={}, headers=auth_headers)
    assert resp.status_code == 404
    resp = await client.post(
        f"{BASE}/strategies/momentum/backtest", json={"params": {"bogus": 1}}, headers=auth_headers
    )
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "backtest_invalid"
    resp = await client.post(
        f"{BASE}/strategies/arbitrage/backtest",
        json={"prices": {"real_time": [50.0, 51.0, 52.0]}},
        headers=auth_headers,
    )
    assert resp.status_code == 422
    resp = await client.post(
        f"{BASE}/strategies/momentum/backtest",
        json={"prices": {"a": [1.0, 2.0], "b": [1.0]}},
        headers=auth_headers,
    )
    assert resp.status_code == 422
    resp = await client.post(
        f"{BASE}/strategies/momentum/backtest",
        json={"prices": {"a": [1.0, -2.0]}},
        headers=auth_headers,
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_run_strategy_dry_run_and_live(client: AsyncClient, operator_headers, trading):
    markets = await _markets(client, operator_headers)
    gap = abs(markets["real_time"]["last_price"] - markets["day_ahead"]["last_price"])
    params = {"price_threshold": max(gap / 4, 0.01), "transaction_cost": 0.0, "base_quantity": 5.0}

    dry = await client.post(
        f"{BASE}/strategies/arbitrage/run",
        json={"params": params, "dry_run": True},
        headers=operator_headers,
    )
    assert dry.status_code == 200, dry.text
    signals = dry.json()["signals"]
    assert len(signals) == 2
    assert all(s["status"] == "not_submitted" for s in signals)
    assert (await client.get(f"{BASE}/orders", headers=operator_headers)).json() == []

    live = await client.post(
        f"{BASE}/strategies/arbitrage/run",
        json={"params": params, "dry_run": False},
        headers=operator_headers,
    )
    assert live.status_code == 200
    results = live.json()["signals"]
    assert len(results) == 2
    for r in results:
        assert r["order_id"]
        order = (
            await client.get(f"{BASE}/orders/{r['order_id']}", headers=operator_headers)
        ).json()
        assert order["metadata"]["strategy"] == "arbitrage"
        assert order["status"] == r["status"]

    missing = await client.post(f"{BASE}/strategies/nope/run", json={}, headers=operator_headers)
    assert missing.status_code == 404


@pytest.mark.asyncio
async def test_run_strategy_rejections_are_reported(
    client: AsyncClient, operator_headers, trading
):
    trading.risk_limits.max_position = 0.5
    markets = await _markets(client, operator_headers)
    gap = abs(markets["real_time"]["last_price"] - markets["day_ahead"]["last_price"])
    params = {"price_threshold": max(gap / 4, 0.01), "transaction_cost": 0.0, "base_quantity": 5.0}
    live = await client.post(
        f"{BASE}/strategies/arbitrage/run",
        json={"params": params, "dry_run": False},
        headers=operator_headers,
    )
    results = live.json()["signals"]
    assert results and all(r["status"] == "rejected" for r in results)
    assert all(any("Position limit" in reason for reason in r["reasons"]) for r in results)


def test_service_singleton_is_resettable():
    a = reset_trading_service(TradingServiceConfig(seed=1))
    assert get_trading_service() is a
    b = reset_trading_service(
        TradingServiceConfig(seed=1, risk_limits=RiskLimits(max_position=5.0))
    )
    assert get_trading_service() is b and b is not a
    assert b.risk_limits.max_position == 5.0


@pytest.mark.asyncio
async def test_lifespan_starts_market_data_loop(monkeypatch):
    """The simulated market-data ticker runs as a lifespan task when enabled."""
    import asyncio

    from vpp.api import app as app_module
    from vpp.db import engine as db_engine
    from vpp.settings import Settings
    from vpp.trading import service as service_module

    started = asyncio.Event()
    intervals = []

    async def fake_loop(interval_seconds):
        intervals.append(interval_seconds)
        started.set()
        await asyncio.sleep(3600)

    monkeypatch.setattr(service_module, "run_market_data_loop", fake_loop)
    saved = (db_engine._engine, db_engine._session_factory)
    try:
        for enabled in (True, False):
            settings = Settings(
                database_url="sqlite+aiosqlite:///./test_lifespan_trading.db",
                degradation_updater_enabled=False,
                trading_market_data_enabled=enabled,
                trading_market_data_interval_seconds=2.5,
            )
            monkeypatch.setattr(app_module, "get_settings", lambda s=settings: s)
            fastapi_app = app_module.create_app()
            async with fastapi_app.router.lifespan_context(fastapi_app):
                task = fastapi_app.state.trading_market_data_task
                if enabled:
                    await asyncio.wait_for(started.wait(), timeout=2.0)
                    assert task is not None and not task.done()
                else:
                    assert task is None
            if enabled:
                assert task.cancelled() or task.done()
    finally:
        db_engine._engine, db_engine._session_factory = saved
        import contextlib
        import os

        for suffix in ("", "-journal"):
            with contextlib.suppress(FileNotFoundError):
                os.remove("test_lifespan_trading.db" + suffix)
    assert intervals == [2.5]


@pytest.mark.asyncio
async def test_market_data_loop_ticks_and_survives_errors(monkeypatch, trading):
    import asyncio

    from vpp.trading import service as service_module

    calls = []
    original_tick = trading.tick

    async def flaky_tick(session, now=None):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("boom")
        return await original_tick(session, now)

    monkeypatch.setattr(trading, "tick", flaky_tick)
    task = asyncio.create_task(service_module.run_market_data_loop(0.01))
    try:
        for _ in range(200):
            if len(calls) >= 3:
                break
            await asyncio.sleep(0.01)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert len(calls) >= 3
