"""Tests for /metrics, HTTP instrumentation, EventBus metrics and request ids."""

from __future__ import annotations

import io
import json
import logging
import uuid

import pytest
import structlog
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from vpp import metrics as vpp_metrics
from vpp.api import observability
from vpp.api.middleware import PrometheusMiddleware, RequestIdMiddleware, route_template
from vpp.events import Event, EventType
from vpp.events.bus import EventBus
from vpp.logging import configure_logging
from vpp.metrics import REGISTRY, MetricsCollector
from vpp.settings import Settings


def _sample(name: str, labels: dict[str, str] | None = None) -> float:
    return REGISTRY.get_sample_value(name, labels or {}) or 0.0


# ---------------------------------------------------------------------------
# /metrics endpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_metrics_endpoint_is_mounted_and_unauthenticated(client: AsyncClient):
    resp = await client.get("/metrics")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    body = resp.text
    assert "vpp_api_requests_total" in body
    assert "vpp_info" in body


@pytest.mark.asyncio
async def test_http_metrics_use_route_template_not_raw_path(client: AsyncClient, auth_headers):
    rid = uuid.uuid4().hex
    labels = {"method": "GET", "endpoint": "/api/v1/resources/{resource_id}", "status": "404"}
    before = _sample("vpp_api_requests_total", labels)
    resp = await client.get(f"/api/v1/resources/{rid}", headers=auth_headers)
    assert resp.status_code == 404
    assert _sample("vpp_api_requests_total", labels) == before + 1
    # The raw id must never appear as a label value.
    metrics_text = (await client.get("/metrics")).text
    assert rid not in metrics_text
    assert _sample(
        "vpp_api_request_duration_seconds_count",
        {"method": "GET", "endpoint": "/api/v1/resources/{resource_id}"},
    ) >= 1


@pytest.mark.asyncio
async def test_unmatched_paths_share_one_label(client: AsyncClient):
    labels = {"method": "GET", "endpoint": "<unmatched>", "status": "404"}
    before = _sample("vpp_api_requests_total", labels)
    for i in range(3):
        await client.get(f"/no/such/path/{i}")
    assert _sample("vpp_api_requests_total", labels) == before + 3


@pytest.mark.asyncio
async def test_metrics_bearer_token(app, client: AsyncClient, monkeypatch):
    monkeypatch.setattr(app.state, "metrics_bearer_token", "s3cret", raising=False)
    assert (await client.get("/metrics")).status_code == 401
    bad = await client.get("/metrics", headers={"Authorization": "Bearer nope"})
    assert bad.status_code == 401
    ok = await client.get("/metrics", headers={"Authorization": "Bearer s3cret"})
    assert ok.status_code == 200


async def _status(app: FastAPI, path: str) -> int:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        return (await c.get(path)).status_code


@pytest.mark.asyncio
async def test_metrics_not_mounted_when_disabled():
    app = FastAPI()
    observability.install_observability(app, Settings(metrics_enabled=False))
    assert await _status(app, "/metrics") == 404
    # Alerts routes are mounted regardless (401: they need auth).
    assert await _status(app, "/api/v1/alerts") == 401


@pytest.mark.asyncio
async def test_metrics_not_mounted_without_prometheus_client(monkeypatch):
    monkeypatch.setattr(vpp_metrics, "_HAS_PROMETHEUS", False)
    app = FastAPI()
    observability.install_observability(app, Settings(metrics_enabled=True))
    assert await _status(app, "/metrics") == 404


def test_collector_is_noop_without_prometheus(monkeypatch):
    monkeypatch.setattr(vpp_metrics, "_HAS_PROMETHEUS", False)
    c = MetricsCollector()
    assert not c.enabled
    c.record_optimization("dispatch", "success", 1.0)
    c.record_order("day_ahead", "buy", "filled")
    c.set_resource_power("r", "battery", 1.0)
    vpp_metrics.observe_event(
        Event(event_type=EventType.RESOURCE_UPDATED, data={"resource_id": "r", "soc": 50}), c,
    )
    assert c.get_metrics_text().startswith(b"#")
    assert c.get_content_type() == "text/plain"


# ---------------------------------------------------------------------------
# EventBus -> metrics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resource_updated_events_update_gauges():
    bus = EventBus()
    sub = vpp_metrics.subscribe_event_bus(bus)
    rid = f"res-{uuid.uuid4().hex[:8]}"
    try:
        await bus.publish(Event(
            event_type=EventType.RESOURCE_ADDED,
            data={"id": rid, "resource_type": "battery"},
        ))
        await bus.publish(Event(
            event_type=EventType.RESOURCE_UPDATED,
            data={"resource_id": rid, "soc": 55.0, "power": -12.5},
            source="mqtt.telemetry_ingestion",
        ))
        assert _sample("vpp_battery_soc", {"resource_id": rid}) == pytest.approx(0.55)
        assert _sample(
            "vpp_resource_power_kw", {"resource_id": rid, "resource_type": "battery"},
        ) == -12.5
        assert _sample("vpp_resource_last_update_timestamp_seconds", {"resource_id": rid}) > 0

        # Modbus-style payload (no soc, current_power_kw) keeps the known type.
        await bus.publish(Event(
            event_type=EventType.RESOURCE_UPDATED,
            data={"resource_id": rid, "current_power_kw": 7.0},
            source="modbus.telemetry_ingestion",
        ))
        assert _sample(
            "vpp_resource_power_kw", {"resource_id": rid, "resource_type": "battery"},
        ) == 7.0

        await bus.publish(Event(event_type=EventType.RESOURCE_REMOVED, data={"id": rid}))
        assert REGISTRY.get_sample_value("vpp_battery_soc", {"resource_id": rid}) is None
        assert REGISTRY.get_sample_value(
            "vpp_resource_power_kw", {"resource_id": rid, "resource_type": "battery"},
        ) is None
    finally:
        bus.unsubscribe(sub)


@pytest.mark.asyncio
async def test_optimization_and_trading_events_update_counters():
    bus = EventBus()
    sub = vpp_metrics.subscribe_event_bus(bus)
    pt = f"pt_{uuid.uuid4().hex[:6]}"
    market = f"mkt_{uuid.uuid4().hex[:6]}"
    try:
        await bus.publish(Event(
            event_type=EventType.OPTIMIZATION_COMPLETED,
            data={"problem_type": pt, "solve_time_s": 0.2},
        ))
        await bus.publish(Event(
            event_type=EventType.OPTIMIZATION_FAILED, data={"problem_type": pt},
        ))
        assert _sample("vpp_optimization_runs_total", {"problem_type": pt, "status": "success"}) == 1
        assert _sample("vpp_optimization_runs_total", {"problem_type": pt, "status": "error"}) == 1
        assert _sample("vpp_optimization_duration_seconds_count", {"problem_type": pt}) == 1

        await bus.publish(Event(
            event_type=EventType.ORDER_SUBMITTED, data={"market": market, "side": "buy"},
        ))
        await bus.publish(Event(
            event_type=EventType.ORDER_FILLED, data={"market": market, "side": "buy"},
        ))
        await bus.publish(Event(
            event_type=EventType.TRADE_EXECUTED,
            data={"market": market, "side": "buy", "quantity_mwh": 2.5, "total_pnl": 42.0},
        ))
        assert _sample(
            "vpp_trading_orders_total", {"market": market, "side": "buy", "status": "submitted"},
        ) == 1
        assert _sample(
            "vpp_trading_orders_total", {"market": market, "side": "buy", "status": "filled"},
        ) == 1
        assert _sample("vpp_trading_trades_total", {"market": market, "side": "buy"}) == 1
        assert _sample("vpp_trading_volume_mwh_total", {"market": market, "side": "buy"}) == 2.5
        assert _sample("vpp_trading_pnl_total") == 42.0

        await bus.publish(Event(event_type=EventType.PROTOCOL_ERROR, data={"protocol": market}))
        assert _sample("vpp_protocol_errors_total", {"protocol": market}) == 1
    finally:
        bus.unsubscribe(sub)


def test_time_optimization_helper_records_success_and_error():
    pt = f"timed_{uuid.uuid4().hex[:6]}"
    with vpp_metrics.time_optimization(pt):
        pass
    with pytest.raises(ValueError), vpp_metrics.time_optimization(pt):
        raise ValueError("boom")
    assert _sample("vpp_optimization_runs_total", {"problem_type": pt, "status": "success"}) == 1
    assert _sample("vpp_optimization_runs_total", {"problem_type": pt, "status": "error"}) == 1
    assert _sample("vpp_optimization_duration_seconds_count", {"problem_type": pt}) == 2


def test_module_helpers_delegate():
    market = f"h_{uuid.uuid4().hex[:6]}"
    vpp_metrics.record_order(market, "sell", "rejected")
    vpp_metrics.record_trade(market, "sell", -1.5)
    vpp_metrics.set_trading_pnl(-3.0)
    vpp_metrics.record_alert_fired("critical", market)
    assert _sample("vpp_trading_orders_total", {"market": market, "side": "sell", "status": "rejected"}) == 1
    assert _sample("vpp_trading_volume_mwh_total", {"market": market, "side": "sell"}) == 1.5
    assert _sample("vpp_trading_pnl_total") == -3.0
    assert _sample("vpp_alerts_fired_total", {"severity": "critical", "rule": market}) == 1


# ---------------------------------------------------------------------------
# Request id middleware + structured logging
# ---------------------------------------------------------------------------


def _mini_app(collector: MetricsCollector | None = None) -> FastAPI:
    app = FastAPI()
    seen: dict[str, object] = {}

    @app.get("/items/{item_id}")
    async def item(item_id: str):
        seen["ctx"] = structlog.contextvars.get_contextvars().get("request_id")
        logging.getLogger("vpp.test.request").warning("handling item %s", item_id)
        return {"item_id": item_id}

    @app.get("/boom")
    async def boom():
        raise RuntimeError("boom")

    app.add_middleware(PrometheusMiddleware, collector=collector)
    app.add_middleware(RequestIdMiddleware)
    app.state.seen = seen
    return app


@pytest.mark.asyncio
async def test_request_id_generated_and_bound():
    app = _mini_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        resp = await c.get("/items/1")
    rid = resp.headers["X-Request-ID"]
    assert len(rid) == 32
    assert app.state.seen["ctx"] == rid
    # Context is cleared after the request.
    assert "request_id" not in structlog.contextvars.get_contextvars()


@pytest.mark.asyncio
async def test_incoming_request_id_is_propagated_or_replaced():
    app = _mini_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        ok = await c.get("/items/1", headers={"X-Request-ID": "abc-123.def"})
        bad = await c.get("/items/1", headers={"X-Request-ID": "x" * 500})
        evil = await c.get("/items/1", headers={"X-Request-ID": "a b<script>"})
    assert ok.headers["X-Request-ID"] == "abc-123.def"
    assert bad.headers["X-Request-ID"] != "x" * 500
    assert evil.headers["X-Request-ID"] != "a b<script>"
    assert len(ok.headers.get_list("X-Request-ID")) == 1


@pytest.mark.asyncio
async def test_request_id_on_app_responses(client: AsyncClient):
    resp = await client.get("/health", headers={"X-Request-ID": "corr-1"})
    assert resp.headers["X-Request-ID"] == "corr-1"


@pytest.mark.asyncio
async def test_stdlib_log_lines_carry_request_id():
    configure_logging(level="INFO", json_output=True)
    root = logging.getLogger()
    vpp_handler = next(h for h in root.handlers if getattr(h, "_vpp_structlog_handler", False))
    buf = io.StringIO()
    capture = logging.StreamHandler(buf)
    capture.setFormatter(vpp_handler.formatter)
    root.addHandler(capture)
    try:
        app = _mini_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            await c.get("/items/42", headers={"X-Request-ID": "trace-42"})
    finally:
        root.removeHandler(capture)
    lines = [json.loads(line) for line in buf.getvalue().splitlines() if line.strip()]
    handled = [ln for ln in lines if "handling item 42" in ln.get("event", "")]
    assert handled and handled[0]["request_id"] == "trace-42"
    access = [ln for ln in lines if ln.get("event") == "http_request"]
    assert access and access[0]["route"] == "/items/{item_id}"
    assert access[0]["status"] == 200
    assert access[0]["request_id"] == "trace-42"


def test_configure_logging_is_idempotent_and_keeps_foreign_handlers():
    root = logging.getLogger()
    foreign = logging.NullHandler()
    root.addHandler(foreign)
    try:
        configure_logging()
        configure_logging()
        ours = [h for h in root.handlers if getattr(h, "_vpp_structlog_handler", False)]
        assert len(ours) == 1
        assert foreign in root.handlers
    finally:
        root.removeHandler(foreign)


@pytest.mark.asyncio
async def test_prometheus_middleware_counts_500s():
    collector = MetricsCollector()
    app = _mini_app(collector)
    labels = {"method": "GET", "endpoint": "/boom", "status": "500"}
    before = _sample("vpp_api_requests_total", labels)
    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False), base_url="http://t",
    ) as c:
        resp = await c.get("/boom")
    assert resp.status_code == 500
    assert _sample("vpp_api_requests_total", labels) == before + 1


def test_route_template_without_route():
    assert route_template({"type": "http"}) == "<unmatched>"
