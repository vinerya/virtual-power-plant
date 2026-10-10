"""Optional OpenTelemetry tracing (vpp.tracing): no-op path, span tree, propagation, logs."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from vpp import tracing

# ---------------------------------------------------------------------------
# Tracing off / extra missing
# ---------------------------------------------------------------------------


def test_span_is_a_shared_noop_when_disabled():
    tracing.disable()
    assert not tracing.is_enabled()
    sp = tracing.span("vpp.test", resource_id="r1", kw=1.5)
    assert sp is tracing.span("vpp.other")  # one shared object, nothing allocated
    with sp as inner:
        inner.set(status="ok")
        inner.error("ignored")
        assert inner.recording is False
    assert tracing.inject() == {}
    assert tracing.current_ids() is None
    with tracing.continue_trace({"traceparent": "00-" + "1" * 32 + "-" + "2" * 16 + "-01"}):
        assert tracing.current_ids() is None
    assert tracing.add_trace_ids(None, "info", {"event": "x"}) == {"event": "x"}


def test_disabled_span_costs_nothing_measurable():
    tracing.disable()
    n = 100_000

    def traced():
        for i in range(n):
            with tracing.span("vpp.actuator.setpoint", resource_id="r", requested_kw=i) as sp:
                sp.set(status="accepted")

    def plain():
        for _i in range(n):
            pass

    best = min(_timed(traced) for _ in range(3)) - min(_timed(plain) for _ in range(3))
    # A few hundred ns per span site at most (well under a microsecond on CI
    # hardware); a dispatch writes a handful of devices, so this is noise.
    assert best / n < 2e-6, f"{best / n * 1e9:.0f} ns per disabled span"


def _timed(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def test_setup_is_a_noop_unless_enabled():
    tracing.disable()
    assert tracing.setup_tracing(SimpleNamespace(tracing_enabled=False)) is False
    assert not tracing.is_enabled()


def test_missing_extra_logs_a_warning_and_continues(monkeypatch, caplog):
    tracing.disable()
    for mod in (
        "opentelemetry",
        "opentelemetry.trace",
        "opentelemetry.sdk",
        "opentelemetry.sdk.resources",
        "opentelemetry.sdk.trace",
        "opentelemetry.sdk.trace.export",
    ):
        monkeypatch.setitem(sys.modules, mod, None)  # import raises ImportError
    with caplog.at_level(logging.WARNING, logger="vpp.tracing"):
        assert tracing.setup_tracing(SimpleNamespace(tracing_enabled=True)) is False
    assert not tracing.is_enabled()
    assert "virtual-power-plant[tracing]" in caplog.text
    with tracing.span("vpp.test") as sp:  # still a working no-op
        sp.set(a=1)


def test_core_package_does_not_import_opentelemetry():
    # Only the SDK, exporters and instrumentation are ours to load lazily:
    # FastAPI >= 0.143 imports the OpenTelemetry *API* itself when installed.
    code = (
        "import sys, vpp, vpp.tracing, vpp.logging, vpp.api.app, vpp.cluster.rpc, "
        "vpp.control.actuator, vpp.optimization.planning; "
        "prefixes = ('opentelemetry.sdk', 'opentelemetry.exporter', "
        "'opentelemetry.instrumentation'); "
        "bad = sorted(m for m in sys.modules if m.startswith(prefixes)); "
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_settings_flag_defaults_off(monkeypatch):
    from vpp.settings import Settings

    monkeypatch.delenv("VPP_TRACING_ENABLED", raising=False)
    assert Settings(_env_file=None).tracing_enabled is False
    monkeypatch.setenv("VPP_TRACING_ENABLED", "true")
    assert Settings(_env_file=None).tracing_enabled is True


# ---------------------------------------------------------------------------
# With the SDK (tracing extra)
# ---------------------------------------------------------------------------


@pytest.fixture
def spans():
    """Record spans in memory for the duration of a test."""
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracing.enable(provider)
    try:
        yield exporter
    finally:
        tracing.disable()
        provider.shutdown()


def _by_name(finished, name):
    return [s for s in finished if s.name == name]


def test_span_attributes_and_errors(spans):
    with tracing.span("vpp.test", resource_id="r1", skipped=None, obj=object()) as sp:
        assert sp.recording
        sp.set(status="failed", verified=False, ids=["a", "b"])
        sp.error("device said no")
    [s] = spans.get_finished_spans()
    assert s.attributes["vpp.resource_id"] == "r1"
    assert "vpp.skipped" not in s.attributes
    assert isinstance(s.attributes["vpp.obj"], str)
    assert s.attributes["vpp.verified"] is False
    assert list(s.attributes["vpp.ids"]) == ["a", "b"]
    assert s.status.status_code.name == "ERROR"
    assert s.status.description == "device said no"


def test_exception_marks_span_failed(spans):
    with pytest.raises(RuntimeError), tracing.span("vpp.boom"):
        raise RuntimeError("bang")
    [s] = spans.get_finished_spans()
    assert s.status.status_code.name == "ERROR"
    assert any(e.name == "exception" for e in s.events)


def test_inject_and_continue_trace_round_trip(spans):
    with tracing.span("vpp.parent"):
        carrier = tracing.inject()
        parent_ids = tracing.current_ids()
    assert carrier["traceparent"].startswith("00-" + parent_ids[0])
    with tracing.continue_trace(carrier), tracing.span("vpp.child"):
        assert tracing.current_ids()[0] == parent_ids[0]
    child = _by_name(spans.get_finished_spans(), "vpp.child")[0]
    assert format(child.parent.span_id, "016x") == parent_ids[1]


@pytest.fixture
def control_actuator():
    from test_device_control import FakeWriter

    from vpp.control.actuator import SetpointActuator, set_setpoint_actuator

    writers: dict = {}

    def factory(rid, modbus, control, rated):
        return writers.setdefault(rid, FakeWriter())

    actuator = SetpointActuator(enabled=True, writer_factory=factory)
    set_setpoint_actuator(actuator)
    yield writers
    set_setpoint_actuator(None)


async def test_dispatch_request_produces_the_span_tree(app, auth_headers, spans, control_actuator):
    """POST /dispatch?apply -> request span -> dispatch -> solve + per-device writes."""
    from vpp.api.app import create_app
    from vpp.db import engine as db_engine

    # A fresh app picks up the active tracer (FastAPI instrumentation); the
    # session-scoped ``app`` fixture already initialised the database.
    traced_app = create_app()
    assert db_engine._engine is not None
    tracing.instrument_sqlalchemy(db_engine._engine)
    control = {"host": "10.0.0.9", "control": {"enabled": True, "address": 40, "unit": "W"}}

    async with AsyncClient(transport=ASGITransport(app=traced_app), base_url="http://t") as c:
        ids = []
        for i in range(2):
            resp = await c.post(
                "/api/v1/resources/",
                json={
                    "name": f"trace-battery-{i}-{time.monotonic_ns()}",
                    "resource_type": "battery",
                    "rated_power": 50.0,
                    "metadata": {"capacity_kwh": 400, "soc": 0.5, "modbus": control},
                },
                headers=auth_headers,
            )
            assert resp.status_code == 201, resp.text
            ids.append(resp.json()["id"])
        spans.clear()
        resp = await c.post(
            "/api/v1/optimization/dispatch",
            json={"target_power_kw": 60.0, "resource_ids": ids, "apply": True},
            headers=auth_headers,
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["applied"] is True

    finished = spans.get_finished_spans()
    by_id = {s.context.span_id: s for s in finished}
    [dispatch] = _by_name(finished, "vpp.dispatch.execute")
    [solve] = _by_name(finished, "vpp.optimization.solve")
    writes = _by_name(finished, "vpp.actuator.setpoint")

    # Root: the HTTP server span created by the FastAPI instrumentation.
    root = by_id[dispatch.parent.span_id]
    while root.parent is not None and root.parent.span_id in by_id:
        root = by_id[root.parent.span_id]
    assert root.kind.name == "SERVER"
    assert "/api/v1/optimization/dispatch" in root.name
    assert {s.context.trace_id for s in finished} == {root.context.trace_id}

    # Solve (in the worker thread) and every device write hang off the dispatch.
    assert solve.parent.span_id == dispatch.context.span_id
    assert solve.attributes["vpp.strategy"] == "power_allocation"
    assert solve.attributes["vpp.solver"]
    assert solve.attributes["vpp.status"] in ("success", "fallback_used")
    assert "vpp.objective" in solve.attributes
    assert "vpp.horizon" in solve.attributes
    assert dispatch.attributes["vpp.run_id"] == body["run_id"]
    assert dispatch.attributes["vpp.status"] == body["status"]

    assert sorted(w.attributes["vpp.resource_id"] for w in writes) == sorted(ids)
    for w in writes:
        assert w.parent.span_id == dispatch.context.span_id
        assert w.attributes["vpp.status"] == "accepted"
        assert w.attributes["vpp.verified"] is True
        assert w.attributes["vpp.failed"] is False
        assert w.attributes["vpp.profile"]
        assert w.attributes["vpp.run_id"] == body["run_id"]

    # SQLAlchemy statements are traced inside the same request.
    assert any(s.attributes.get("db.system") or "db.statement" in s.attributes for s in finished)


async def test_failed_device_write_marks_its_span(spans):
    from test_device_control import FakeWriter

    from vpp.control.actuator import SetpointActuator
    from vpp.optimization.planning import FleetAsset

    writer = FakeWriter()
    writer.fail = True
    actuator = SetpointActuator(enabled=True, writer_factory=lambda *a: writer)
    asset = FleetAsset(
        id="b1",
        name="b1",
        resource_type="battery",
        rated_power_kw=50.0,
        online=True,
        metadata={"modbus": {"host": "h", "control": {"enabled": True, "address": 40}}},
    )
    [d] = await actuator.apply([asset], {"b1": 10.0}, source="test")
    assert d["status"] == "failed"
    [s] = _by_name(spans.get_finished_spans(), "vpp.actuator.setpoint")
    assert s.attributes["vpp.failed"] is True
    assert s.status.status_code.name == "ERROR"


async def test_forwarded_cluster_call_continues_the_trace(tmp_path, spans):
    from _v2g_helpers import tmp_session_factory

    from vpp.cluster import rpc
    from vpp.db.models import ClusterCallModel

    factory = tmp_session_factory(tmp_path / "trace-cluster.db")
    seen: list = []

    async def handler(payload, session):
        seen.append(payload)
        with tracing.span("vpp.test.handler"):
            pass
        return {"ok": True}

    rpc.register_handler("t-trace", "run", handler)
    # Created outside any span: the executor can only learn the caller's
    # trace from the call row, as a separate process would.
    executor = asyncio.create_task(rpc.run_executor(factory, lambda: ["t-trace"], poll_s=0.01))
    try:
        with tracing.span("vpp.test.request"):
            result = await rpc.call(
                "t-trace", "run", {"x": 1}, timeout_s=5, session_factory=factory
            )
    finally:
        executor.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await executor
        rpc._handlers.pop(("t-trace", "run"), None)

    assert result == {"ok": True}
    assert seen == [{"x": 1}]  # the trace context is stripped before the handler
    async with factory() as s:
        row = (await s.execute(select(ClusterCallModel))).scalar_one()
    stored = json.loads(row.payload_json)
    assert stored[rpc.TRACE_CONTEXT_KEY]["traceparent"].startswith("00-")

    finished = spans.get_finished_spans()
    [request] = _by_name(finished, "vpp.test.request")
    [call] = _by_name(finished, "vpp.cluster.call")
    [execute] = _by_name(finished, "vpp.cluster.execute")
    [inner] = _by_name(finished, "vpp.test.handler")
    assert call.parent.span_id == request.context.span_id
    assert execute.parent.span_id == call.context.span_id
    assert execute.context.trace_id == request.context.trace_id
    assert inner.parent.span_id == execute.context.span_id
    assert execute.kind.name == "SERVER" and call.kind.name == "CLIENT"
    assert execute.attributes["vpp.call_id"] == row.id == call.attributes["vpp.call_id"]


def test_log_records_carry_trace_ids(spans, tmp_path):
    from vpp.logging import configure_logging, get_logger

    log_file = tmp_path / "vpp.log"
    configure_logging(level="INFO", json_output=True, log_file=str(log_file))
    try:
        with tracing.span("vpp.test.logging"):
            ids = tracing.current_ids()
            logging.getLogger("vpp.test.stdlib").info("stdlib line")
            get_logger("vpp.test.structlog").info("structlog line")
        logging.getLogger("vpp.test.stdlib").info("outside")
    finally:
        configure_logging(level="INFO")  # drop the file handler
    lines = [json.loads(line) for line in log_file.read_text().splitlines()]
    by_event = {line["event"]: line for line in lines}
    for event in ("stdlib line", "structlog line"):
        assert by_event[event]["trace_id"] == ids[0]
        assert by_event[event]["span_id"] == ids[1]
    assert "trace_id" not in by_event["outside"]
