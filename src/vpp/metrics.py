"""Prometheus metrics for the VPP platform.

All metrics use the ``vpp_`` prefix and live on a private
:data:`REGISTRY` (not the ``prometheus_client`` global default registry),
so importing this module never collides with other libraries' metrics.

``prometheus_client`` is an optional dependency (the ``monitoring``
extra).  When it is not installed every helper in this module is a cheap
no-op and ``create_app`` simply does not mount ``/metrics``.

How metrics get populated
-------------------------
* **HTTP** -- :class:`vpp.api.middleware.PrometheusMiddleware` calls
  :func:`record_api_request` for every request, labelled by the *route
  template* (``/api/v1/resources/{resource_id}``), never the raw path, so
  a scanner hitting random URLs cannot blow up label cardinality.
* **EventBus** -- :func:`subscribe_event_bus` registers
  :func:`observe_event`, which maps platform events onto metrics:

  ===========================  ===========================================
  Event type                   Data keys read
  ===========================  ===========================================
  ``RESOURCE_UPDATED``         ``resource_id``; optional ``resource_type``,
                               power (first of ``current_power_kw``,
                               ``power_kw``, ``current_power``, ``power``),
                               ``soc`` (0-1, or 0-100 which is normalised)
  ``RESOURCE_ADDED``           ``resource_id``/``id``, ``resource_type``
  ``RESOURCE_REMOVED``         ``resource_id``/``id`` (drops its series)
  ``OPTIMIZATION_COMPLETED``   ``problem_type`` (default ``"dispatch"``),
  ``OPTIMIZATION_FAILED``      duration (first of ``duration_s``,
                               ``solve_time_s``, ``solve_time``)
  ``ORDER_SUBMITTED``/         ``market``, ``side``; ``status`` defaults to
  ``ORDER_FILLED``/            ``submitted``/``filled``/``cancelled``
  ``ORDER_CANCELLED``
  ``TRADE_EXECUTED``           ``market``, ``side``, ``quantity_mwh``/
                               ``quantity``, optional ``total_pnl``
                               (portfolio P&L, sets the P&L gauge)
  ``PROTOCOL_ERROR``           ``protocol`` (falls back to event source)
  ===========================  ===========================================

* **Direct calls** -- modules that do not publish events can call the
  module-level helpers (:func:`record_optimization`,
  :func:`time_optimization`, :func:`record_order`, :func:`record_trade`,
  :func:`set_trading_pnl`, :func:`record_alert_fired`, ...) directly.
  Do not do *both* for the same occurrence or it will be counted twice.

Cardinality note: per-resource gauges are labelled by ``resource_id``.
That is bounded by fleet size (resources are registered, not arbitrary
input) and removed again on ``RESOURCE_REMOVED``.
"""

from __future__ import annotations

import time
from contextlib import contextmanager, suppress
from typing import TYPE_CHECKING, Any

from vpp._version import __version__

if TYPE_CHECKING:
    from collections.abc import Iterator

    from vpp.events.bus import Event, EventBus

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        Info,
        generate_latest,
    )

    _HAS_PROMETHEUS = True
except ImportError:  # pragma: no cover - exercised via monkeypatch in tests
    _HAS_PROMETHEUS = False

# ---------------------------------------------------------------------------
# Metric instances (only created when prometheus_client is available)
# ---------------------------------------------------------------------------

if _HAS_PROMETHEUS:
    REGISTRY = CollectorRegistry(auto_describe=True)

    # -- Platform info
    VPP_INFO = Info("vpp", "VPP platform information", registry=REGISTRY)
    VPP_INFO.info({"version": __version__})

    # -- Resource metrics
    RESOURCE_COUNT = Gauge(
        "vpp_resource_count",
        "Number of registered resources",
        ["resource_type"],
        registry=REGISTRY,
    )
    RESOURCE_POWER = Gauge(
        "vpp_resource_power_kw",
        "Current power output (kW)",
        ["resource_id", "resource_type"],
        registry=REGISTRY,
    )
    BATTERY_SOC = Gauge(
        "vpp_battery_soc",
        "Battery state of charge (0-1)",
        ["resource_id"],
        registry=REGISTRY,
    )
    RESOURCE_LAST_UPDATE = Gauge(
        "vpp_resource_last_update_timestamp_seconds",
        "Unix time of the last telemetry update received for a resource",
        ["resource_id"],
        registry=REGISTRY,
    )
    TELEMETRY_EVENTS = Counter(
        "vpp_telemetry_events_total",
        "RESOURCE_UPDATED telemetry events observed on the event bus",
        ["source"],
        registry=REGISTRY,
    )

    # -- Optimization metrics
    OPTIMIZATION_DURATION = Histogram(
        "vpp_optimization_duration_seconds",
        "Optimization solve time",
        ["problem_type"],
        buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0],
        registry=REGISTRY,
    )
    OPTIMIZATION_TOTAL = Counter(
        "vpp_optimization_runs_total",
        "Total optimization runs",
        ["problem_type", "status"],
        registry=REGISTRY,
    )

    # -- Trading metrics
    TRADING_ORDERS = Counter(
        "vpp_trading_orders_total",
        "Total trading orders",
        ["market", "side", "status"],
        registry=REGISTRY,
    )
    TRADING_TRADES = Counter(
        "vpp_trading_trades_total",
        "Executed trades",
        ["market", "side"],
        registry=REGISTRY,
    )
    TRADING_VOLUME = Counter(
        "vpp_trading_volume_mwh_total",
        "Executed trade volume (MWh)",
        ["market", "side"],
        registry=REGISTRY,
    )
    TRADING_PNL = Gauge(
        "vpp_trading_pnl_total",
        "Total trading P&L",
        registry=REGISTRY,
    )

    # -- Protocol metrics
    PROTOCOL_MESSAGES = Counter(
        "vpp_protocol_messages_total",
        "Protocol messages sent/received",
        ["protocol", "direction"],
        registry=REGISTRY,
    )
    PROTOCOL_ERRORS = Counter(
        "vpp_protocol_errors_total",
        "Protocol errors",
        ["protocol"],
        registry=REGISTRY,
    )

    # -- V2G metrics
    V2G_FLEET_SOC = Gauge(
        "vpp_v2g_fleet_avg_soc",
        "V2G fleet average SOC",
        registry=REGISTRY,
    )
    V2G_DISPATCH_POWER = Gauge(
        "vpp_v2g_dispatch_power_kw",
        "Current V2G dispatch power (kW)",
        registry=REGISTRY,
    )
    V2G_CONNECTED_EVS = Gauge(
        "vpp_v2g_connected_evs",
        "Number of connected EVs",
        registry=REGISTRY,
    )

    # -- Alert metrics
    ALERTS_FIRED = Counter(
        "vpp_alerts_fired_total",
        "Alerts fired by the alert manager",
        ["severity", "rule"],
        registry=REGISTRY,
    )

    # -- API metrics
    API_REQUESTS = Counter(
        "vpp_api_requests_total",
        "API request count",
        ["method", "endpoint", "status"],
        registry=REGISTRY,
    )
    API_DURATION = Histogram(
        "vpp_api_request_duration_seconds",
        "API request duration",
        ["method", "endpoint"],
        buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 5.0],
        registry=REGISTRY,
    )
    API_IN_PROGRESS = Gauge(
        "vpp_api_requests_in_progress",
        "HTTP requests currently being served",
        registry=REGISTRY,
    )


def prometheus_available() -> bool:
    """True when ``prometheus_client`` is importable."""
    return _HAS_PROMETHEUS


# ---------------------------------------------------------------------------
# Metrics collector
# ---------------------------------------------------------------------------


class MetricsCollector:
    """Thin, no-op-safe facade over the module's Prometheus metrics."""

    def __init__(self) -> None:
        self._enabled = _HAS_PROMETHEUS
        # resource_id -> resource_type label used for its power series, so
        # the series can be removed on RESOURCE_REMOVED and so telemetry
        # events without a resource_type reuse the last known one.
        self._resource_types: dict[str, str] = {}

    @property
    def enabled(self) -> bool:
        return self._enabled

    # -- Optimization ---------------------------------------------------

    def record_optimization(
        self,
        problem_type: str,
        status: str,
        duration_seconds: float | None = None,
    ) -> None:
        if not self._enabled:
            return
        if duration_seconds is not None:
            OPTIMIZATION_DURATION.labels(problem_type=problem_type).observe(duration_seconds)
        OPTIMIZATION_TOTAL.labels(problem_type=problem_type, status=status).inc()

    # -- Trading --------------------------------------------------------

    def record_order(self, market: str, side: str, status: str) -> None:
        if not self._enabled:
            return
        TRADING_ORDERS.labels(market=market, side=side, status=status).inc()

    def record_trade(self, market: str, side: str, quantity_mwh: float = 0.0) -> None:
        if not self._enabled:
            return
        TRADING_TRADES.labels(market=market, side=side).inc()
        if quantity_mwh:
            TRADING_VOLUME.labels(market=market, side=side).inc(abs(quantity_mwh))

    def set_trading_pnl(self, pnl: float) -> None:
        if not self._enabled:
            return
        TRADING_PNL.set(pnl)

    # -- Protocols ------------------------------------------------------

    def record_protocol_message(self, protocol: str, direction: str) -> None:
        if not self._enabled:
            return
        PROTOCOL_MESSAGES.labels(protocol=protocol, direction=direction).inc()

    def record_protocol_error(self, protocol: str) -> None:
        if not self._enabled:
            return
        PROTOCOL_ERRORS.labels(protocol=protocol).inc()

    # -- Resources ------------------------------------------------------

    def set_resource_power(self, resource_id: str, resource_type: str, power_kw: float) -> None:
        if not self._enabled:
            return
        previous = self._resource_types.get(resource_id)
        if previous is not None and previous != resource_type:
            _safe_remove(RESOURCE_POWER, resource_id, previous)
        self._resource_types[resource_id] = resource_type
        RESOURCE_POWER.labels(resource_id=resource_id, resource_type=resource_type).set(power_kw)

    def set_battery_soc(self, resource_id: str, soc: float) -> None:
        if not self._enabled:
            return
        BATTERY_SOC.labels(resource_id=resource_id).set(soc)

    def set_resource_count(self, resource_type: str, count: int) -> None:
        if not self._enabled:
            return
        RESOURCE_COUNT.labels(resource_type=resource_type).set(count)

    def mark_resource_updated(
        self, resource_id: str, source: str, ts: float | None = None
    ) -> None:
        if not self._enabled:
            return
        RESOURCE_LAST_UPDATE.labels(resource_id=resource_id).set(
            ts if ts is not None else time.time()
        )
        TELEMETRY_EVENTS.labels(source=source or "unknown").inc()

    def forget_resource(self, resource_id: str) -> None:
        """Drop every per-resource series for a removed resource."""
        if not self._enabled:
            return
        resource_type = self._resource_types.pop(resource_id, None)
        if resource_type is not None:
            _safe_remove(RESOURCE_POWER, resource_id, resource_type)
        _safe_remove(BATTERY_SOC, resource_id)
        _safe_remove(RESOURCE_LAST_UPDATE, resource_id)

    def known_resource_type(self, resource_id: str) -> str | None:
        return self._resource_types.get(resource_id)

    def remember_resource_type(self, resource_id: str, resource_type: str) -> None:
        self._resource_types.setdefault(resource_id, resource_type)

    # -- V2G ------------------------------------------------------------

    def set_v2g_fleet_metrics(self, avg_soc: float, connected: int, dispatch_kw: float) -> None:
        if not self._enabled:
            return
        V2G_FLEET_SOC.set(avg_soc)
        V2G_CONNECTED_EVS.set(connected)
        V2G_DISPATCH_POWER.set(dispatch_kw)

    # -- Alerts ---------------------------------------------------------

    def record_alert_fired(self, severity: str, rule: str) -> None:
        if not self._enabled:
            return
        ALERTS_FIRED.labels(severity=severity, rule=rule).inc()

    # -- HTTP -----------------------------------------------------------

    def record_api_request(self, method: str, endpoint: str, status: int, duration: float) -> None:
        if not self._enabled:
            return
        API_REQUESTS.labels(method=method, endpoint=endpoint, status=str(status)).inc()
        API_DURATION.labels(method=method, endpoint=endpoint).observe(duration)

    def request_started(self) -> None:
        if self._enabled:
            API_IN_PROGRESS.inc()

    def request_finished(self) -> None:
        if self._enabled:
            API_IN_PROGRESS.dec()

    # -- Exposition -----------------------------------------------------

    def get_metrics_text(self) -> bytes:
        """Generate Prometheus text exposition format."""
        if not self._enabled:
            return b"# prometheus_client not installed\n"
        return generate_latest(REGISTRY)

    def get_content_type(self) -> str:
        if not self._enabled:
            return "text/plain"
        return CONTENT_TYPE_LATEST


def _safe_remove(metric: Any, *labels: str) -> None:
    with suppress(KeyError):
        metric.remove(*labels)


# Module-level singleton
metrics_collector = MetricsCollector()


# ---------------------------------------------------------------------------
# Module-level helpers (stable hooks for other modules)
# ---------------------------------------------------------------------------


def record_optimization(
    problem_type: str,
    status: str,
    duration_seconds: float | None = None,
) -> None:
    """Count one optimization run (and observe its duration if given)."""
    metrics_collector.record_optimization(problem_type, status, duration_seconds)


@contextmanager
def time_optimization(problem_type: str) -> Iterator[None]:
    """Time a block as one optimization run.

    Records ``status="success"`` on normal exit and ``status="error"`` if the
    block raises (the exception propagates)::

        with time_optimization("dispatch"):
            result = optimizer.solve(problem)
    """
    start = time.perf_counter()
    try:
        yield
    except BaseException:
        record_optimization(problem_type, "error", time.perf_counter() - start)
        raise
    record_optimization(problem_type, "success", time.perf_counter() - start)


def record_order(market: str, side: str, status: str) -> None:
    metrics_collector.record_order(market, side, status)


def record_trade(market: str, side: str, quantity_mwh: float = 0.0) -> None:
    metrics_collector.record_trade(market, side, quantity_mwh)


def set_trading_pnl(pnl: float) -> None:
    metrics_collector.set_trading_pnl(pnl)


def record_protocol_error(protocol: str) -> None:
    metrics_collector.record_protocol_error(protocol)


def record_alert_fired(severity: str, rule: str) -> None:
    metrics_collector.record_alert_fired(severity, rule)


# ---------------------------------------------------------------------------
# EventBus integration
# ---------------------------------------------------------------------------

_POWER_KEYS = ("current_power_kw", "power_kw", "current_power", "power")
_DURATION_KEYS = ("duration_s", "solve_time_s", "solve_time", "duration_seconds")


def _first_float(data: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = data.get(key)
        if value is None or isinstance(value, bool):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def normalise_soc(value: float) -> float:
    """Map SOC reported as a percentage (0-100) onto the 0-1 scale."""
    return value / 100.0 if value > 1.0 else value


def observe_event(event: Event, collector: MetricsCollector | None = None) -> None:
    """Update metrics from one EventBus event (see module docstring)."""
    from vpp.events.bus import EventType

    c = collector or metrics_collector
    if not c.enabled:
        return
    data = event.data or {}
    et = event.event_type

    if et == EventType.RESOURCE_UPDATED:
        resource_id = data.get("resource_id") or data.get("id")
        if not resource_id:
            return
        resource_id = str(resource_id)
        resource_type = (
            data.get("resource_type") or c.known_resource_type(resource_id) or "unknown"
        )
        power = _first_float(data, _POWER_KEYS)
        if power is not None:
            c.set_resource_power(resource_id, str(resource_type), power)
        soc = _first_float(data, ("soc",))
        if soc is not None:
            c.set_battery_soc(resource_id, normalise_soc(soc))
        c.mark_resource_updated(resource_id, event.source, event.timestamp)
    elif et == EventType.RESOURCE_ADDED:
        resource_id = data.get("resource_id") or data.get("id")
        if resource_id and data.get("resource_type"):
            c.remember_resource_type(str(resource_id), str(data["resource_type"]))
    elif et == EventType.RESOURCE_REMOVED:
        resource_id = data.get("resource_id") or data.get("id")
        if resource_id:
            c.forget_resource(str(resource_id))
    elif et in (EventType.OPTIMIZATION_COMPLETED, EventType.OPTIMIZATION_FAILED):
        status = "success" if et == EventType.OPTIMIZATION_COMPLETED else "error"
        c.record_optimization(
            str(data.get("problem_type") or "dispatch"),
            str(data.get("status") or status),
            _first_float(data, _DURATION_KEYS),
        )
    elif et in (EventType.ORDER_SUBMITTED, EventType.ORDER_FILLED, EventType.ORDER_CANCELLED):
        default_status = {
            EventType.ORDER_SUBMITTED: "submitted",
            EventType.ORDER_FILLED: "filled",
            EventType.ORDER_CANCELLED: "cancelled",
        }[et]
        c.record_order(
            str(data.get("market") or "unknown"),
            str(data.get("side") or "unknown"),
            str(data.get("status") or default_status),
        )
    elif et == EventType.TRADE_EXECUTED:
        c.record_trade(
            str(data.get("market") or "unknown"),
            str(data.get("side") or "unknown"),
            _first_float(data, ("quantity_mwh", "quantity")) or 0.0,
        )
        pnl = _first_float(data, ("total_pnl", "portfolio_pnl"))
        if pnl is not None:
            c.set_trading_pnl(pnl)
    elif et == EventType.PROTOCOL_ERROR:
        c.record_protocol_error(str(data.get("protocol") or event.source or "unknown"))


def subscribe_event_bus(bus: EventBus, collector: MetricsCollector | None = None) -> str:
    """Subscribe :func:`observe_event` to *bus*; returns the subscription id."""
    from vpp.events.bus import EventType

    async def _on_event(event: Event) -> None:
        observe_event(event, collector)

    return bus.subscribe(
        _on_event,
        event_types={
            EventType.RESOURCE_ADDED,
            EventType.RESOURCE_REMOVED,
            EventType.RESOURCE_UPDATED,
            EventType.OPTIMIZATION_COMPLETED,
            EventType.OPTIMIZATION_FAILED,
            EventType.ORDER_SUBMITTED,
            EventType.ORDER_FILLED,
            EventType.ORDER_CANCELLED,
            EventType.TRADE_EXECUTED,
            EventType.PROTOCOL_ERROR,
        },
    )
