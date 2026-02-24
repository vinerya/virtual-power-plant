"""Prometheus metrics for the VPP platform.

All metrics use the ``vpp_`` prefix.  A ``MetricsCollector`` class
periodically samples VPP state and updates gauges / counters.
The ``/metrics`` endpoint is mounted by the API app.
"""

from __future__ import annotations

import time
from typing import Any

try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        Info,
        CollectorRegistry,
        generate_latest,
        CONTENT_TYPE_LATEST,
    )

    _HAS_PROMETHEUS = True
except ImportError:
    _HAS_PROMETHEUS = False

# ---------------------------------------------------------------------------
# Metric instances (only created when prometheus_client is available)
# ---------------------------------------------------------------------------

if _HAS_PROMETHEUS:
    REGISTRY = CollectorRegistry(auto_describe=True)

    # -- Platform info
    VPP_INFO = Info("vpp", "VPP platform information", registry=REGISTRY)
    VPP_INFO.info({"version": "2.0.0"})

    # -- Resource metrics
    RESOURCE_COUNT = Gauge(
        "vpp_resource_count", "Number of registered resources",
        ["resource_type"], registry=REGISTRY,
    )
    RESOURCE_POWER = Gauge(
        "vpp_resource_power_kw", "Current power output (kW)",
        ["resource_id", "resource_type"], registry=REGISTRY,
    )
    BATTERY_SOC = Gauge(
        "vpp_battery_soc", "Battery state of charge (0-1)",
        ["resource_id"], registry=REGISTRY,
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


# ---------------------------------------------------------------------------
# Metrics collector
# ---------------------------------------------------------------------------

class MetricsCollector:
    """Periodically samples VPP state and updates Prometheus metrics."""

    def __init__(self) -> None:
        self._enabled = _HAS_PROMETHEUS

    @property
    def enabled(self) -> bool:
        return self._enabled

    def record_optimization(
        self, problem_type: str, status: str, duration_seconds: float,
    ) -> None:
        if not self._enabled:
            return
        OPTIMIZATION_DURATION.labels(problem_type=problem_type).observe(duration_seconds)
        OPTIMIZATION_TOTAL.labels(problem_type=problem_type, status=status).inc()

    def record_order(self, market: str, side: str, status: str) -> None:
        if not self._enabled:
            return
        TRADING_ORDERS.labels(market=market, side=side, status=status).inc()

    def record_protocol_message(self, protocol: str, direction: str) -> None:
        if not self._enabled:
            return
        PROTOCOL_MESSAGES.labels(protocol=protocol, direction=direction).inc()

    def record_protocol_error(self, protocol: str) -> None:
        if not self._enabled:
            return
        PROTOCOL_ERRORS.labels(protocol=protocol).inc()

    def set_resource_power(self, resource_id: str, resource_type: str, power_kw: float) -> None:
        if not self._enabled:
            return
        RESOURCE_POWER.labels(resource_id=resource_id, resource_type=resource_type).set(power_kw)

    def set_battery_soc(self, resource_id: str, soc: float) -> None:
        if not self._enabled:
            return
        BATTERY_SOC.labels(resource_id=resource_id).set(soc)

    def set_v2g_fleet_metrics(self, avg_soc: float, connected: int, dispatch_kw: float) -> None:
        if not self._enabled:
            return
        V2G_FLEET_SOC.set(avg_soc)
        V2G_CONNECTED_EVS.set(connected)
        V2G_DISPATCH_POWER.set(dispatch_kw)

    def record_api_request(self, method: str, endpoint: str, status: int, duration: float) -> None:
        if not self._enabled:
            return
        API_REQUESTS.labels(method=method, endpoint=endpoint, status=str(status)).inc()
        API_DURATION.labels(method=method, endpoint=endpoint).observe(duration)

    def get_metrics_text(self) -> bytes:
        """Generate Prometheus text exposition format."""
        if not self._enabled:
            return b"# prometheus_client not installed\n"
        return generate_latest(REGISTRY)

    def get_content_type(self) -> str:
        if not self._enabled:
            return "text/plain"
        return CONTENT_TYPE_LATEST


# Module-level singleton
metrics_collector = MetricsCollector()
