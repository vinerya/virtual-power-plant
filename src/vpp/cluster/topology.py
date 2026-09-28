"""Deployment topology: which work is leader-only, what is per-process, what is refused."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Lease names. Each guards work that must run once per deployment.
LEASE_TRADING = "trading-venue"
LEASE_DEGRADATION = "degradation-updater"
LEASE_ALERTS = "alert-evaluator"
LEASE_MQTT = "mqtt-ingestion"
LEASE_MODBUS = "modbus-ingestion"
LEASE_PROTOCOLS = "protocol-adapters"

LEADER_ONLY = {
    LEASE_TRADING: "simulated trading venue (order matching, market-data tick); "
    "other workers forward trading commands to it",
    LEASE_DEGRADATION: "battery degradation updater",
    LEASE_ALERTS: "alert evaluation; other workers forward telemetry to it",
    LEASE_MQTT: "MQTT telemetry ingestion",
    LEASE_MODBUS: "Modbus telemetry ingestion",
    LEASE_PROTOCOLS: "protocol adapters (OCPP / OpenADR / IEEE 2030.5) and the DR orchestrator",
}

PER_PROCESS = (
    "WebSocket broadcasts reach other workers' clients through the database relay "
    "(cluster_events), about one VPP_CLUSTER_POLL_INTERVAL_SECONDS later",
)


class TopologyError(ValueError):
    """The configured topology cannot work."""


def validate_topology(settings: Any) -> None:
    """Refuse combinations that would silently misbehave with several workers."""
    workers = int(getattr(settings, "api_workers", 1) or 1)
    if workers > 1 and getattr(settings, "ocpp_enabled", False):
        raise TopologyError(
            "VPP_OCPP_ENABLED requires VPP_API_WORKERS=1: charge-point WebSocket sessions "
            "and the Central System's state live in one process, so commands and reads "
            "handled by another worker could not reach them. Run a single worker (or a "
            "separate single-worker deployment for OCPP)."
        )
    if workers > 1 and getattr(settings, "control_enabled", False):
        raise TopologyError(
            "VPP_CONTROL_ENABLED requires VPP_API_WORKERS=1: each device's active "
            "setpoint, rate limiter and expiry watchdog live in the process that wrote "
            "it, so two workers could issue conflicting commands to the same device. "
            "Run a single worker when writing setpoints to devices."
        )


def log_topology(settings: Any, leadership: dict[str, bool]) -> None:
    """Log the effective topology once at startup."""
    from vpp.cluster.node import node_id

    workers = int(getattr(settings, "api_workers", 1) or 1)
    held = sorted(n for n, lead in leadership.items() if lead)
    elsewhere = sorted(n for n, lead in leadership.items() if not lead)
    logger.info(
        "API topology: VPP_API_WORKERS=%d, process %s; leader-only subsystems: %s; "
        "leases held here: %s; held elsewhere: %s",
        workers,
        node_id(),
        ", ".join(f"{k} ({v})" for k, v in LEADER_ONLY.items()),
        ", ".join(held) or "none",
        ", ".join(elsewhere) or "none",
    )
    if workers > 1:
        for item in PER_PROCESS:
            logger.warning("Multi-worker deployment: %s", item)
        from vpp.auth.shared_limits import MEMORY, resolve_backend

        if resolve_backend(settings) == MEMORY:
            logger.warning(
                "Multi-worker deployment with VPP_RATE_LIMIT_BACKEND=memory: each worker "
                "counts HTTP requests and failed logins separately, so a client gets up to "
                "workers x VPP_RATE_LIMIT_REQUESTS_PER_MINUTE and a password guesser "
                "workers x VPP_LOGIN_MAX_FAILURES attempts per lockout window"
            )
        else:
            logger.info(
                "HTTP rate limit and login throttle are shared by all workers "
                "(table shared_rate_limits)"
            )
        if getattr(settings, "openadr_enabled", False) or getattr(
            settings, "ieee2030_5_enabled", False
        ):
            logger.warning(
                "Multi-worker deployment: OpenADR / IEEE 2030.5 run on the protocol-adapters "
                "lease holder only; their live views (/api/v1/protocols/openadr, "
                "/ieee2030_5, /api/v1/dr/status) answer 404 on other workers."
            )
        if getattr(settings, "database_is_sqlite", False):
            logger.warning(
                "Multi-worker deployment on SQLite: concurrent writers serialise on the "
                "database file; use PostgreSQL for more than one worker."
            )
