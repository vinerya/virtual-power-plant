"""App wiring for logging, metrics and alerting.

Kept out of :mod:`vpp.api.app` so the factory only needs two calls:

* :func:`install_observability` from ``create_app`` -- logging config,
  request-id + Prometheus middleware, ``/metrics`` and alerts routes, and
  the sites' alert-count provider.
* :func:`start_observability` / :func:`stop_observability` from the
  lifespan -- EventBus subscriptions and the alert evaluator task.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vpp import metrics as vpp_metrics

if TYPE_CHECKING:
    from fastapi import FastAPI

    from vpp.alert_service import AlertService
    from vpp.events.bus import EventBus
    from vpp.settings import Settings

logger = logging.getLogger(__name__)


def metrics_endpoint_enabled(settings: Settings) -> bool:
    return bool(settings.metrics_enabled) and vpp_metrics.prometheus_available()


def install_observability(app: FastAPI, settings: Settings) -> None:
    """Configure logging, add middleware and mount observability routes.

    Call after all other middleware has been added: the last-added
    middleware is outermost, so request ids and metrics cover every
    response, including rate-limit rejections.
    """
    from vpp.api.middleware import PrometheusMiddleware, RequestIdMiddleware
    from vpp.api.routes import alerts as alerts_routes
    from vpp.logging import configure_logging

    json_output = settings.is_production if settings.log_json is None else settings.log_json
    configure_logging(level=settings.log_level, json_output=json_output)

    if metrics_endpoint_enabled(settings):
        from vpp.api.routes import metrics as metrics_routes

        app.state.metrics_bearer_token = settings.metrics_bearer_token
        app.add_middleware(PrometheusMiddleware)
        app.include_router(metrics_routes.router)
    elif settings.metrics_enabled:
        logger.warning(
            "VPP_METRICS_ENABLED is set but prometheus_client is not installed; "
            "/metrics is not mounted (pip install 'vpp[monitoring]')."
        )
    app.add_middleware(RequestIdMiddleware, access_log=settings.access_log_enabled)

    app.include_router(alerts_routes.router)

    # Sites report ``active_alerts`` from the persisted alert store.
    from vpp.alert_service import AlertRepository
    from vpp.portal.sites import register_alert_count_provider

    register_alert_count_provider(AlertRepository.count_open_by_source)


@dataclass
class ObservabilityHandle:
    bus: EventBus
    metrics_subscription: str | None = None
    alert_service: AlertService | None = None


async def start_observability(settings: Settings, bus: EventBus) -> ObservabilityHandle:
    """Subscribe metrics to the bus and start the alert evaluator."""
    from vpp.alert_service import AlertService, set_alert_service
    from vpp.alerts import WebhookAlertChannel
    from vpp.api.websocket import manager as websocket_manager
    from vpp.db.engine import get_session_factory

    handle = ObservabilityHandle(bus=bus)
    if metrics_endpoint_enabled(settings):
        handle.metrics_subscription = vpp_metrics.subscribe_event_bus(bus)

    if settings.alerts_enabled:
        service = AlertService(get_session_factory(), broadcaster=websocket_manager)
        if settings.alert_webhook_url:
            service.manager.add_channel(
                WebhookAlertChannel(
                    settings.alert_webhook_url,
                    secret=settings.alert_webhook_secret,
                    timeout_s=settings.alert_webhook_timeout_seconds,
                    max_retries=settings.alert_webhook_max_retries,
                )
            )
        try:
            await service.start(bus, seed_defaults=settings.alerts_seed_default_rules)
        except Exception:
            logger.exception("Alert service failed to start; alerts will not be evaluated")
        else:
            handle.alert_service = service
            set_alert_service(service)
    return handle


async def stop_observability(handle: ObservabilityHandle) -> None:
    from vpp.alert_service import get_alert_service, set_alert_service

    if handle.metrics_subscription is not None:
        handle.bus.unsubscribe(handle.metrics_subscription)
    if handle.alert_service is not None:
        await handle.alert_service.stop()
        if get_alert_service() is handle.alert_service:
            set_alert_service(None)
