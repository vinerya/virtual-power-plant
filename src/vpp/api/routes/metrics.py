"""Prometheus scrape endpoint.

``GET /metrics`` serves the text exposition format of :data:`vpp.metrics.REGISTRY`.
It is mounted by :func:`vpp.api.app.create_app` only when
``VPP_METRICS_ENABLED`` is true *and* ``prometheus_client`` is installed.

Access control
--------------
Unauthenticated by default, which is the Prometheus convention: the endpoint
exposes only aggregate operational numbers (no credentials, no PII), and is
normally reachable only on the internal network.  To restrict it, set
``VPP_METRICS_BEARER_TOKEN``; scrapes must then send
``Authorization: Bearer <token>`` (Prometheus: ``authorization.credentials``
in the scrape config).  Note that per-resource series carry resource ids.
"""

from __future__ import annotations

import hmac

from fastapi import APIRouter, HTTPException, Request, Response, status

from vpp.metrics import metrics_collector

router = APIRouter(tags=["Metrics"])


@router.get("/metrics", include_in_schema=False)
async def metrics(request: Request) -> Response:
    # Captured from the Settings passed to create_app (install_observability).
    token = getattr(request.app.state, "metrics_bearer_token", None)
    if token:
        header = request.headers.get("authorization", "")
        scheme, _, supplied = header.partition(" ")
        valid = hmac.compare_digest(supplied.strip().encode(), token.encode())
        if scheme.lower() != "bearer" or not valid:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing metrics bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )
    return Response(
        content=metrics_collector.get_metrics_text(),
        media_type=metrics_collector.get_content_type(),
    )
