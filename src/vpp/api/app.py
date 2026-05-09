"""FastAPI application factory."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select

from vpp.settings import get_settings
from vpp.db.engine import init_db, close_db, get_session_factory

logger = logging.getLogger(__name__)


async def _placeholder_fetch_telemetry(battery_id: str):
    """Placeholder telemetry fetcher used by the periodic loop.

    Returns the most recent SOC observation as a single-point window so the
    updater can roll forward without crashing when no fresh telemetry has
    arrived.  Returns ``None`` for batteries that have never reported.

    TODO M5: real telemetry ingestion via MQTT subscriber writes to a
    ``battery_soc_telemetry`` table; ``fetch_telemetry`` then pulls the
    rows newer than ``last_degradation_update`` and assembles a window.
    The current placeholder is a no-op for production but lets the loop
    exercise its happy-path bookkeeping in development.
    """
    from vpp.db.models import BatteryStateModel
    from vpp.degradation.telemetry import TelemetryWindow

    try:
        factory = get_session_factory()
    except RuntimeError:
        return None

    async with factory() as session:
        stmt = (
            select(BatteryStateModel)
            .where(BatteryStateModel.resource_id == battery_id)
            .order_by(BatteryStateModel.timestamp.desc())
            .limit(2)
        )
        rows = list((await session.execute(stmt)).scalars().all())

    if len(rows) < 2:
        return None
    rows = list(reversed(rows))  # chronological order
    return TelemetryWindow(
        battery_id=battery_id,
        soc_trace=[r.soc / 100.0 if r.soc > 1.0 else r.soc for r in rows],
        timestamps=[r.timestamp or datetime.now(timezone.utc) for r in rows],
        temperatures_c=[r.temperature for r in rows],
    )


async def _degradation_periodic_loop(interval_minutes: int) -> None:
    """Long-running task that refreshes battery SOH on a fixed cadence.

    Iterates over all currently-registered battery resources every
    ``interval_minutes`` minutes, asks ``_placeholder_fetch_telemetry`` for
    a SOC window, and pushes the result through ``DegradationUpdater``.
    Exceptions on a single battery are logged and do not break the loop.
    """
    from vpp.db.models import ResourceModel
    from vpp.degradation.telemetry import DegradationUpdater

    factory = get_session_factory()
    updater = DegradationUpdater(session_factory=factory)

    while True:
        try:
            async with factory() as session:
                rows = list(
                    (
                        await session.execute(
                            select(ResourceModel.id).where(
                                ResourceModel.resource_type == "battery"
                            )
                        )
                    ).scalars().all()
                )
            for bid in rows:
                try:
                    window = await _placeholder_fetch_telemetry(bid)
                    if window is not None:
                        await updater.apply_window(window)
                except Exception:
                    logger.exception("Degradation tick failed for battery %s", bid)
        except Exception:
            logger.exception("Degradation periodic loop iteration failed")

        await asyncio.sleep(max(1, interval_minutes) * 60)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup / shutdown lifecycle."""
    settings = get_settings()
    await init_db(
        settings.database_url,
        echo=settings.debug,
        use_alembic=settings.use_alembic,
    )

    task: Optional[asyncio.Task] = None
    if settings.degradation_updater_enabled:
        task = asyncio.create_task(
            _degradation_periodic_loop(
                settings.degradation_updater_interval_minutes
            ),
            name="vpp-degradation-updater",
        )
        app.state.degradation_task = task
    else:
        app.state.degradation_task = None

    try:
        yield
    finally:
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("Degradation updater raised during shutdown")
        await close_db()


def create_app() -> FastAPI:
    """Build and return the configured FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Virtual Power Plant Platform",
        description=(
            "Production-ready API for managing distributed energy resources, "
            "optimization dispatch, multi-market trading, and grid protocol integration."
        ),
        version="2.0.0",
        lifespan=_lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # -- Middleware ----------------------------------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -- Routes -------------------------------------------------------------
    from .routes import (
        health, resources, optimization, trading, auth, config, protocols, v2g,
        degradation,
    )

    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(resources.router)
    app.include_router(optimization.router)
    app.include_router(trading.router)
    app.include_router(config.router)
    app.include_router(protocols.router)
    app.include_router(v2g.router)
    app.include_router(degradation.router)

    # -- WebSocket ----------------------------------------------------------
    from .websocket import websocket_endpoint

    app.add_api_websocket_route("/ws", websocket_endpoint)

    return app
