"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from vpp.settings import get_settings
from vpp.db.engine import init_db, close_db


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup / shutdown lifecycle."""
    settings = get_settings()
    await init_db(settings.database_url, echo=settings.debug)
    yield
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
    from .routes import health, resources, optimization, trading, auth, config, protocols, v2g

    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(resources.router)
    app.include_router(optimization.router)
    app.include_router(trading.router)
    app.include_router(config.router)
    app.include_router(protocols.router)
    app.include_router(v2g.router)

    # -- WebSocket ----------------------------------------------------------
    from .websocket import websocket_endpoint

    app.add_api_websocket_route("/ws", websocket_endpoint)

    return app
