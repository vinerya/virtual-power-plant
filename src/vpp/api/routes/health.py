"""Health-check and readiness probes."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import text

import vpp as _vpp
from vpp.db.engine import get_session_factory

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health() -> dict:
    """Basic liveness probe."""
    return {"status": "ok"}


@router.get("/ready", response_model=None)
async def readiness() -> dict | JSONResponse:
    """Readiness probe: 200 only when the database answers a trivial query.

    Returns 503 otherwise, so orchestrators stop routing traffic to an
    instance whose database is unreachable.
    """
    try:
        async with get_session_factory()() as session:
            await session.execute(text("SELECT 1"))
    except Exception as exc:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "subsystems": {"database": False},
                "error": type(exc).__name__,
            },
        )
    return {"status": "ready", "subsystems": {"database": True}}


@router.get("/version")
async def version() -> dict:
    """Return platform and library version info."""
    return {
        "platform": "Virtual Power Plant",
        "version": _vpp.__version__,
        "api_version": "v1",
    }
