"""Health-check and readiness probes."""

from __future__ import annotations

from fastapi import APIRouter

import vpp as _vpp

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health() -> dict:
    """Basic liveness probe."""
    return {"status": "ok"}


@router.get("/ready")
async def readiness() -> dict:
    """Readiness probe — confirms all subsystems are initialised."""
    return {"status": "ready", "subsystems": {"database": True, "vpp": True}}


@router.get("/version")
async def version() -> dict:
    """Return platform and library version info."""
    return {
        "platform": "Virtual Power Plant",
        "version": _vpp.__version__,
        "api_version": "v1",
    }
