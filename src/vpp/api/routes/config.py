"""Configuration management routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from vpp.auth.security import get_current_user, require_role
from vpp.db.models import UserModel
from vpp.schemas.auth import UserRole
from vpp.settings import get_settings

router = APIRouter(prefix="/api/v1/config", tags=["Configuration"])


@router.get("/")
async def get_config(_user: UserModel = Depends(get_current_user)):
    """Return current VPP platform configuration (non-secret fields)."""
    settings = get_settings()
    return {
        "env": settings.env,
        "log_level": settings.log_level,
        "api_host": settings.api_host,
        "api_port": settings.api_port,
        "database_backend": "postgresql" if "postgresql" in settings.database_url else "sqlite",
        "metrics_enabled": settings.metrics_enabled,
        "default_timezone": settings.default_timezone,
    }


@router.post("/validate")
async def validate_config(
    body: dict,
    _user: UserModel = Depends(get_current_user),
):
    """Validate a configuration payload without applying it."""
    errors: list[str] = []
    warnings: list[str] = []

    if "optimization" in body:
        opt = body["optimization"]
        if opt.get("time_horizon", 24) > 168:
            warnings.append("time_horizon > 168h may be slow")

    return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}
