"""Configuration management routes.

API surface
-----------
- GET  /api/v1/config           live config document (YAML) + platform settings
- GET  /api/v1/config/schema    JSON Schema of the VPP config document
- PUT  /api/v1/config           validate + persist + apply a new document (admin)
- POST /api/v1/config/validate  validate a JSON document without applying it

Persistence model
-----------------
Applied documents are stored append-only in ``config_documents`` with a
monotonically increasing ``version``; the newest row is the live config and
older rows form the audit trail (who applied what, when). The YAML text is
stored verbatim so operator comments survive a round-trip. Until the first
``PUT``, ``GET`` serves the in-process defaults (``version`` = 0,
``updated_at`` = null).

Applying a document replaces the live ``VPPConfig`` held in
:mod:`vpp.api.deps`; on startup :func:`apply_stored_config` (called from the
API lifespan) re-applies the newest stored document. Settings that are
read from the environment at process start (``VPP_*`` variables: database,
auth, rate limiting, ingestion toggles) are *not* part of this document and
are reported read-only alongside it.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime  # noqa: TC003 -- pydantic needs it at runtime
from typing import TYPE_CHECKING, Any

import yaml
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncSession,  # noqa: TC002 -- FastAPI resolves dependency annotations at runtime
)

from vpp.api.deps import get_live_config, set_live_config
from vpp.auth.security import get_current_user, require_role
from vpp.config import VPPConfig
from vpp.config.schema import validate_config_mapping, vpp_config_json_schema
from vpp.db.engine import get_db
from vpp.db.models import ConfigDocumentModel, UserModel
from vpp.schemas.auth import UserRole
from vpp.settings import get_settings

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import async_sessionmaker

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/config", tags=["Configuration"])

#: Upper bound on an applied document; the full default config is ~1.5 KB.
MAX_CONFIG_BYTES = 256 * 1024


class ConfigApplyRequest(BaseModel):
    yaml: str = Field(..., description="Full VPP configuration document as YAML")
    base_hash: str | None = Field(
        None,
        description=(
            "Optional optimistic-concurrency guard: the ``hash`` of the document "
            "the edit was based on. If the live document has changed since, the "
            "request is rejected with 409 instead of overwriting it."
        ),
    )


class ConfigValidationErrorDTO(BaseModel):
    path: str
    message: str


class ConfigDocumentResponse(BaseModel):
    yaml: str
    hash: str
    version: int = Field(description="0 = built-in defaults, never applied")
    updated_at: datetime | None = None
    updated_by: str | None = None
    warnings: list[str] = Field(default_factory=list)
    # Read-only process settings (environment-driven, not editable here).
    env: str
    log_level: str
    api_host: str
    api_port: int
    database_backend: str
    metrics_enabled: bool
    default_timezone: str


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _default_yaml() -> str:
    return yaml.safe_dump(get_live_config().to_dict(), sort_keys=False)


async def _latest(session: AsyncSession) -> ConfigDocumentModel | None:
    result = await session.execute(
        select(ConfigDocumentModel).order_by(ConfigDocumentModel.version.desc()).limit(1)
    )
    return result.scalar_one_or_none()


def _response(row: ConfigDocumentModel | None, *, warnings: list[str] | None = None) -> dict:
    settings = get_settings()
    if row is None:
        text = _default_yaml()
        doc: dict[str, Any] = {
            "yaml": text,
            "hash": _sha256(text),
            "version": 0,
            "updated_at": None,
            "updated_by": None,
        }
    else:
        doc = {
            "yaml": row.yaml,
            "hash": row.hash,
            "version": row.version,
            "updated_at": row.created_at,
            "updated_by": row.updated_by,
        }
    doc.update(
        warnings=warnings or [],
        env=settings.env,
        log_level=settings.log_level,
        api_host=settings.api_host,
        api_port=settings.api_port,
        database_backend="postgresql" if "postgresql" in settings.database_url else "sqlite",
        metrics_enabled=settings.metrics_enabled,
        default_timezone=settings.default_timezone,
    )
    return doc


def validate_config_document(
    data: Any,
) -> tuple[VPPConfig | None, list[dict[str, str]], list[str]]:
    """Full validation pipeline: structure (schema) then semantics (``validate()``).

    Returns ``(config, errors, warnings)``; ``config`` is ``None`` iff there
    are errors.
    """
    document, errors = validate_config_mapping(data)
    if document is None:
        return None, errors, []
    if document.monitoring.log_file is not None:
        # VPPConfig opens a FileHandler at construction time; letting an API
        # caller choose an arbitrary server-side path is a file-write primitive.
        return (
            None,
            [
                {
                    "path": "/monitoring/log_file",
                    "message": "log_file cannot be set through the API; configure log "
                    "destinations in the deployment environment",
                }
            ],
            [],
        )
    config = VPPConfig.from_dict(document.model_dump())
    result = config.validate()
    if not result.is_valid:
        return None, [{"path": "$", "message": m} for m in result.errors], list(result.warnings)
    return config, [], list(result.warnings)


async def apply_stored_config(session_factory: async_sessionmaker[AsyncSession]) -> int | None:
    """Apply the newest stored config document on boot; return its version.

    Never raises: an empty or missing ``config_documents`` table (fresh
    database, migrations not yet run) keeps the built-in defaults, and a
    stored document that no longer validates (e.g. after a schema change)
    is logged and skipped rather than blocking startup.
    """
    try:
        async with session_factory() as session:
            row = await _latest(session)
    except SQLAlchemyError:
        logger.warning(
            "Could not read config_documents; serving default configuration", exc_info=True
        )
        return None
    if row is None:
        return None
    try:
        data = yaml.safe_load(row.yaml)
    except yaml.YAMLError:
        logger.error("Stored config version %s is not valid YAML; using defaults", row.version)
        return None
    config, errors, _warnings = validate_config_document(data)
    if config is None:
        logger.error(
            "Stored config version %s failed validation (%s); using defaults",
            row.version,
            "; ".join(f"{e['path']}: {e['message']}" for e in errors),
        )
        return None
    set_live_config(config)
    logger.info("Applied stored configuration version %s", row.version)
    return row.version


@router.get("", response_model=ConfigDocumentResponse)
@router.get("/", response_model=ConfigDocumentResponse, include_in_schema=False)
async def get_config(
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(get_current_user),
):
    """Return the live VPP configuration document plus read-only process settings."""
    return _response(await _latest(session))


@router.get("/schema")
async def get_config_schema(_user: UserModel = Depends(get_current_user)) -> dict[str, Any]:
    """JSON Schema for the configuration document accepted by ``PUT /api/v1/config``."""
    return vpp_config_json_schema()


@router.put("", response_model=ConfigDocumentResponse)
@router.put("/", response_model=ConfigDocumentResponse, include_in_schema=False)
async def apply_config(
    body: ConfigApplyRequest,
    session: AsyncSession = Depends(get_db),
    user: UserModel = Depends(require_role(UserRole.ADMIN)),
):
    """Validate, persist and apply a new configuration document (admin only).

    Errors are returned as 422 with ``detail = {"message", "errors": [{path,
    message}]}`` using JSON-Pointer paths, matching the console's
    client-side schema errors.
    """
    if len(body.yaml.encode("utf-8")) > MAX_CONFIG_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"configuration document exceeds {MAX_CONFIG_BYTES} bytes",
        )
    try:
        data = yaml.safe_load(body.yaml)
    except yaml.YAMLError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": "Invalid YAML", "errors": [{"path": "$", "message": str(exc)}]},
        ) from exc

    config, errors, warnings = validate_config_document(data)
    if config is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": "Configuration is invalid", "errors": errors},
        )

    live = await _latest(session)
    live_hash = live.hash if live is not None else _sha256(_default_yaml())
    if body.base_hash is not None and body.base_hash != live_hash:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Configuration changed since it was loaded; reload and re-apply",
        )

    new_hash = _sha256(body.yaml)
    if live is not None and live.hash == new_hash:
        set_live_config(config)
        return _response(live, warnings=warnings)

    next_version = (
        await session.execute(select(func.coalesce(func.max(ConfigDocumentModel.version), 0)))
    ).scalar_one() + 1
    row = ConfigDocumentModel(
        version=next_version, yaml=body.yaml, hash=new_hash, updated_by=user.id
    )
    session.add(row)
    await session.flush()
    await session.refresh(row)
    set_live_config(config)
    return _response(row, warnings=warnings)


@router.post("/validate")
async def validate_config(
    body: dict,
    _user: UserModel = Depends(get_current_user),
):
    """Validate a configuration payload (JSON) without applying it."""
    config, errors, warnings = validate_config_document(body)
    opt = body.get("optimization") if isinstance(body, dict) else None
    if (
        isinstance(opt, dict)
        and isinstance(opt.get("time_horizon"), int)
        and opt["time_horizon"] > 168
    ):
        warnings.append("time_horizon > 168h may be slow")
    return {
        "valid": config is not None,
        "errors": [f"{e['path']}: {e['message']}" for e in errors],
        "warnings": warnings,
    }
