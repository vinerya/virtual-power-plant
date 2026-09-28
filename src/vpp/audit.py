"""Audit log of security-relevant actions (``audit_log`` table).

Route handlers call :func:`record` with the request's database session. The
row is *queued* on the session and inserted in its own transaction when the
request finishes (:func:`vpp.db.engine.defer_row`):

* after a successful request, once its own changes have committed -- so an
  action that rolls back is not audited as done;
* with ``always=True`` also after a failed request (e.g. a rejected login
  answering ``401``), since those failures are what an audit trail is for.

A failure to write the row is logged as a warning and never fails the
request. Details are passed through :func:`sanitize_details`: keys that look
like credentials are dropped and the JSON is size-capped, so passwords,
tokens and API keys never reach the table.

Action names are ``<area>.<verb>`` (see :data:`ACTIONS`); ``outcome`` is
``success``, ``failure`` or ``denied``.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from vpp.client_ip import parse_trusted_proxies, resolve_client_ip
from vpp.db.engine import defer_row
from vpp.db.models import AuditLogModel
from vpp.settings import get_settings

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession
    from starlette.requests import HTTPConnection

    from vpp.db.models import UserModel

logger = logging.getLogger(__name__)

#: Actions recorded by the API (documented in docs/security.md).
ACTIONS: tuple[str, ...] = (
    "auth.login",
    "auth.logout",
    "auth.logout_all",
    "auth.password_change",
    "api_key.create",
    "api_key.revoke",
    "user.create",
    "user.update",
    "user.role_change",
    "user.deactivate",
    "user.activate",
    "user.password_reset",
    "user.sessions_revoke",
    "user.delete",
    "control.dispatch",
    "control.v2g_schedule",
    "control.v2g_dispatch",
    "control.v2g_bid",
    "control.ocpp_remote_start",
    "control.ocpp_remote_stop",
    "control.dr_opt",
    "control.protocol_connect",
    "control.protocol_disconnect",
    "config.update",
    "market.order_submit",
    "market.order_cancel",
    "market.strategy_run",
)

OUTCOMES: tuple[str, ...] = ("success", "failure", "denied")

_SECRET_MARKERS = ("password", "secret", "token", "authorization", "credential", "hashed")
_SECRET_KEYS = frozenset({"key", "api_key", "apikey", "raw_key"})
_MAX_DETAILS_CHARS = 2000
_MAX_STRING_CHARS = 256


def _is_secret_key(key: str) -> bool:
    k = key.lower()
    return k in _SECRET_KEYS or any(marker in k for marker in _SECRET_MARKERS)


def _clean(value: Any, depth: int = 0) -> Any:
    if depth > 4:
        return "..."
    if isinstance(value, dict):
        return {
            str(k): _clean(v, depth + 1) for k, v in value.items() if not _is_secret_key(str(k))
        }
    if isinstance(value, (list, tuple)):
        return [_clean(v, depth + 1) for v in list(value)[:50]]
    if isinstance(value, str):
        return value[:_MAX_STRING_CHARS]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)[:_MAX_STRING_CHARS]


def sanitize_details(details: dict[str, Any] | None) -> str | None:
    """JSON for ``details_json``: credential-like keys removed, size-capped."""
    if not details:
        return None
    text = json.dumps(_clean(details), default=str, separators=(",", ":"))
    if len(text) > _MAX_DETAILS_CHARS:
        text = json.dumps({"truncated": True, "preview": text[: _MAX_DETAILS_CHARS - 64]})
    return text


def client_ip(request: HTTPConnection | None) -> str | None:
    """Client address of ``request`` honouring ``VPP_TRUSTED_PROXIES``."""
    if request is None:
        return None
    try:
        trusted = parse_trusted_proxies(get_settings().trusted_proxies)
    except ValueError:
        trusted = ()
    forwarded = request.headers.getlist("x-forwarded-for")
    ip = resolve_client_ip(
        request.client.host if request.client else None,
        ",".join(forwarded) if forwarded else None,
        request.headers.get("x-real-ip"),
        trusted,
    )
    return None if ip == "unknown" else ip[:64]


def build_entry(
    request: HTTPConnection | None,
    action: str,
    *,
    actor: UserModel | None = None,
    actor_username: str | None = None,
    target_type: str | None = None,
    target_id: str | None = None,
    outcome: str = "success",
    details: dict[str, Any] | None = None,
) -> AuditLogModel:
    """Build (but do not persist) an audit row."""
    username = actor_username
    if username is None and actor is not None:
        username = getattr(actor, "username", None)
    return AuditLogModel(
        ts=datetime.now(timezone.utc),
        actor_id=getattr(actor, "id", None) if actor is not None else None,
        actor_username=username[:64] if username else None,
        action=action,
        target_type=target_type,
        target_id=str(target_id)[:128] if target_id is not None else None,
        client_ip=client_ip(request),
        outcome=outcome,
        details_json=sanitize_details(details),
    )


def record(
    session: AsyncSession,
    request: HTTPConnection | None,
    action: str,
    *,
    actor: UserModel | None = None,
    actor_username: str | None = None,
    target_type: str | None = None,
    target_id: str | None = None,
    outcome: str = "success",
    details: dict[str, Any] | None = None,
    always: bool = False,
) -> None:
    """Queue an audit row on ``session`` (a :func:`~vpp.db.engine.get_db` session).

    ``always=True`` writes it even when the request then fails (use it for
    failed / denied attempts that raise an HTTP error right after).
    Never raises.
    """
    try:
        entry = build_entry(
            request,
            action,
            actor=actor,
            actor_username=actor_username,
            target_type=target_type,
            target_id=target_id,
            outcome=outcome,
            details=details,
        )
        defer_row(session, entry, on_error=always)
    except Exception:
        logger.warning("Could not queue audit entry %s", action, exc_info=True)
