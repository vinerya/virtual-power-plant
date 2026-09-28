"""JWT token creation/validation, password hashing, and FastAPI auth dependencies."""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from sqlalchemy.ext.asyncio import AsyncSession

from vpp.db.engine import get_db
from vpp.db.models import APIKeyModel, UserModel
from vpp.db.repositories import UserRepository
from vpp.schemas.auth import (
    AUDIENCE_CUSTOMER,
    AUDIENCE_OPERATOR,
    TokenPayload,
    UserRole,
    audience_for_role,
)
from vpp.settings import Settings, get_settings

_bearer_scheme = HTTPBearer(auto_error=False)
_LAST_USED_RESOLUTION = timedelta(minutes=1)


class _ConfiguredAPIKeyHeader(APIKeyHeader):
    """``APIKeyHeader`` whose header name comes from ``VPP_API_KEY_HEADER``.

    The name is read from the settings on every request (so it follows
    ``get_settings.cache_clear()`` in tests); the OpenAPI security scheme
    uses the name configured when this module is imported.
    """

    def __init__(self) -> None:
        try:
            name = get_settings().api_key_header
        except ValueError:  # invalid settings surface later, in create_app
            name = str(Settings.model_fields["api_key_header"].default)
        super().__init__(name=name, auto_error=False)

    async def __call__(self, request: Request) -> str | None:
        return request.headers.get(get_settings().api_key_header) or None


_api_key_header = _ConfiguredAPIKeyHeader()


# ---------------------------------------------------------------------------
# Password helpers
# ---------------------------------------------------------------------------


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except ValueError:
        # bcrypt >= 5 refuses inputs over 72 bytes (and malformed hashes);
        # such a password can never have been set, so it cannot match.
        return False


def get_password_hash(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def require_valid_password(password: str, *, username: str | None = None) -> None:
    """HTTP wrapper around the password policy: 422 with the reason."""
    from vpp.auth.passwords import password_problem

    problem = password_problem(password, username=username)
    if problem is not None:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=problem)


_dummy_hash: str | None = None


def verify_user_password(user: UserModel | None, plain: str) -> bool:
    """Check ``plain`` against ``user``'s hash in (roughly) constant time.

    For an unknown user a bcrypt check still runs against a dummy hash of
    the same cost, so response timing does not reveal whether a username
    exists. Inactive users are verified too and then refused by the caller.
    """
    global _dummy_hash
    if user is None:
        if _dummy_hash is None:
            _dummy_hash = get_password_hash(secrets.token_urlsafe(16))
        verify_password(plain, _dummy_hash)
        return False
    return verify_password(plain, user.hashed_password)


# ---------------------------------------------------------------------------
# API Key helpers
# ---------------------------------------------------------------------------


def generate_api_key() -> str:
    """Generate a cryptographically secure API key."""
    return f"vpp_{secrets.token_urlsafe(32)}"


def hash_api_key(key: str) -> str:
    """Hash an API key for storage."""
    return hashlib.sha256(key.encode()).hexdigest()


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------


def token_version_ok(claimed: int | None, current: int) -> bool:
    """Whether a token carrying ``ver=claimed`` is still valid for the user.

    Tokens issued before per-user versioning existed have no ``ver`` claim;
    they stay valid (until their normal expiry) only while the user's
    version is still 0 -- the first revocation event invalidates them too.
    """
    return (claimed if claimed is not None else 0) == (current or 0)


def revoke_user_sessions(user: UserModel) -> None:
    """Invalidate every JWT (and WebSocket token) issued to ``user`` so far."""
    user.token_version = (user.token_version or 0) + 1


def issue_access_token(user: UserModel, settings: Settings | None = None) -> str:
    """Mint a session JWT for ``user`` with the standard claims."""
    return create_access_token(
        {
            "sub": user.id,
            "username": user.username,
            "role": user.role,
            "aud": audience_for_role(user.role),
            "ver": user.token_version or 0,
        },
        settings,
    )


def create_access_token(data: dict[str, Any], settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expire_minutes)
    payload = {**data, "exp": expire}
    token: str = jwt.encode(payload, settings.secret_key, algorithm=settings.jwt_algorithm)
    return token


def decode_access_token(token: str, settings: Settings | None = None) -> TokenPayload:
    settings = settings or get_settings()
    try:
        # ``aud`` is validated below (and against the user's role in
        # get_current_principal) rather than by python-jose, which would
        # otherwise reject every token carrying an ``aud`` claim when no
        # single expected audience is passed.
        payload = jwt.decode(
            token,
            settings.secret_key,
            algorithms=[settings.jwt_algorithm],
            options={"verify_aud": False},
        )
        decoded = TokenPayload(**payload)
    except (JWTError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    if decoded.aud is not None and decoded.aud not in (AUDIENCE_OPERATOR, AUDIENCE_CUSTOMER):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token audience",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return decoded


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------


async def get_current_principal(
    bearer: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    api_key: str | None = Depends(_api_key_header),
    session: AsyncSession = Depends(get_db),
) -> UserModel:
    """Resolve the caller from a JWT bearer token or an API key, any role.

    This includes ``customer`` accounts. Only routes that scope every
    response to the caller's own data (the member portal, ``/auth/me``,
    ownership-checked site/metrics reads) should depend on this directly;
    everything else should use :func:`get_current_user`.
    """

    # Try JWT first
    if bearer is not None:
        payload = decode_access_token(bearer.credentials)
        if payload.typ is not None:
            # e.g. short-lived WebSocket tokens — not valid for the HTTP API.
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token type not accepted for this endpoint",
                headers={"WWW-Authenticate": "Bearer"},
            )
        user = await UserRepository.get_by_id(session, payload.sub)
        if user is None or not user.is_active:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
        if not token_version_ok(payload.ver, user.token_version):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Session has been revoked",
                headers={"WWW-Authenticate": "Bearer"},
            )
        # A token minted for one console must not outlive a role change
        # that moves the user to the other one.
        if payload.aud is not None and payload.aud != audience_for_role(user.role):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token audience does not match account",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return user

    # Fall back to API key
    if api_key is not None:
        return await get_api_key_user(api_key, session)

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_current_user(
    user: UserModel = Depends(get_current_principal),
) -> UserModel:
    """Resolve the current *operator-side* user (deny-by-default for customers).

    Every pre-existing operator endpoint depends on this, so adding the
    ``customer`` role cannot silently expose fleet-wide data: customer
    accounts get 403 here and must go through routes that explicitly opt in
    via :func:`get_current_principal` or ``require_role(UserRole.CUSTOMER)``.
    """
    if user.role == UserRole.CUSTOMER.value:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Customer accounts cannot access operator endpoints",
        )
    return user


async def get_api_key_user(api_key: str, session: AsyncSession) -> UserModel:
    """Resolve user from an API key."""
    from vpp.db.repositories import UserRepository as _UR

    hashed = hash_api_key(api_key)
    from sqlalchemy import select

    result = await session.execute(
        select(APIKeyModel).where(
            APIKeyModel.hashed_key == hashed, APIKeyModel.is_active.is_(True)
        )
    )
    key_obj = result.scalar_one_or_none()
    if key_obj is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

    user = await _UR.get_by_id(session, key_obj.user_id)
    if user is None or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    # Record usage, at most once per minute per key so a busy integration
    # does not turn every read into a write. Committed with the request.
    now = datetime.now(timezone.utc)
    last = key_obj.last_used_at
    if last is not None and last.tzinfo is None:  # SQLite returns naive datetimes
        last = last.replace(tzinfo=timezone.utc)
    if last is None or now - last >= _LAST_USED_RESOLUTION:
        key_obj.last_used_at = now

    # A key acts with the *lesser* of its own role and its owner's current
    # role: a viewer-scoped key minted by an admin must not carry admin
    # rights, and a key must not outlive a demotion of its owner. The user
    # is detached before its role is narrowed so the override can never be
    # flushed back to the users table.
    effective = effective_api_key_role(key_obj.role, user.role)
    if effective != user.role:
        session.expunge(user)
        user.role = effective
    return user


#: Privilege order for API-key scoping. ``customer`` is ranked lowest: it
#: only reaches self-scoped portal routes.
_ROLE_RANK: dict[str, int] = {
    UserRole.CUSTOMER.value: 0,
    UserRole.VIEWER.value: 1,
    UserRole.RESEARCHER.value: 1,  # read-only, like viewer
    UserRole.OPERATOR.value: 2,
    UserRole.ADMIN.value: 3,
}


def effective_api_key_role(key_role: str, owner_role: str) -> str:
    """Return the role an API-key request runs with: the less privileged one.

    Unknown roles rank below everything, so a corrupted key role fails closed.
    """
    return min(key_role, owner_role, key=lambda r: _ROLE_RANK.get(r, -1))


def require_role(*roles: str | UserRole):
    """Dependency factory that enforces role-based access."""

    allowed = {r.value if isinstance(r, UserRole) else r for r in roles}

    async def _check(user: UserModel = Depends(get_current_principal)) -> UserModel:
        if user.role not in allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{user.role}' is not authorised for this action",
            )
        return user

    return _check
