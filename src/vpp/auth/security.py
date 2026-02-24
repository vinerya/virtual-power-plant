"""JWT token creation/validation, password hashing, and FastAPI auth dependencies."""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, APIKeyHeader
from jose import JWTError, jwt
from sqlalchemy.ext.asyncio import AsyncSession

from vpp.db.engine import get_db
from vpp.db.models import UserModel, APIKeyModel
from vpp.db.repositories import UserRepository
from vpp.schemas.auth import TokenPayload, UserRole
from vpp.settings import Settings, get_settings

_bearer_scheme = HTTPBearer(auto_error=False)
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


# ---------------------------------------------------------------------------
# Password helpers
# ---------------------------------------------------------------------------

def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


def get_password_hash(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


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

def create_access_token(data: dict[str, Any], settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expire_minutes)
    payload = {**data, "exp": expire}
    return jwt.encode(payload, settings.secret_key, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str, settings: Settings | None = None) -> TokenPayload:
    settings = settings or get_settings()
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.jwt_algorithm])
        return TokenPayload(**payload)
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------

async def get_current_user(
    bearer: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    api_key: str | None = Depends(_api_key_header),
    session: AsyncSession = Depends(get_db),
) -> UserModel:
    """Resolve the current user from either a JWT bearer token or an API key."""

    # Try JWT first
    if bearer is not None:
        payload = decode_access_token(bearer.credentials)
        user = await UserRepository.get_by_id(session, payload.sub)
        if user is None or not user.is_active:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
        return user

    # Fall back to API key
    if api_key is not None:
        return await get_api_key_user(api_key, session)

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_api_key_user(api_key: str, session: AsyncSession) -> UserModel:
    """Resolve user from an API key."""
    from vpp.db.repositories import UserRepository as _UR

    hashed = hash_api_key(api_key)
    from sqlalchemy import select
    result = await session.execute(
        select(APIKeyModel).where(APIKeyModel.hashed_key == hashed, APIKeyModel.is_active.is_(True))
    )
    key_obj = result.scalar_one_or_none()
    if key_obj is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

    user = await _UR.get_by_id(session, key_obj.user_id)
    if user is None or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


def require_role(*roles: str | UserRole):
    """Dependency factory that enforces role-based access."""

    allowed = {r.value if isinstance(r, UserRole) else r for r in roles}

    async def _check(user: UserModel = Depends(get_current_user)) -> UserModel:
        if user.role not in allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{user.role}' is not authorised for this action",
            )
        return user

    return _check
