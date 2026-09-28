"""Authentication routes — login, register, API key management."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from vpp.auth.security import (
    create_access_token,
    generate_api_key,
    get_current_principal,
    get_current_user,
    get_password_hash,
    hash_api_key,
    require_role,
    verify_password,
)
from vpp.db.engine import get_db
from vpp.db.models import UserModel
from vpp.db.repositories import UserRepository
from vpp.schemas.auth import (
    APIKeyCreate,
    APIKeyResponse,
    MeResponse,
    Token,
    UserCreate,
    UserResponse,
    UserRole,
    audience_for_role,
)
from vpp.settings import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])


_TOKEN_REQUEST_BODY = {
    "required": True,
    "content": {
        "application/x-www-form-urlencoded": {
            "schema": {
                "type": "object",
                "required": ["username", "password"],
                "properties": {
                    "grant_type": {"type": "string", "enum": ["password"]},
                    "username": {"type": "string"},
                    "password": {"type": "string", "format": "password"},
                },
            }
        },
        "application/json": {
            "schema": {
                "type": "object",
                "required": ["username", "password"],
                "properties": {
                    "username": {"type": "string"},
                    "password": {"type": "string", "format": "password"},
                },
            }
        },
    },
}


def _bad_request(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=detail)


async def _read_credentials(request: Request) -> tuple[str, str, bool]:
    """Return ``(username, password, deprecated_query)`` from the request.

    Accepts the OAuth2 password-grant form body (RFC 6749 section 4.3) or a JSON
    body. Credentials in the query string are still accepted when the body
    is empty, for older clients, but are deprecated: query strings end up in
    proxy/server access logs and browser history.
    """
    ctype = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if ctype in ("application/x-www-form-urlencoded", "multipart/form-data"):
        form = await request.form()
        grant_type = form.get("grant_type")
        if grant_type not in (None, "", "password"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="unsupported_grant_type",
            )
        username, password = form.get("username"), form.get("password")
    elif ctype == "application/json" or ctype.endswith("+json"):
        try:
            data = await request.json()
        except ValueError as exc:
            raise _bad_request("Body is not valid JSON") from exc
        if not isinstance(data, dict):
            raise _bad_request("Body must be a JSON object")
        username, password = data.get("username"), data.get("password")
    elif not await request.body():
        username = request.query_params.get("username")
        password = request.query_params.get("password")
        if isinstance(username, str) and isinstance(password, str):
            return username, password, True
    else:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Send credentials as application/x-www-form-urlencoded or application/json",
        )
    if not isinstance(username, str) or not isinstance(password, str) or not username:
        raise _bad_request("username and password are required")
    return username, password, False


@router.post("/token", response_model=Token, openapi_extra={"requestBody": _TOKEN_REQUEST_BODY})
async def login(request: Request, response: Response, session: AsyncSession = Depends(get_db)):
    """Authenticate and receive a JWT access token.

    Send ``username`` and ``password`` as an OAuth2 password-grant form
    (``application/x-www-form-urlencoded``, optional ``grant_type=password``)
    or as a JSON object. **Deprecated:** passing them as query parameters
    still works (the response then carries a ``Deprecation`` header) but
    leaks the password into access logs; it will be removed in a future
    release.
    """
    username, password, deprecated = await _read_credentials(request)
    if deprecated:
        logger.warning(
            "POST /api/v1/auth/token with credentials in the query string is deprecated; "
            "send a form or JSON body instead"
        )
        response.headers["Deprecation"] = "true"
        response.headers["Warning"] = (
            '299 - "Credentials in the query string are deprecated; use a form or JSON body"'
        )
    user = await UserRepository.get_by_username(session, username)
    if user is None or not user.is_active or not verify_password(password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    settings = get_settings()
    token = create_access_token(
        {
            "sub": user.id,
            "username": user.username,
            "role": user.role,
            "aud": audience_for_role(user.role),
        },
        settings,
    )
    return Token(access_token=token, expires_in=settings.jwt_expire_minutes * 60)


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    body: UserCreate,
    session: AsyncSession = Depends(get_db),
    _admin: UserModel = Depends(require_role(UserRole.ADMIN)),
):
    """Create a new user (admin only)."""
    existing = await UserRepository.get_by_username(session, body.username)
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Username already taken")

    user = await UserRepository.create_user(
        session,
        username=body.username,
        hashed_password=get_password_hash(body.password),
        role=body.role.value,
    )
    return user


@router.get("/me", response_model=MeResponse)
async def me(user: UserModel = Depends(get_current_principal)):
    """Return the currently authenticated user (any role, incl. customers).

    ``audience`` tells the web console which UI to route the user to; it is
    derived from the user's *current* role, not echoed from the token.
    """
    return MeResponse(
        id=user.id,
        username=user.username,
        role=UserRole(user.role),
        is_active=user.is_active,
        created_at=user.created_at,
        audience=audience_for_role(user.role),
    )


@router.post("/api-key", response_model=APIKeyResponse, status_code=status.HTTP_201_CREATED)
async def create_api_key(
    body: APIKeyCreate,
    session: AsyncSession = Depends(get_db),
    user: UserModel = Depends(get_current_user),
):
    """Generate a new API key for programmatic access.

    Non-admins may only mint a key scoped to their own role — otherwise any
    authenticated user could self-issue an admin-role key.
    """
    if user.role != UserRole.ADMIN.value and body.role.value != user.role:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot create an API key with a role higher than your own",
        )

    raw_key = generate_api_key()
    key_obj = await UserRepository.create_api_key(
        session,
        user_id=user.id,
        name=body.name,
        hashed_key=hash_api_key(raw_key),
        role=body.role.value,
    )
    return APIKeyResponse(
        id=key_obj.id,
        name=key_obj.name,
        key=raw_key,
        role=UserRole(key_obj.role),
        created_at=key_obj.created_at,
    )
