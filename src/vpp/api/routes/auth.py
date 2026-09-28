"""Authentication routes — login, register, self-service credentials, API keys.

Admin management of *other* users lives in :mod:`vpp.api.routes.users`.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from vpp.auth.security import (
    generate_api_key,
    get_current_principal,
    get_current_user,
    get_password_hash,
    hash_api_key,
    issue_access_token,
    require_role,
    require_valid_password,
    revoke_user_sessions,
    verify_user_password,
)
from vpp.auth.throttle import login_throttle
from vpp.db.engine import get_db
from vpp.db.models import APIKeyModel, UserModel
from vpp.db.repositories import UserRepository
from vpp.schemas.auth import (
    APIKeyCreate,
    APIKeyInfo,
    APIKeyResponse,
    MeResponse,
    PasswordChange,
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


def _too_many_attempts(retry_after: int) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail="Too many failed login attempts; try again later",
        headers={"Retry-After": str(retry_after)},
    )


def _token_response(user: UserModel) -> Token:
    settings = get_settings()
    return Token(
        access_token=issue_access_token(user, settings),
        expires_in=settings.jwt_expire_minutes * 60,
    )


async def _reload(session: AsyncSession, user: UserModel) -> UserModel:
    """Return the persistent row for ``user``.

    API-key principals may be a detached copy whose role was narrowed to the
    key's scope; credential changes must be applied to the real row.
    """
    fresh = await session.get(UserModel, user.id)
    if fresh is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return fresh


@router.post("/token", response_model=Token, openapi_extra={"requestBody": _TOKEN_REQUEST_BODY})
async def login(request: Request, response: Response, session: AsyncSession = Depends(get_db)):
    """Authenticate and receive a JWT access token.

    Send ``username`` and ``password`` as an OAuth2 password-grant form
    (``application/x-www-form-urlencoded``, optional ``grant_type=password``)
    or as a JSON object. **Deprecated:** passing them as query parameters
    still works (the response then carries a ``Deprecation`` header) but
    leaks the password into access logs; it will be removed in a future
    release.

    After ``VPP_LOGIN_MAX_FAILURES`` failures for one username, attempts
    for it get ``429`` (with ``Retry-After``) for
    ``VPP_LOGIN_LOCKOUT_SECONDS``. Unknown usernames, inactive accounts and
    wrong passwords all answer the same ``401`` in about the same time.
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
    retry_after = login_throttle.retry_after(username)
    if retry_after is not None:
        raise _too_many_attempts(retry_after)

    user = await UserRepository.get_by_username(session, username)
    # Always run bcrypt (a dummy hash for unknown users) so response timing
    # does not reveal which usernames exist.
    password_ok = verify_user_password(user, password)
    if user is None or not password_ok or not user.is_active:
        login_throttle.record_failure(username)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    login_throttle.record_success(username)
    user.last_login_at = datetime.now(timezone.utc)
    return _token_response(user)


async def create_user_account(session: AsyncSession, body: UserCreate) -> UserModel:
    """Create a user after the uniqueness and password-policy checks."""
    existing = await UserRepository.get_by_username(session, body.username)
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Username already taken")
    require_valid_password(body.password, username=body.username)
    return await UserRepository.create_user(
        session,
        username=body.username,
        hashed_password=get_password_hash(body.password),
        role=body.role.value,
    )


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    body: UserCreate,
    session: AsyncSession = Depends(get_db),
    _admin: UserModel = Depends(require_role(UserRole.ADMIN)),
):
    """Create a new user (admin only). Same as ``POST /api/v1/users``."""
    return await create_user_account(session, body)


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


# ---------------------------------------------------------------------------
# Self-service credentials
# ---------------------------------------------------------------------------


@router.post("/password", response_model=Token)
async def change_password(
    body: PasswordChange,
    session: AsyncSession = Depends(get_db),
    principal: UserModel = Depends(get_current_principal),
):
    """Change your own password (any role; the current password is required).

    Every existing session of the account -- including the one making this
    call -- is revoked; the response carries a fresh token for the caller.
    API keys are separate credentials and keep working. Wrong current
    passwords count towards the login throttle.
    """
    user = await _reload(session, principal)
    retry_after = login_throttle.retry_after(user.username)
    if retry_after is not None:
        raise _too_many_attempts(retry_after)
    if not verify_user_password(user, body.current_password):
        login_throttle.record_failure(user.username)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Current password is incorrect"
        )
    if body.new_password == body.current_password:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="New password must differ from the current one",
        )
    require_valid_password(body.new_password, username=user.username)
    user.hashed_password = get_password_hash(body.new_password)
    revoke_user_sessions(user)
    logger.info("User %s changed their password; sessions revoked", user.username)
    return _token_response(user)


@router.post("/logout-all", status_code=status.HTTP_204_NO_CONTENT)
async def logout_everywhere(
    session: AsyncSession = Depends(get_db),
    principal: UserModel = Depends(get_current_principal),
) -> Response:
    """Revoke every session token (and WebSocket token) of your account.

    Includes the token used for this call. API keys are not affected --
    revoke those individually with ``DELETE /api/v1/auth/api-keys/{id}``.
    """
    user = await _reload(session, principal)
    revoke_user_sessions(user)
    logger.info("User %s logged out of all sessions", user.username)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ---------------------------------------------------------------------------
# API keys
# ---------------------------------------------------------------------------


def api_key_info(key: APIKeyModel, username: str | None = None) -> APIKeyInfo:
    return APIKeyInfo(
        id=key.id,
        name=key.name,
        role=UserRole(key.role),
        is_active=key.is_active,
        created_at=key.created_at,
        last_used_at=key.last_used_at,
        key_prefix=key.key_prefix,
        user_id=key.user_id,
        username=username,
    )


async def list_api_keys_for(
    session: AsyncSession, *, user_id: str | None, include_revoked: bool
) -> list[APIKeyInfo]:
    """API keys of one user (or all users when ``user_id`` is None), newest first."""
    stmt = (
        select(APIKeyModel, UserModel.username)
        .join(UserModel, UserModel.id == APIKeyModel.user_id)
        .order_by(APIKeyModel.created_at.desc(), APIKeyModel.id)
    )
    if user_id is not None:
        stmt = stmt.where(APIKeyModel.user_id == user_id)
    if not include_revoked:
        stmt = stmt.where(APIKeyModel.is_active.is_(True))
    rows = (await session.execute(stmt)).all()
    return [api_key_info(key, username) for key, username in rows]


@router.post("/api-key", response_model=APIKeyResponse, status_code=status.HTTP_201_CREATED)
@router.post(
    "/api-keys",
    response_model=APIKeyResponse,
    status_code=status.HTTP_201_CREATED,
    include_in_schema=False,
)
async def create_api_key(
    body: APIKeyCreate,
    session: AsyncSession = Depends(get_db),
    user: UserModel = Depends(get_current_user),
):
    """Generate a new API key for programmatic access (alias: ``POST /api-keys``).

    Non-admins may only mint a key scoped to their own role — otherwise any
    authenticated user could self-issue an admin-role key. The raw key is
    returned only in this response.
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
        key_prefix=raw_key[:12],
    )
    return APIKeyResponse(
        id=key_obj.id,
        name=key_obj.name,
        key=raw_key,
        role=UserRole(key_obj.role),
        created_at=key_obj.created_at,
        key_prefix=key_obj.key_prefix,
    )


@router.get("/api-keys", response_model=list[APIKeyInfo])
async def list_api_keys(
    all_users: bool = Query(False, alias="all", description="Admins only: every user's keys"),
    include_revoked: bool = Query(False, description="Also list revoked keys"),
    session: AsyncSession = Depends(get_db),
    user: UserModel = Depends(get_current_principal),
):
    """List your API keys (never the keys themselves); ``?all=true`` for admins."""
    if all_users and user.role != UserRole.ADMIN.value:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Only admins can list all API keys"
        )
    return await list_api_keys_for(
        session, user_id=None if all_users else user.id, include_revoked=include_revoked
    )


@router.delete("/api-keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_api_key(
    key_id: str,
    session: AsyncSession = Depends(get_db),
    user: UserModel = Depends(get_current_principal),
) -> Response:
    """Revoke an API key: your own, or anyone's for admins. Idempotent.

    Revoked keys stay listed (``include_revoked=true``) for auditing.
    """
    key = await session.get(APIKeyModel, key_id)
    if key is None or (key.user_id != user.id and user.role != UserRole.ADMIN.value):
        # 404 rather than 403: do not confirm that someone else's key exists.
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="API key not found")
    if key.is_active:
        key.is_active = False
        logger.info("API key %s (%s) revoked by %s", key.id, key.name, user.username)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
