"""Admin user management: ``/api/v1/users``.

Every route requires the ``admin`` role. Self-service (own password, own
API keys, "log out everywhere") lives under ``/api/v1/auth``.

Safety rules enforced here:

* an admin cannot deactivate, demote or delete **their own** account
  (another admin has to), so nobody locks themselves out by accident;
* the **last active admin** can never be deactivated, demoted or deleted;
* role changes and deactivation revoke the user's sessions (JWT
  ``token_version`` bump); deactivation also revokes all of their API keys,
  so re-activating an account does not silently revive old keys;
* deleting a user removes the account together with its API keys, customer
  profile and programme enrolments; sites they owned and config versions
  they saved are kept with the reference cleared. Their tokens stop working
  immediately (the subject no longer exists). The audit log keeps the
  user's id and username;
* an admin resets *their own* password through ``POST /api/v1/auth/password``
  (which needs the current password), not through the reset route.

Every change is recorded in the audit log (:mod:`vpp.audit`).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from vpp import audit
from vpp.api.pagination import Page, page_params, paginate
from vpp.api.routes.auth import api_key_page, create_user_account, list_api_keys_for
from vpp.auth.security import (
    get_password_hash,
    require_role,
    require_valid_password,
    revoke_user_sessions,
)
from vpp.db.engine import get_db
from vpp.db.models import (
    APIKeyModel,
    ConfigDocumentModel,
    CustomerProfileModel,
    ProgramEnrollmentModel,
    SiteModel,
    UserModel,
)
from vpp.schemas.auth import (
    APIKeyInfo,
    PasswordReset,
    UserCreate,
    UserDetail,
    UserRole,
    UserUpdate,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/users", tags=["Users"])

_admin = require_role(UserRole.ADMIN)
_user_page = page_params(default_limit=100, max_limit=500)


async def _get_user(session: AsyncSession, user_id: str) -> UserModel:
    user = await session.get(UserModel, user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return user


async def _key_counts(session: AsyncSession, user_ids: list[str]) -> dict[str, int]:
    if not user_ids:
        return {}
    rows = await session.execute(
        select(APIKeyModel.user_id, func.count(APIKeyModel.id))
        .where(APIKeyModel.user_id.in_(user_ids), APIKeyModel.is_active.is_(True))
        .group_by(APIKeyModel.user_id)
    )
    return {uid: int(n) for uid, n in rows.all()}


def _detail(user: UserModel, key_count: int = 0) -> UserDetail:
    return UserDetail(
        id=user.id,
        username=user.username,
        role=UserRole(user.role),
        is_active=user.is_active,
        created_at=user.created_at,
        last_login_at=user.last_login_at,
        api_key_count=key_count,
    )


async def _active_admin_ids(session: AsyncSession) -> list[str]:
    # FOR UPDATE (PostgreSQL) serialises concurrent demotions of the last
    # two admins; SQLite ignores it and serialises writers anyway.
    result = await session.execute(
        select(UserModel.id)
        .where(UserModel.role == UserRole.ADMIN.value, UserModel.is_active.is_(True))
        .with_for_update()
    )
    return list(result.scalars().all())


def _refuse(
    session: AsyncSession,
    request: Request,
    admin: UserModel,
    action: str,
    user: UserModel,
    detail: str,
) -> HTTPException:
    """Audit a refused change (``denied``) and return the 409 to raise."""
    audit.record(
        session,
        request,
        action,
        actor=admin,
        target_type="user",
        target_id=user.id,
        outcome="denied",
        details={"username": user.username, "reason": detail},
        always=True,
    )
    return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=detail)


@router.get("", response_model=list[UserDetail])
async def list_users(
    response: Response,
    role: UserRole | None = Query(None),
    is_active: bool | None = Query(None),
    page: Page = Depends(_user_page),
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(_admin),
):
    """List users (optionally filtered by role / active state), by username.

    Paginated (``limit`` / ``offset``); the total is in ``X-Total-Count``.
    """
    stmt = select(UserModel).order_by(UserModel.username, UserModel.id)
    if role is not None:
        stmt = stmt.where(UserModel.role == role.value)
    if is_active is not None:
        stmt = stmt.where(UserModel.is_active.is_(is_active))
    users = await paginate(session, response, stmt, page)
    counts = await _key_counts(session, [u.id for u in users])
    return [_detail(u, counts.get(u.id, 0)) for u in users]


@router.post("", response_model=UserDetail, status_code=status.HTTP_201_CREATED)
async def create_user(
    body: UserCreate,
    request: Request,
    session: AsyncSession = Depends(get_db),
    admin: UserModel = Depends(_admin),
):
    """Create a user (same as ``POST /api/v1/auth/register``)."""
    user = await create_user_account(session, body)
    await session.refresh(user)
    logger.info("User %s (%s) created by %s", user.username, user.role, admin.username)
    audit.record(
        session,
        request,
        "user.create",
        actor=admin,
        target_type="user",
        target_id=user.id,
        details={"username": user.username, "role": user.role},
    )
    return _detail(user)


@router.get("/{user_id}", response_model=UserDetail)
async def get_user(
    user_id: str,
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(_admin),
):
    user = await _get_user(session, user_id)
    counts = await _key_counts(session, [user.id])
    return _detail(user, counts.get(user.id, 0))


@router.patch("/{user_id}", response_model=UserDetail)
async def update_user(
    user_id: str,
    body: UserUpdate,
    request: Request,
    session: AsyncSession = Depends(get_db),
    admin: UserModel = Depends(_admin),
):
    """Change a user's role and/or activate / deactivate them.

    ``409`` when the change would deactivate or demote your own account or
    the last active admin.
    """
    user = await _get_user(session, user_id)
    old_role, old_active = user.role, user.is_active
    new_role = body.role.value if body.role is not None else user.role
    new_active = body.is_active if body.is_active is not None else user.is_active
    role_changed = new_role != user.role
    deactivating = user.is_active and not new_active
    activating = not user.is_active and new_active
    action = (
        "user.role_change"
        if role_changed
        else (
            "user.deactivate" if deactivating else "user.activate" if activating else "user.update"
        )
    )

    loses_admin = (
        user.role == UserRole.ADMIN.value
        and user.is_active
        and (new_role != UserRole.ADMIN.value or not new_active)
    )
    if loses_admin:
        if user.id == admin.id:
            raise _refuse(
                session,
                request,
                admin,
                action,
                user,
                "You cannot deactivate or demote your own account; ask another admin",
            )
        if [uid for uid in await _active_admin_ids(session) if uid != user.id] == []:
            raise _refuse(
                session,
                request,
                admin,
                action,
                user,
                "Cannot deactivate or demote the last active admin",
            )

    user.role = new_role
    user.is_active = new_active
    if role_changed or deactivating:
        revoke_user_sessions(user)
    if deactivating:
        await session.execute(
            update(APIKeyModel)
            .where(APIKeyModel.user_id == user.id, APIKeyModel.is_active.is_(True))
            .values(is_active=False)
        )
    await session.flush()
    await session.refresh(user)
    if role_changed or deactivating or activating:
        logger.info(
            "User %s updated by %s: role=%s active=%s",
            user.username,
            admin.username,
            user.role,
            user.is_active,
        )
        audit.record(
            session,
            request,
            action,
            actor=admin,
            target_type="user",
            target_id=user.id,
            details={
                "username": user.username,
                "role": {"from": old_role, "to": user.role},
                "is_active": {"from": old_active, "to": user.is_active},
            },
        )
    counts = await _key_counts(session, [user.id])
    return _detail(user, counts.get(user.id, 0))


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: str,
    request: Request,
    session: AsyncSession = Depends(get_db),
    admin: UserModel = Depends(_admin),
) -> Response:
    """Permanently delete a user.

    Removes the account with its API keys, customer profile and programme
    enrolments; sites it owned and config versions it saved are kept with
    the reference cleared. Existing tokens stop working immediately.
    ``409`` for your own account or the last active admin. Deactivate
    (``PATCH`` with ``is_active=false``) instead to keep the account.
    """
    user = await _get_user(session, user_id)
    if user.id == admin.id:
        raise _refuse(
            session,
            request,
            admin,
            "user.delete",
            user,
            "You cannot delete your own account; ask another admin",
        )
    if (
        user.role == UserRole.ADMIN.value
        and user.is_active
        and not [uid for uid in await _active_admin_ids(session) if uid != user.id]
    ):
        raise _refuse(
            session, request, admin, "user.delete", user, "Cannot delete the last active admin"
        )

    key_count = (
        await session.execute(
            select(func.count(APIKeyModel.id)).where(APIKeyModel.user_id == user.id)
        )
    ).scalar_one()
    # Explicit clean-up rather than relying on ON DELETE: SQLite enforces
    # foreign keys only when the connection enables them.
    await session.execute(delete(APIKeyModel).where(APIKeyModel.user_id == user.id))
    await session.execute(
        delete(ProgramEnrollmentModel).where(ProgramEnrollmentModel.user_id == user.id)
    )
    await session.execute(
        delete(CustomerProfileModel).where(CustomerProfileModel.user_id == user.id)
    )
    await session.execute(
        update(SiteModel).where(SiteModel.owner_id == user.id).values(owner_id=None)
    )
    await session.execute(
        update(ConfigDocumentModel)
        .where(ConfigDocumentModel.updated_by == user.id)
        .values(updated_by=None)
    )
    username, role = user.username, user.role
    # Bump the session generation too, in case the row is ever restored.
    revoke_user_sessions(user)
    await session.delete(user)
    await session.flush()
    logger.info("User %s (%s) deleted by %s", username, role, admin.username)
    audit.record(
        session,
        request,
        "user.delete",
        actor=admin,
        target_type="user",
        target_id=user_id,
        details={"username": username, "role": role, "api_keys_deleted": int(key_count)},
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{user_id}/password", status_code=status.HTTP_204_NO_CONTENT)
async def reset_password(
    user_id: str,
    body: PasswordReset,
    request: Request,
    session: AsyncSession = Depends(get_db),
    admin: UserModel = Depends(_admin),
) -> Response:
    """Set a new password for another user and revoke their sessions."""
    user = await _get_user(session, user_id)
    if user.id == admin.id:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Change your own password with POST /api/v1/auth/password",
        )
    require_valid_password(body.new_password, username=user.username)
    user.hashed_password = get_password_hash(body.new_password)
    revoke_user_sessions(user)
    logger.info("Password of %s reset by %s; sessions revoked", user.username, admin.username)
    audit.record(
        session,
        request,
        "user.password_reset",
        actor=admin,
        target_type="user",
        target_id=user.id,
        details={"username": user.username},
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{user_id}/revoke-sessions", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_sessions(
    user_id: str,
    request: Request,
    session: AsyncSession = Depends(get_db),
    admin: UserModel = Depends(_admin),
) -> Response:
    """Invalidate every JWT / WebSocket token of the user (API keys unaffected)."""
    user = await _get_user(session, user_id)
    revoke_user_sessions(user)
    logger.info("Sessions of %s revoked by %s", user.username, admin.username)
    audit.record(
        session,
        request,
        "user.sessions_revoke",
        actor=admin,
        target_type="user",
        target_id=user.id,
        details={"username": user.username},
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/{user_id}/api-keys", response_model=list[APIKeyInfo])
async def list_user_api_keys(
    user_id: str,
    response: Response,
    include_revoked: bool = Query(False),
    page: Page = Depends(api_key_page),
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(_admin),
):
    """A user's API keys, newest first; paginated, total in ``X-Total-Count``."""
    await _get_user(session, user_id)
    return await list_api_keys_for(
        session, user_id=user_id, include_revoked=include_revoked, page=page, response=response
    )
