"""Authentication routes — login, register, API key management."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from vpp.auth.security import (
    create_access_token,
    generate_api_key,
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
    Token,
    UserCreate,
    UserResponse,
    UserRole,
)
from vpp.settings import get_settings

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])


@router.post("/token", response_model=Token)
async def login(username: str, password: str, session: AsyncSession = Depends(get_db)):
    """Authenticate and receive a JWT access token."""
    user = await UserRepository.get_by_username(session, username)
    if user is None or not verify_password(password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    settings = get_settings()
    token = create_access_token(
        {"sub": user.id, "username": user.username, "role": user.role},
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


@router.get("/me", response_model=UserResponse)
async def me(user: UserModel = Depends(get_current_user)):
    """Return the currently authenticated user."""
    return user


@router.post("/api-key", response_model=APIKeyResponse, status_code=status.HTTP_201_CREATED)
async def create_api_key(
    body: APIKeyCreate,
    session: AsyncSession = Depends(get_db),
    user: UserModel = Depends(get_current_user),
):
    """Generate a new API key for programmatic access."""
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
