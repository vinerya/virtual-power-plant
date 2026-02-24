"""Pydantic schemas for authentication and authorization."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class UserRole(str, Enum):
    """User roles for RBAC."""

    ADMIN = "admin"
    OPERATOR = "operator"
    VIEWER = "viewer"
    RESEARCHER = "researcher"


class UserCreate(BaseModel):
    """Schema for creating a new user."""

    username: str = Field(..., min_length=3, max_length=64, pattern="^[a-zA-Z0-9_-]+$")
    password: str = Field(..., min_length=8, max_length=128)
    role: UserRole = UserRole.VIEWER


class UserResponse(BaseModel):
    """Schema returned for user queries (no password)."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    username: str
    role: UserRole
    is_active: bool = True
    created_at: datetime


class Token(BaseModel):
    """JWT access token response."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(description="Seconds until expiration")


class TokenPayload(BaseModel):
    """Decoded JWT payload."""

    sub: str  # user id
    username: str
    role: UserRole
    exp: int  # expiration timestamp


class APIKeyCreate(BaseModel):
    """Schema for generating an API key."""

    name: str = Field(..., min_length=1, max_length=128, description="Human label for the key")
    role: UserRole = UserRole.VIEWER


class APIKeyResponse(BaseModel):
    """Returned once after key creation; the raw key is shown only this once."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    key: str = Field(description="Store securely — not retrievable after creation")
    role: UserRole
    created_at: datetime
