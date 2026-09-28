"""Pydantic schemas for authentication and authorization."""

from __future__ import annotations

from datetime import datetime  # noqa: TC003 -- pydantic resolves annotations at runtime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class UserRole(str, Enum):
    """User roles for RBAC."""

    ADMIN = "admin"
    OPERATOR = "operator"
    VIEWER = "viewer"
    RESEARCHER = "researcher"
    #: End customer (household / C&I site owner) using the member portal.
    #: Customers are denied on every operator endpoint by default (see
    #: :func:`vpp.auth.security.get_current_user`) and may only reach
    #: routes that explicitly opt in and scope data to their own sites.
    CUSTOMER = "customer"


#: JWT ``aud`` claim values. The web console's middleware routes on this
#: claim (``customer`` -> member portal, anything else -> operator console);
#: the backend re-checks it against the user's current role on every request.
AUDIENCE_OPERATOR = "operator"
AUDIENCE_CUSTOMER = "customer"


def audience_for_role(role: str) -> str:
    """Return the JWT audience a user with ``role`` is issued."""
    return AUDIENCE_CUSTOMER if role == UserRole.CUSTOMER.value else AUDIENCE_OPERATOR


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


class MeResponse(UserResponse):
    """``GET /api/v1/auth/me`` payload: the user plus the console audience."""

    audience: str = Field(description="'customer' for member-portal users, else 'operator'")


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
    aud: str | None = None  # "operator" | "customer"; absent on legacy tokens
    # Token type. None for regular access tokens; "ws" for the short-lived
    # WebSocket handshake tokens, which the HTTP API must refuse.
    typ: str | None = None
    # WebSocket tokens only: expiry (epoch seconds) of the session credential
    # the token was minted from. An open socket is closed at this time.
    sexp: int | None = None


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
