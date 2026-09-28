"""Audit-log API schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class AuditEntry(BaseModel):
    """One ``audit_log`` row."""

    id: int
    ts: datetime
    actor_id: str | None = Field(None, description="User id (NULL for anonymous callers)")
    actor_username: str | None = Field(
        None, description="Username at the time (the attempted one for failed logins)"
    )
    action: str = Field(description="e.g. auth.login, user.delete, control.dispatch")
    target_type: str | None = None
    target_id: str | None = None
    client_ip: str | None = None
    outcome: str = Field(description="success | failure | denied")
    details: dict[str, Any] = Field(default_factory=dict)
