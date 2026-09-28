"""Pydantic v2 schemas for the alerts API.

The :class:`AlertRead` shape matches ``Alert`` in ``web/lib/api/types.ts``
(the operator console), plus a few extra fields the console ignores.
"""

from __future__ import annotations

from datetime import datetime  # noqa: TC003 - pydantic resolves annotations at runtime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

AlertSeverityLiteral = Literal["info", "warning", "critical"]
AlertStatusLiteral = Literal["active", "acknowledged", "snoozed", "resolved"]
RuleTypeLiteral = Literal["threshold", "rate_of_change", "anomaly"]
ComparisonLiteral = Literal[">", "<", ">=", "<=", "=="]

# Longest snooze accepted by the API (7 days).
MAX_SNOOZE_MS = 7 * 24 * 60 * 60 * 1000


class AlertRead(BaseModel):
    """A fired alert as returned by ``/api/v1/alerts``."""

    id: str
    timestamp: datetime = Field(..., description="When the alert first fired (UTC)")
    severity: AlertSeverityLiteral
    source: str = Field(..., description="Resource id, 'system', ...")
    source_kind: str = "system"
    source_link: str | None = None
    title: str
    message: str
    status: AlertStatusLiteral = Field(
        ...,
        description="Effective status: an expired snooze reads as 'active'",
    )
    snoozed_until: datetime | None = None
    acknowledged_at: datetime | None = None
    acknowledged_by: str | None = None
    resolved_at: datetime | None = None
    rule_id: str | None = None
    rule_name: str = ""
    metric: str | None = None
    value: float | None = None
    threshold: float | None = None
    occurrences: int = 1
    last_fired_at: datetime | None = None


class SnoozeRequest(BaseModel):
    """Body of ``POST /api/v1/alerts/{id}/snooze``.

    Provide ``until`` (absolute, ISO-8601) and/or ``duration_ms``; when both
    are present ``until`` wins (the console sends both, computed from the
    same clock).
    """

    until: datetime | None = None
    duration_ms: int | None = Field(default=None, gt=0, le=MAX_SNOOZE_MS)

    @model_validator(mode="after")
    def _one_of(self) -> SnoozeRequest:
        if self.until is None and self.duration_ms is None:
            raise ValueError("Provide 'until' or 'duration_ms'")
        return self


class AlertRuleBase(BaseModel):
    description: str = Field("", max_length=2000)
    rule_type: RuleTypeLiteral = "threshold"
    metric: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description=(
            "Numeric key of RESOURCE_UPDATED telemetry, e.g. 'soc', 'soh', "
            "'temperature', 'current_power_kw'. 'soc'/'soh' are compared on a "
            "0-1 scale (percent inputs are normalised)."
        ),
    )
    comparison: ComparisonLiteral = ">"
    threshold: float = 0.0
    severity: AlertSeverityLiteral = "warning"
    rate_window_s: float = Field(60.0, gt=0)
    rate_threshold: float = Field(0.0, ge=0)
    z_score_threshold: float = Field(3.0, gt=0)
    cooldown_s: float = Field(300.0, ge=0)
    resource_id: str | None = Field(
        default=None,
        max_length=36,
        description="Restrict the rule to one resource",
    )
    auto_resolve: bool = Field(
        True,
        description="Threshold rules: resolve open alerts once the value is back in range",
    )
    enabled: bool = True


class AlertRuleCreate(AlertRuleBase):
    name: str = Field(..., min_length=1, max_length=255)


class AlertRuleUpdate(BaseModel):
    """Partial update -- all fields optional."""

    name: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=2000)
    rule_type: RuleTypeLiteral | None = None
    metric: str | None = Field(default=None, min_length=1, max_length=64)
    comparison: ComparisonLiteral | None = None
    threshold: float | None = None
    severity: AlertSeverityLiteral | None = None
    rate_window_s: float | None = Field(default=None, gt=0)
    rate_threshold: float | None = Field(default=None, ge=0)
    z_score_threshold: float | None = Field(default=None, gt=0)
    cooldown_s: float | None = Field(default=None, ge=0)
    resource_id: str | None = Field(default=None, max_length=36)
    auto_resolve: bool | None = None
    enabled: bool | None = None


class AlertRuleRead(AlertRuleBase):
    id: str
    name: str
    created_at: datetime | None = None
    updated_at: datetime | None = None
