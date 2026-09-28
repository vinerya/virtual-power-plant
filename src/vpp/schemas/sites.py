"""Schemas for sites, meter readings and resource telemetry history.

Field names mirror the web console's ``Site`` / ``ResourceMetricsResponse``
types (``web/lib/api/types.ts``) exactly; extra fields are additive.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _check_tz(v: str | None) -> str | None:
    if v is None:
        return v
    try:
        ZoneInfo(v)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(f"unknown IANA timezone {v!r}") from exc
    return v


class SiteCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, max_length=255)
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    region: str | None = Field(None, max_length=128)
    address: str | None = Field(None, max_length=512)
    timezone: str = Field("UTC", description="IANA zone for billing periods / TOU")
    owner_id: str | None = Field(None, description="User id of the owning customer")
    resource_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    _tz = field_validator("timezone")(_check_tz)


class SiteUpdate(BaseModel):
    """Partial update. ``owner_id: null`` explicitly unassigns the owner;
    ``resource_ids`` (when present) *replaces* the site's membership."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(None, min_length=1, max_length=255)
    lat: float | None = Field(None, ge=-90, le=90)
    lon: float | None = Field(None, ge=-180, le=180)
    region: str | None = Field(None, max_length=128)
    address: str | None = Field(None, max_length=512)
    timezone: str | None = None
    owner_id: str | None = None
    resource_ids: list[str] | None = None
    metadata: dict[str, Any] | None = None

    _tz = field_validator("timezone")(_check_tz)


class SiteResponse(BaseModel):
    id: str
    name: str
    lat: float
    lon: float
    region: str | None = None
    address: str | None = None
    timezone: str = "UTC"
    owner_id: str | None = None
    resource_ids: list[str]
    total_resources: int
    online_count: int
    current_power: float = Field(description="kW, sum over resources")
    rated_power: float = Field(description="kW, sum over resources")
    capacity_kwh: float = Field(description="Sum of known battery energy capacities")
    state_of_charge: float | None = Field(
        None, description="Capacity-weighted latest SOC (0-1) of batteries with telemetry"
    )
    active_alerts: int
    health: Literal["green", "yellow", "red"]
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class MeterReadingIn(BaseModel):
    timestamp: datetime = Field(..., description="Interval start (tz-aware; naive = UTC)")
    import_kwh: float = Field(..., ge=0)
    export_kwh: float = Field(0.0, ge=0)

    @field_validator("import_kwh", "export_kwh")
    @classmethod
    def _finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("must be finite")
        return v


class MeterReadingsIngest(BaseModel):
    interval_minutes: Literal[5, 15, 30, 60] = 15
    readings: list[MeterReadingIn] = Field(..., min_length=1, max_length=20_000)


class MeterReadingsIngestResult(BaseModel):
    site_id: str
    received: int
    inserted: int
    updated: int


class MeterReadingOut(BaseModel):
    timestamp: datetime
    interval_minutes: int
    import_kwh: float
    export_kwh: float


class TelemetrySampleIn(BaseModel):
    timestamp: datetime | None = Field(None, description="Defaults to now (UTC)")
    power_kw: float = Field(
        ..., description="+ = consuming/charging, - = discharging; generators report output"
    )
    state_of_charge: float | None = Field(None, ge=0, le=1, description="0-1 fraction")

    @field_validator("power_kw")
    @classmethod
    def _finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("must be finite")
        return v


class TelemetryIngest(BaseModel):
    samples: list[TelemetrySampleIn] = Field(..., min_length=1, max_length=5_000)
    source: str = Field("api", min_length=1, max_length=64)


class TelemetryIngestResult(BaseModel):
    resource_id: str
    accepted: int
    current_power: float


class ResourceMetricsPoint(BaseModel):
    timestamp: datetime
    power: float | None = Field(None, description="Mean kW over the bucket")
    state_of_charge: float | None = Field(None, description="Mean SOC (0-1) over the bucket")
    samples: int


class ResourceMetricsResponse(BaseModel):
    resource_id: str
    window: str = Field(description="Named window (e.g. '24h') or 'custom'")
    start: datetime
    end: datetime
    bucket_seconds: int
    points: list[ResourceMetricsPoint]
