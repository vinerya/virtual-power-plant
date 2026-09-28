"""Pydantic schemas for battery degradation API surface (M3)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class SOHResponse(BaseModel):
    """Current SOH snapshot for a battery."""

    model_config = ConfigDict(from_attributes=True)

    battery_id: str
    state_of_health: float = Field(..., ge=0.0, le=1.0)
    cumulative_throughput_kwh: float = Field(..., ge=0.0)
    last_update: datetime | None = None
    chemistry: str | None = None
    daily_efc: float = Field(0.0, description="Estimated equivalent full cycles per day.")
    projected_eol_date: datetime | None = Field(
        None,
        description="Projected end-of-life date (SOH reaches 0.8) at the current daily_efc.",
    )


class SOHHistoryItem(BaseModel):
    """Single SOH sample."""

    model_config = ConfigDict(from_attributes=True)

    timestamp: datetime
    state_of_health: float
    cumulative_throughput_kwh: float
    loss_fraction: float = 0.0


class SOHUpdateRequest(BaseModel):
    """Manually trigger a degradation update over a SOC window."""

    soc_trace: list[float] = Field(..., min_length=2)
    timestamps: list[datetime] = Field(..., min_length=2)
    temperatures_c: list[float] | None = None
