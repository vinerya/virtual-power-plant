"""Schemas for the customer portal and customer/program administration.

Field names mirror ``Customer``, ``CustomerDevice``, ``DRProgram``, ``Bill``
and ``CustomerBillResponse`` in ``web/lib/api`` exactly; extra fields are
additive.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CustomerResponse(BaseModel):
    id: str = Field(description="The customer's user id")
    username: str
    name: str
    email: str | None = None
    address: str | None = None
    tariff_id: str | None = None
    baseline_kwh_per_month: float | None = None
    site_ids: list[str] = Field(default_factory=list)
    is_active: bool = True


class CustomerCreate(BaseModel):
    """Admin onboarding: creates a ``customer``-role user plus its profile."""

    model_config = ConfigDict(extra="forbid")

    username: str = Field(..., min_length=3, max_length=64, pattern="^[a-zA-Z0-9_-]+$")
    password: str = Field(..., min_length=8, max_length=128)
    name: str = Field(..., min_length=1, max_length=255)
    email: str | None = Field(None, max_length=320, pattern=r"^[^@\s]+@[^@\s]+$")
    address: str | None = Field(None, max_length=512)
    tariff_id: str | None = None
    baseline_kwh_per_month: float | None = Field(None, gt=0)


class CustomerUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(None, min_length=1, max_length=255)
    email: str | None = Field(None, max_length=320, pattern=r"^[^@\s]+@[^@\s]+$")
    address: str | None = Field(None, max_length=512)
    tariff_id: str | None = None
    baseline_kwh_per_month: float | None = Field(None, gt=0)


class CustomerDevice(BaseModel):
    id: str
    kind: str = Field(description="Resource type: battery | solar | wind_turbine | ...")
    name: str
    state: str = Field(description="offline | charging | discharging | generating | idle")
    current_power: float
    state_of_charge: float | None = Field(None, description="0-1, latest telemetry")
    setpoint_c: float | None = None
    online: bool
    rated_power: float
    site_id: str | None = None


class BillLineItemOut(BaseModel):
    kind: str
    name: str
    amount: float
    quantity: float | None = None
    unit: str | None = None
    rate: float | None = None


class BillPeriod(BaseModel):
    start: datetime
    end: datetime


class BillOut(BaseModel):
    total: float
    currency: str = "USD"
    tariff_id: str | None = None
    line_items: list[BillLineItemOut]
    period: BillPeriod
    metadata: dict[str, Any] = Field(default_factory=dict)


class CustomerBillResponse(BaseModel):
    bill: BillOut
    baseline: BillOut | None = None
    savings: float | None = None
    this_month_kwh: float
    last_month_kwh: float


class DRProgramResponse(BaseModel):
    id: str
    name: str
    description: str
    utility: str | None = None
    incentive_per_event: float | None = None
    enrolled: bool | None = None
    active: bool = True
    enrolled_count: int | None = None


class DRProgramCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, max_length=255)
    description: str = Field("", max_length=4000)
    utility: str | None = Field(None, max_length=255)
    incentive_per_event: float | None = Field(None, ge=0)
    active: bool = True


class DRProgramUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(None, min_length=1, max_length=255)
    description: str | None = Field(None, max_length=4000)
    utility: str | None = Field(None, max_length=255)
    incentive_per_event: float | None = Field(None, ge=0)
    active: bool | None = None


class EnrollmentRequest(BaseModel):
    program_ids: list[str] = Field(..., min_length=1, max_length=50)
    acknowledged: bool


class EnrollmentResponse(BaseModel):
    ok: bool = True
    enrolled: list[str] = Field(description="All program ids the customer is now enrolled in")
    device_ids: list[str] = Field(description="Devices covered by the enrollment")
