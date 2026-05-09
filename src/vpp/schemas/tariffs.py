"""Pydantic v2 schemas for tariff CRUD and bill simulation."""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TariffCreate(BaseModel):
    """Request body for creating a tariff."""

    name: str = Field(..., min_length=1, max_length=255)
    utility: str = Field("", max_length=255)
    urdb_json: dict[str, Any] = Field(..., description="URDB-shaped tariff JSON")
    effective_date: Optional[date] = None
    urdb_label: Optional[str] = Field(default=None, max_length=128)


class TariffUpdate(BaseModel):
    """Partial update — all fields optional."""

    name: Optional[str] = Field(default=None, min_length=1, max_length=255)
    utility: Optional[str] = Field(default=None, max_length=255)
    urdb_json: Optional[dict[str, Any]] = None
    effective_date: Optional[date] = None
    urdb_label: Optional[str] = Field(default=None, max_length=128)


class TariffRead(BaseModel):
    """Response model for tariff resources."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    utility: str
    urdb_label: Optional[str] = None
    urdb_json: dict[str, Any]
    effective_date: Optional[date] = None
    created_at: datetime
    updated_at: datetime


class MeterTraceDTO(BaseModel):
    """API representation of a metered interval-energy trace."""

    timestamps: list[datetime] = Field(..., description="Interval-start timestamps (ISO, tz-aware)")
    import_kwh: list[float]
    export_kwh: list[float] = Field(default_factory=list)
    interval_minutes: int = 60

    @model_validator(mode="after")
    def _check_lengths(self) -> "MeterTraceDTO":
        n = len(self.timestamps)
        if len(self.import_kwh) != n:
            raise ValueError("import_kwh length must equal timestamps length")
        if self.export_kwh and len(self.export_kwh) != n:
            raise ValueError("export_kwh length must equal timestamps length")
        return self


class BillSimulationRequest(BaseModel):
    """Bill simulation: provide either a stored tariff_id OR an inline urdb_json."""

    tariff_id: Optional[str] = None
    urdb_json: Optional[dict[str, Any]] = None
    meter_trace: MeterTraceDTO
    billing_period_start: datetime
    billing_period_end: datetime

    @model_validator(mode="after")
    def _xor(self) -> "BillSimulationRequest":
        if (self.tariff_id is None) == (self.urdb_json is None):
            raise ValueError("Provide exactly one of tariff_id or urdb_json")
        if self.billing_period_end <= self.billing_period_start:
            raise ValueError("billing_period_end must be after billing_period_start")
        return self


class BillLineItemDTO(BaseModel):
    kind: str
    label: str
    quantity: float
    unit: str
    rate: float
    amount: float


class BillResponse(BaseModel):
    total: float
    tariff_name: str = ""
    line_items: list[BillLineItemDTO]
    period_start: datetime
    period_end: datetime


class URDBImportRequest(BaseModel):
    """Request to import a URDB record from openei.org."""

    urdb_label: str = Field(..., min_length=1, max_length=128, description="URDB record id (getpage)")
    name_override: Optional[str] = None
