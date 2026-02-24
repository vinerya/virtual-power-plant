"""Pydantic schemas for energy resource operations."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ResourceType(str, Enum):
    """Supported resource types."""

    BATTERY = "battery"
    SOLAR = "solar"
    WIND_TURBINE = "wind_turbine"


# ---------------------------------------------------------------------------
# Base schemas
# ---------------------------------------------------------------------------

class ResourceBase(BaseModel):
    """Shared resource fields."""

    name: str = Field(..., min_length=1, max_length=255, description="Unique resource name")
    resource_type: ResourceType = Field(..., description="Type of energy resource")
    rated_power: float = Field(..., gt=0, description="Rated power capacity in kW")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata")


class ResourceCreate(ResourceBase):
    """Schema for creating a new resource (discriminated by resource_type)."""
    pass


class ResourceUpdate(BaseModel):
    """Schema for updating a resource (all fields optional)."""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    rated_power: Optional[float] = Field(None, gt=0)
    metadata: Optional[dict[str, Any]] = None
    online: Optional[bool] = None


class ResourceResponse(ResourceBase):
    """Schema returned from API for any resource."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    online: bool = True
    current_power: float = 0.0
    efficiency: float = 0.95
    created_at: datetime
    updated_at: datetime


# ---------------------------------------------------------------------------
# Battery
# ---------------------------------------------------------------------------

class BatteryCreate(ResourceCreate):
    """Create a battery resource."""

    resource_type: ResourceType = ResourceType.BATTERY
    capacity_kwh: float = Field(..., gt=0, description="Total energy capacity in kWh")
    current_charge_kwh: float = Field(..., ge=0, description="Current charge in kWh")
    nominal_voltage: float = Field(..., gt=0, description="Nominal voltage in V")
    charge_efficiency: float = Field(0.95, gt=0, le=1)
    discharge_efficiency: float = Field(0.95, gt=0, le=1)

    @field_validator("current_charge_kwh")
    @classmethod
    def charge_within_capacity(cls, v: float, info) -> float:
        cap = info.data.get("capacity_kwh")
        if cap is not None and v > cap:
            raise ValueError("current_charge_kwh cannot exceed capacity_kwh")
        return v


class BatteryResponse(ResourceResponse):
    """Battery-specific response fields."""

    capacity_kwh: float
    current_charge_kwh: float
    state_of_charge: float = Field(description="SOC as percentage 0-100")
    nominal_voltage: float
    charge_efficiency: float
    discharge_efficiency: float


# ---------------------------------------------------------------------------
# Solar
# ---------------------------------------------------------------------------

class SolarCreate(ResourceCreate):
    """Create a solar resource."""

    resource_type: ResourceType = ResourceType.SOLAR
    panel_area_m2: float = Field(..., gt=0, description="Total panel area in m²")
    panel_efficiency: float = Field(..., gt=0, le=1, description="Panel efficiency 0-1")


class SolarResponse(ResourceResponse):
    """Solar-specific response fields."""

    panel_area_m2: float
    panel_efficiency: float
    irradiance: float = 0.0
    temperature: float = 25.0


# ---------------------------------------------------------------------------
# Wind Turbine
# ---------------------------------------------------------------------------

class WindTurbineCreate(ResourceCreate):
    """Create a wind turbine resource."""

    resource_type: ResourceType = ResourceType.WIND_TURBINE
    rotor_diameter_m: float = Field(..., gt=0, description="Rotor diameter in metres")
    hub_height_m: float = Field(..., gt=0, description="Hub height in metres")
    cut_in_speed_ms: float = Field(..., ge=0, description="Cut-in wind speed in m/s")
    cut_out_speed_ms: float = Field(..., gt=0, description="Cut-out wind speed in m/s")
    rated_speed_ms: float = Field(..., gt=0, description="Rated wind speed in m/s")

    @field_validator("cut_out_speed_ms")
    @classmethod
    def cut_out_gt_cut_in(cls, v: float, info) -> float:
        ci = info.data.get("cut_in_speed_ms")
        if ci is not None and v <= ci:
            raise ValueError("cut_out_speed must be greater than cut_in_speed")
        return v

    @field_validator("rated_speed_ms")
    @classmethod
    def rated_between_cut_in_out(cls, v: float, info) -> float:
        ci = info.data.get("cut_in_speed_ms")
        co = info.data.get("cut_out_speed_ms")
        if ci is not None and v <= ci:
            raise ValueError("rated_speed must be greater than cut_in_speed")
        if co is not None and v >= co:
            raise ValueError("rated_speed must be less than cut_out_speed")
        return v


class WindTurbineResponse(ResourceResponse):
    """Wind turbine–specific response fields."""

    rotor_diameter_m: float
    hub_height_m: float
    cut_in_speed_ms: float
    cut_out_speed_ms: float
    rated_speed_ms: float
    wind_speed: float = 0.0


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class ResourceMetrics(BaseModel):
    """Snapshot of resource metrics."""

    model_config = ConfigDict(from_attributes=True)

    resource_id: str
    resource_name: str
    resource_type: ResourceType
    rated_power: float
    current_power: float
    efficiency: float
    online: bool
    timestamp: datetime
    extra: dict[str, Any] = Field(default_factory=dict)
