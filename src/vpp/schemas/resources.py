"""Pydantic schemas for energy resource operations.

Create requests are a discriminated union on ``resource_type``: each type has
its own model with its own typed fields (validated, never silently dropped --
unknown fields are rejected with 422). ``"wind"`` is accepted as an input
alias of the canonical ``"wind_turbine"``; responses always carry the
canonical name.

Persistence (see :mod:`vpp.api.routes.resources`):

* ``capacity_kwh`` -> ``resources.nominal_energy_kwh`` (also read by the
  degradation model), ``chemistry`` -> ``resources.chemistry``,
  ``efficiency`` -> ``resources.efficiency``;
* every other type-specific field -> ``resources.config_json``.

``state_of_charge`` is a 0-1 fraction everywhere. On create/update it is the
*configured* SOC, used by the optimizer until telemetry for the resource
arrives; responses report the latest telemetry SOC when there is one and say
which one they report in ``state_of_charge_source``.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    Tag,
    TypeAdapter,
    field_validator,
    model_validator,
)


class ResourceType(str, Enum):
    """Supported resource types."""

    BATTERY = "battery"
    SOLAR = "solar"
    WIND_TURBINE = "wind_turbine"


#: Input aliases accepted for ``resource_type`` (and the list filter).
RESOURCE_TYPE_ALIASES: dict[str, str] = {"wind": ResourceType.WIND_TURBINE.value}

#: Chemistries with a dedicated degradation preset (vpp.degradation).
BatteryChemistry = Literal["lfp", "nmc"]


def normalize_resource_type(value: Any) -> Any:
    """Map aliases (``"wind"``) and case variants onto the canonical type name."""
    if isinstance(value, ResourceType):
        return value.value
    if isinstance(value, str):
        v = value.strip().lower()
        return RESOURCE_TYPE_ALIASES.get(v, v)
    return value


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------


class ResourceBase(BaseModel):
    """Shared resource fields."""

    name: str = Field(..., min_length=1, max_length=255, description="Unique resource name")
    resource_type: ResourceType = Field(..., description="Type of energy resource")
    rated_power: float = Field(..., gt=0, description="Rated power capacity in kW")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata")

    @field_validator("resource_type", mode="before")
    @classmethod
    def _alias_resource_type(cls, v: Any) -> Any:
        return normalize_resource_type(v)


class ResourceCreate(ResourceBase):
    """Fields common to every resource type on create."""

    model_config = ConfigDict(extra="forbid")

    efficiency: float | None = Field(
        None, gt=0, le=1, description="Round-trip / conversion efficiency 0-1 (default 0.95)"
    )

    @classmethod
    def type_fields(cls) -> frozenset[str]:
        """Names of the fields specific to this resource type."""
        return frozenset(cls.model_fields) - frozenset(ResourceCreate.model_fields)


class BatteryCreate(ResourceCreate):
    """Create a battery resource.

    Every type-specific field is optional; a battery without
    ``capacity_kwh`` is still accepted, and the optimizer/degradation model
    then fall back to a documented C/4 assumption (``rated_power * 4`` kWh).
    """

    resource_type: Literal[ResourceType.BATTERY] = ResourceType.BATTERY
    capacity_kwh: float | None = Field(None, gt=0, description="Nameplate energy capacity (kWh)")
    state_of_charge: float | None = Field(
        None, ge=0, le=1, description="Configured SOC (0-1) used until telemetry arrives"
    )
    current_charge_kwh: float | None = Field(
        None,
        ge=0,
        description="Alternative to state_of_charge: stored energy in kWh (needs capacity_kwh)",
    )
    chemistry: BatteryChemistry | None = Field(
        None, description="Cell chemistry; selects the degradation preset (lfp | nmc)"
    )
    nominal_voltage: float | None = Field(None, gt=0, description="Nominal voltage in V")
    charge_efficiency: float | None = Field(None, gt=0, le=1)
    discharge_efficiency: float | None = Field(None, gt=0, le=1)
    max_charge_kw: float | None = Field(
        None, gt=0, description="Charge power limit (kW); defaults to rated_power"
    )
    max_discharge_kw: float | None = Field(
        None, gt=0, description="Discharge power limit (kW); defaults to rated_power"
    )
    soc_min: float | None = Field(None, ge=0, le=1, description="Lower SOC bound (0-1)")
    soc_max: float | None = Field(None, ge=0, le=1, description="Upper SOC bound (0-1)")

    @field_validator("chemistry", mode="before")
    @classmethod
    def _lower_chemistry(cls, v: Any) -> Any:
        return v.strip().lower() if isinstance(v, str) else v

    @model_validator(mode="after")
    def _check_battery(self) -> BatteryCreate:
        if self.current_charge_kwh is not None:
            if self.capacity_kwh is None:
                raise ValueError("current_charge_kwh requires capacity_kwh")
            if self.current_charge_kwh > self.capacity_kwh:
                raise ValueError("current_charge_kwh cannot exceed capacity_kwh")
            soc = self.current_charge_kwh / self.capacity_kwh
            if self.state_of_charge is not None and abs(soc - self.state_of_charge) > 1e-6:
                raise ValueError("current_charge_kwh and state_of_charge disagree; pass one")
        if (
            self.soc_min is not None
            and self.soc_max is not None
            and not self.soc_min < self.soc_max
        ):
            raise ValueError("soc_min must be less than soc_max")
        for key in ("max_charge_kw", "max_discharge_kw"):
            v = getattr(self, key)
            if v is not None and v > self.rated_power:
                raise ValueError(f"{key} cannot exceed rated_power")
        return self

    def configured_soc(self) -> float | None:
        if self.current_charge_kwh is not None and self.capacity_kwh:
            return self.current_charge_kwh / self.capacity_kwh
        return self.state_of_charge


class SolarCreate(ResourceCreate):
    """Create a solar resource."""

    resource_type: Literal[ResourceType.SOLAR] = ResourceType.SOLAR
    dc_capacity_kw: float | None = Field(None, gt=0, description="DC (array) nameplate in kW")
    ac_capacity_kw: float | None = Field(None, gt=0, description="AC (inverter) nameplate in kW")
    panel_area_m2: float | None = Field(None, gt=0, description="Total panel area in m²")
    panel_efficiency: float | None = Field(None, gt=0, le=1, description="Panel efficiency 0-1")


class WindTurbineCreate(ResourceCreate):
    """Create a wind turbine resource (``resource_type`` ``wind_turbine`` or ``wind``)."""

    resource_type: Literal[ResourceType.WIND_TURBINE] = ResourceType.WIND_TURBINE
    rotor_diameter_m: float | None = Field(None, gt=0, description="Rotor diameter in metres")
    hub_height_m: float | None = Field(None, gt=0, description="Hub height in metres")
    cut_in_speed_ms: float | None = Field(None, ge=0, description="Cut-in wind speed in m/s")
    cut_out_speed_ms: float | None = Field(None, gt=0, description="Cut-out wind speed in m/s")
    rated_speed_ms: float | None = Field(None, gt=0, description="Rated wind speed in m/s")

    @model_validator(mode="after")
    def _check_speeds(self) -> WindTurbineCreate:
        ci, co, rs = self.cut_in_speed_ms, self.cut_out_speed_ms, self.rated_speed_ms
        if ci is not None and co is not None and co <= ci:
            raise ValueError("cut_out_speed must be greater than cut_in_speed")
        if rs is not None:
            if ci is not None and rs <= ci:
                raise ValueError("rated_speed must be greater than cut_in_speed")
            if co is not None and rs >= co:
                raise ValueError("rated_speed must be less than cut_out_speed")
        return self


def _resource_type_tag(value: Any) -> str | None:
    raw = (
        value.get("resource_type")
        if isinstance(value, dict)
        else getattr(value, "resource_type", None)
    )
    tag = normalize_resource_type(raw)
    return tag if isinstance(tag, str) else None


#: Request body of ``POST /api/v1/resources``.
ResourceCreateRequest = Annotated[
    Annotated[BatteryCreate, Tag(ResourceType.BATTERY.value)]
    | Annotated[SolarCreate, Tag(ResourceType.SOLAR.value)]
    | Annotated[WindTurbineCreate, Tag(ResourceType.WIND_TURBINE.value)],
    Discriminator(
        _resource_type_tag,
        custom_error_type="invalid_resource_type",
        custom_error_message=(
            "resource_type must be one of 'battery', 'solar', 'wind_turbine' (alias 'wind')"
        ),
    ),
]

resource_create_adapter: TypeAdapter[BatteryCreate | SolarCreate | WindTurbineCreate] = (
    TypeAdapter(ResourceCreateRequest)
)

CREATE_MODELS: dict[str, type[ResourceCreate]] = {
    ResourceType.BATTERY.value: BatteryCreate,
    ResourceType.SOLAR.value: SolarCreate,
    ResourceType.WIND_TURBINE.value: WindTurbineCreate,
}


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------


class ResourceUpdate(BaseModel):
    """Partial update (``PUT /api/v1/resources/{id}``).

    Only the fields sent are changed; sending ``null`` for an optional
    type-specific field clears it. Type-specific fields must belong to the
    resource's type (a solar field on a battery is a 422), and the merged
    result is re-validated with the type's create model, so cross-field rules
    (charge <= capacity, soc_min < soc_max, cut-in < rated < cut-out) hold
    after every update. ``resource_type`` cannot be changed.
    """

    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(None, min_length=1, max_length=255)
    rated_power: float | None = Field(None, gt=0)
    metadata: dict[str, Any] | None = None
    online: bool | None = None
    efficiency: float | None = Field(None, gt=0, le=1)
    # Battery
    capacity_kwh: float | None = Field(None, gt=0)
    state_of_charge: float | None = Field(None, ge=0, le=1)
    current_charge_kwh: float | None = Field(None, ge=0)
    chemistry: str | None = Field(None, max_length=16)
    nominal_voltage: float | None = Field(None, gt=0)
    charge_efficiency: float | None = Field(None, gt=0, le=1)
    discharge_efficiency: float | None = Field(None, gt=0, le=1)
    max_charge_kw: float | None = Field(None, gt=0)
    max_discharge_kw: float | None = Field(None, gt=0)
    soc_min: float | None = Field(None, ge=0, le=1)
    soc_max: float | None = Field(None, ge=0, le=1)
    # Solar
    dc_capacity_kw: float | None = Field(None, gt=0)
    ac_capacity_kw: float | None = Field(None, gt=0)
    panel_area_m2: float | None = Field(None, gt=0)
    panel_efficiency: float | None = Field(None, gt=0, le=1)
    # Wind
    rotor_diameter_m: float | None = Field(None, gt=0)
    hub_height_m: float | None = Field(None, gt=0)
    cut_in_speed_ms: float | None = Field(None, ge=0)
    cut_out_speed_ms: float | None = Field(None, gt=0)
    rated_speed_ms: float | None = Field(None, gt=0)


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------


class ResourceResponse(BaseModel):
    """A resource as returned by the API (all types; type-specific fields are
    ``null`` when not applicable or not recorded)."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    resource_type: str = Field(description="battery | solar | wind_turbine")
    rated_power: float
    online: bool = True
    current_power: float = 0.0
    efficiency: float = 0.95
    site_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime

    # Battery
    capacity_kwh: float | None = Field(None, description="Nameplate energy capacity (kWh)")
    state_of_charge: float | None = Field(None, description="SOC as a 0-1 fraction")
    state_of_charge_source: Literal["telemetry", "configured"] | None = None
    current_charge_kwh: float | None = Field(
        None, description="state_of_charge * capacity_kwh, when both are known"
    )
    state_of_health: float | None = Field(None, description="SOH as a 0-1 fraction")
    equivalent_full_cycles: float | None = Field(
        None, description="Cumulative throughput / (2 * capacity_kwh)"
    )
    chemistry: str | None = None
    nominal_voltage: float | None = None
    charge_efficiency: float | None = None
    discharge_efficiency: float | None = None
    max_charge_kw: float | None = None
    max_discharge_kw: float | None = None
    soc_min: float | None = None
    soc_max: float | None = None

    # Solar
    dc_capacity_kw: float | None = None
    ac_capacity_kw: float | None = None
    panel_area_m2: float | None = None
    panel_efficiency: float | None = None

    # Wind
    rotor_diameter_m: float | None = None
    hub_height_m: float | None = None
    cut_in_speed_ms: float | None = None
    cut_out_speed_ms: float | None = None
    rated_speed_ms: float | None = None


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
