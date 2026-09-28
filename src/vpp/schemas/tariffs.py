"""Pydantic v2 schemas for tariff CRUD and bill simulation."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from vpp.tariffs.nem import NEMConfigError, normalize_avoided_cost, normalize_regime


class TariffCreate(BaseModel):
    """Request body for creating a tariff."""

    name: str = Field(..., min_length=1, max_length=255)
    utility: str = Field("", max_length=255)
    urdb_json: dict[str, Any] = Field(..., description="URDB-shaped tariff JSON")
    effective_date: date | None = None
    urdb_label: str | None = Field(default=None, max_length=128)


class TariffUpdate(BaseModel):
    """Partial update — all fields optional."""

    name: str | None = Field(default=None, min_length=1, max_length=255)
    utility: str | None = Field(default=None, max_length=255)
    urdb_json: dict[str, Any] | None = None
    effective_date: date | None = None
    urdb_label: str | None = Field(default=None, max_length=128)


class TariffComponentView(BaseModel):
    """One human-readable rate component, derived from the parsed tariff."""

    name: str
    kind: str = Field(description="energy | tier | demand | fixed | minimum | adder | tax")
    unit: str = Field(description="$/kWh, $/kW, $/month, $/day, or 'fraction' (0.03 = 3%)")
    rate: float | None = None
    rates: list[float] | None = Field(default=None, description="Tier rates, ascending")
    tiers: list[dict[str, float | None]] | None = Field(
        default=None, description="[{max_kwh (null = unbounded), rate}]"
    )
    sell_rate: float | None = Field(default=None, description="URDB export (sell) rate")
    schedule: list[str] = Field(default_factory=list, description="When a TOU period applies")
    detail: str | None = None


class TariffRead(BaseModel):
    """Response model for tariff resources.

    ``urdb_json`` is the source of truth. The remaining fields after it are
    *derived* for display from the same parsed components the bill engine
    uses; they are read-only and ignored on create/update.
    """

    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    utility: str
    urdb_label: str | None = None
    urdb_json: dict[str, Any]
    effective_date: date | None = None
    created_at: datetime
    updated_at: datetime
    # ---- derived, presentational -------------------------------------
    sector: str | None = None
    source: str | None = Field(
        default=None, description="'URDB' when imported from OpenEI, else the JSON's source"
    )
    description: str | None = None
    components: list[TariffComponentView] = Field(default_factory=list)
    tou_heatmap: list[list[float]] | None = Field(
        default=None, description="12x24 weekday import rate ($/kWh), [month][hour]"
    )
    tou_heatmap_weekend: list[list[float]] | None = Field(
        default=None, description="12x24 weekend/holiday import rate ($/kWh)"
    )
    is_tou: bool = False
    nem_regime: str = Field(
        default="none", description="Default export-credit regime derived from the tariff"
    )
    nem_source: str = Field(default="default", description="tariff | urdb_dgrules | default")
    parse_error: str | None = Field(
        default=None, description="Set when the URDB JSON cannot be billed"
    )


class TariffPresetSummary(BaseModel):
    id: str
    name: str
    utility: str | None = None
    sector: str | None = None
    description: str | None = None
    source_date: str | None = None
    illustrative: bool = False


class TariffPresetRead(TariffPresetSummary):
    urdb_json: dict[str, Any]


class URDBImportStatus(BaseModel):
    configured: bool
    detail: str


class MeterTraceDTO(BaseModel):
    """API representation of a metered interval-energy trace."""

    timestamps: list[datetime] = Field(
        ..., description="Interval-start timestamps (ISO, tz-aware)"
    )
    import_kwh: list[float]
    export_kwh: list[float] = Field(default_factory=list)
    interval_minutes: int = 60

    @model_validator(mode="after")
    def _check_lengths(self) -> MeterTraceDTO:
        n = len(self.timestamps)
        if len(self.import_kwh) != n:
            raise ValueError("import_kwh length must equal timestamps length")
        if self.export_kwh and len(self.export_kwh) != n:
            raise ValueError("export_kwh length must equal timestamps length")
        return self


class SyntheticLoadSpec(BaseModel):
    """Deterministic, illustrative load shape (see ``vpp.tariffs.synthetic_load``)."""

    profile: Literal["residential", "commercial"] | None = Field(
        default=None, description="Defaults from the tariff sector"
    )
    avg_kw: float | None = Field(default=None, gt=0, le=100_000, description="Average load kW")
    pv_kw: float = Field(default=0.0, ge=0, le=100_000, description="Rooftop PV peak kW")
    interval_minutes: Literal[15, 30, 60] = 60


class BillSimulationRequest(BaseModel):
    """Bill simulation against a stored tariff (``tariff_id``) or inline ``urdb_json``.

    Load source -- exactly one of:

    * ``meter_trace``: explicit interval data (``billing_period_start/end``
      required);
    * ``synthetic``: ``true`` or a :class:`SyntheticLoadSpec`; covers
      ``period_days`` from ``billing_period_start`` (default: first day of the
      current month in ``timezone``);
    * ``csv``: an interval-meter CSV (see ``vpp.tariffs.csv_trace``); the
      billing window defaults to the span of the data.

    ``timezone`` (IANA) sets local time for TOU classification, billing-cycle
    boundaries and NEM3 hourly avoided cost; default UTC.

    NEM (M4): ``nem`` is 'none' | 'nem2' | 'nem3' | 'net_billing'; when
    omitted the regime is derived from the tariff (see ``vpp.tariffs.nem``).
    ``nem3_avoided_cost`` is a $/kWh vector by local time: 24 (hour of day),
    12 x 24 (month x hour, nested or flat 288), 8760 / 8784 (hour of year)
    or a single flat value (see ``vpp.tariffs.nem``); falls back to the
    tariff's ``nem3_avoided_cost``.

    ``compare_to``: another stored tariff id billed on the same load.
    """

    tariff_id: str | None = None
    urdb_json: dict[str, Any] | None = None
    meter_trace: MeterTraceDTO | None = None
    synthetic: bool | SyntheticLoadSpec | None = None
    csv: str | None = Field(default=None, max_length=8_000_000)
    period_days: int = Field(default=30, ge=1, le=366)
    billing_period_start: datetime | None = None
    billing_period_end: datetime | None = None
    timezone: str = "UTC"
    billing_cycle: Literal["auto", "single", "monthly"] = "auto"
    nem: str | None = None
    nem3_avoided_cost: list[float] | list[list[float]] | None = None
    compare_to: str | None = None

    @field_validator("nem")
    @classmethod
    def _nem(cls, v: str | None) -> str | None:
        if v is None:
            return None
        try:
            return normalize_regime(v)
        except NEMConfigError as exc:
            raise ValueError(str(exc)) from exc

    @field_validator("nem3_avoided_cost")
    @classmethod
    def _avoided_cost(cls, v: list[float] | list[list[float]] | None) -> list[float] | None:
        try:
            values = normalize_avoided_cost(v)
        except NEMConfigError as exc:
            raise ValueError(str(exc)) from exc
        return list(values) or None

    @field_validator("timezone")
    @classmethod
    def _tz(cls, v: str) -> str:
        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

        try:
            ZoneInfo(v)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"unknown timezone {v!r}") from exc
        return v

    @model_validator(mode="after")
    def _xor(self) -> BillSimulationRequest:
        if self.tariff_id is not None and self.urdb_json is not None:
            raise ValueError("Provide exactly one of tariff_id or urdb_json")
        sources = [
            self.meter_trace is not None,
            bool(self.synthetic),
            self.csv is not None,
        ]
        if sum(sources) != 1:
            raise ValueError("Provide exactly one load source: meter_trace, synthetic or csv")
        if self.meter_trace is not None and (
            self.billing_period_start is None or self.billing_period_end is None
        ):
            raise ValueError("meter_trace requires billing_period_start and billing_period_end")
        if (
            self.billing_period_start is not None
            and self.billing_period_end is not None
            and self.billing_period_end <= self.billing_period_start
        ):
            raise ValueError("billing_period_end must be after billing_period_start")
        return self


class BillLineItemDTO(BaseModel):
    kind: str
    label: str
    quantity: float
    unit: str
    rate: float
    amount: float


class BillCycleDTO(BaseModel):
    period_start: datetime
    period_end: datetime
    total: float
    export_credit: float = 0.0


class LoadSummary(BaseModel):
    source: Literal["meter_trace", "synthetic", "csv"]
    method: str | None = Field(default=None, description="How a synthetic load was built")
    timezone: str
    interval_minutes: int
    intervals: int
    import_kwh: float
    export_kwh: float
    peak_kw: float


class BillResponse(BaseModel):
    total: float
    tariff_name: str = ""
    line_items: list[BillLineItemDTO]
    period_start: datetime
    period_end: datetime
    # ---- additive (non-breaking) ---------------------------------------
    tariff_id: str | None = None
    currency: str = "USD"
    cycles: list[BillCycleDTO] = Field(default_factory=list)
    nem_regime: str = "none"
    nem_source: str = Field(default="default", description="request | tariff | urdb_dgrules")
    export_kwh: float = 0.0
    export_credit: float = 0.0
    notes: list[str] = Field(default_factory=list)
    load_summary: LoadSummary | None = None
    comparison: BillResponse | None = Field(
        default=None, description="Same load billed against compare_to"
    )


class URDBImportRequest(BaseModel):
    """Request to import a URDB record from openei.org."""

    urdb_label: str = Field(
        ..., min_length=1, max_length=128, description="URDB record id (getpage)"
    )
    name_override: str | None = None
