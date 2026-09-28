"""Pydantic schemas for optimization operations."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# Payload bounds. Schedules are persisted with every run and returned by the
# history endpoints, so their size is capped: 288 steps = 3 days at 15 min or
# 1 day at 5 min.
MAX_SCHEDULE_STEPS = 288
MAX_BACKTEST_STEPS = 672  # 1 week at 15 min
MAX_BACKTEST_WORK = 672 * 48  # ticks x horizon_steps
MAX_STOCHASTIC_SIZE = 20_000  # scenarios x horizon hours


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

class DispatchRequest(BaseModel):
    """Request to split a site power target across the online resources.

    ``target_power_kw`` is export-positive: a negative target asks the fleet
    to absorb power (batteries charge; renewables cannot absorb).
    """

    target_power_kw: float = Field(
        ..., ge=-1_000_000, le=1_000_000,
        description="Target total power in kW (export-positive; negative = absorb)",
    )
    resource_ids: list[str] | None = Field(
        None, max_length=500,
        description="Restrict dispatch to these resource ids (default: all online resources)",
    )
    resource_constraints: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description=(
            "Per-resource constraints keyed by resource id or name. Supported keys: "
            "exclude (bool), max_kw, min_kw (<= 0, limits absorption), cost_per_kwh"
        ),
    )
    interval_minutes: int = Field(
        15, ge=1, le=240,
        description="Dispatch interval; bounds battery output by the energy available over it",
    )
    wear_cost: bool = Field(True, description="Price battery throughput with the SOH-aware wear cost")
    replacement_cost_per_kwh: float = Field(250.0, gt=0, le=10_000)
    timeout_ms: int = Field(5000, gt=0, le=60_000, description="Solver timeout in ms")
    force_fallback: bool = Field(False, description="Force rule-based fallback")


class ResourceAllocation(BaseModel):
    """Power allocation for a single resource."""

    resource_id: str
    resource_name: str
    allocated_power_kw: float
    max_power_kw: float
    resource_type: str | None = None
    min_power_kw: float | None = None
    available_power_kw: float | None = Field(
        None, description="Upper bound actually used for this interval (after SOC/availability limits)"
    )
    marginal_cost_per_kwh: float | None = None
    state_of_charge: float | None = None
    soc_source: str | None = None
    availability_basis: str | None = None


class DispatchResponse(BaseModel):
    """Result of a dispatch operation."""

    success: bool
    target_power_kw: float
    actual_power_kw: float
    allocations: list[ResourceAllocation]
    solve_time_ms: float
    fallback_used: bool = False
    message: str = ""
    run_id: str | None = None
    status: str = ""
    method: str = Field(
        "", description="Solver path used, e.g. pyomo_highs_allocation or proportional_allocation_rules"
    )
    shortfall_kw: float = 0.0
    objective_value: float = Field(0.0, description="Linear marginal cost of the allocation over the interval")


# ---------------------------------------------------------------------------
# Generic optimization
# ---------------------------------------------------------------------------

class OptimizationRequest(BaseModel):
    """Generic optimization request."""

    problem_type: str = Field(..., description="stochastic | realtime | distributed")
    parameters: dict[str, Any] = Field(default_factory=dict)
    timeout_ms: int = Field(5000, gt=0, le=300_000)
    force_fallback: bool = False


class OptimizationResponse(BaseModel):
    """Result of an optimization run."""

    status: str
    objective_value: float
    solution: dict[str, Any]
    solve_time_ms: float
    fallback_used: bool = False
    solver: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    run_id: str | None = None


# ---------------------------------------------------------------------------
# Stochastic optimization
# ---------------------------------------------------------------------------

class StochasticRequest(BaseModel):
    """Request for stochastic (CVaR) battery dispatch over price scenarios."""

    num_scenarios: int = Field(50, ge=1, le=10_000, description="Number of scenarios")
    time_horizon_hours: int = Field(24, ge=1, le=168)
    risk_level: float = Field(0.05, gt=0, lt=1, description="CVaR tail probability (alpha = 1 - risk_level)")
    base_prices: list[float] = Field(default_factory=list)
    base_load: list[float] = Field(default_factory=list)
    volatility: float = Field(0.2, ge=0, le=5, description="Price volatility factor (log-normal sigma)")
    risk_weight: float = Field(
        0.5, ge=0, le=100,
        description="lambda in  E[cost] + lambda * CVaR_(1-risk_level)[cost]",
    )
    seed: int | None = Field(None, description="Seed for reproducible scenario generation")
    resource_id: str | None = Field(None, description="Battery resource to optimise (default: inline parameters)")
    battery_capacity_kwh: float = Field(1000.0, gt=0, le=1_000_000)
    max_power_kw: float = Field(250.0, gt=0, le=1_000_000)
    soc_init: float = Field(0.5, ge=0, le=1)
    soc_min: float = Field(0.1, ge=0, lt=1)
    soc_max: float = Field(0.9, gt=0, le=1)
    efficiency: float = Field(0.95, gt=0, le=1, description="One-way charge/discharge efficiency")
    timeout_ms: int = Field(10_000, gt=0, le=300_000)
    force_fallback: bool = False

    @model_validator(mode="after")
    def _check(self) -> StochasticRequest:
        n = self.time_horizon_hours
        for name in ("base_prices", "base_load"):
            vals = getattr(self, name)
            if vals and len(vals) != n:
                raise ValueError(f"{name} must have time_horizon_hours ({n}) entries")
        if self.num_scenarios * n > MAX_STOCHASTIC_SIZE:
            raise ValueError(
                f"num_scenarios x time_horizon_hours must be <= {MAX_STOCHASTIC_SIZE} "
                "(extensive-form model size)"
            )
        if not (self.soc_min <= self.soc_init <= self.soc_max) or self.soc_min >= self.soc_max:
            raise ValueError("require soc_min <= soc_init <= soc_max and soc_min < soc_max")
        return self


# ---------------------------------------------------------------------------
# Real-time optimization
# ---------------------------------------------------------------------------

class RealTimeRequest(BaseModel):
    """Request for real-time / fast-dispatch optimization."""

    grid_frequency_hz: float = Field(50.0, ge=45, le=55)
    grid_voltage_pu: float = Field(1.0, ge=0.8, le=1.2)
    active_power_demand_kw: float = Field(0.0)
    reactive_power_demand_kvar: float = Field(0.0)
    forecasts: dict[str, Any] = Field(default_factory=dict)
    timeout_ms: int = Field(1000, gt=0, le=10_000)
    force_fallback: bool = False


# ---------------------------------------------------------------------------
# Distributed optimization
# ---------------------------------------------------------------------------

class SiteData(BaseModel):
    """Data for a single VPP site in distributed optimization."""

    site_id: str
    resources: list[dict[str, Any]] = Field(default_factory=list)
    local_load_kw: float = 0.0
    local_generation_kw: float = 0.0


class DistributedRequest(BaseModel):
    """Request for distributed multi-site optimization."""

    sites: list[SiteData] = Field(..., min_length=1)
    target_power_kw: float = 0.0
    target_reserve_kw: float = 0.0
    coordination_mode: str = Field("merit_order", description="merit_order | equal_split | priority")
    timeout_ms: int = Field(30_000, gt=0, le=300_000)
    force_fallback: bool = False


# ---------------------------------------------------------------------------
# Schedule (horizon MPC) optimization
# ---------------------------------------------------------------------------

def _check_series(name: str, values: list[float] | None, n: int) -> None:
    if values is not None and len(values) != n:
        raise ValueError(f"{name} must have {n} entries (one per step), got {len(values)}")


def _check_interval(v: int) -> int:
    if 60 % v != 0:
        raise ValueError("interval_minutes must divide 60 (5, 6, 10, 12, 15, 20, 30 or 60)")
    return v


class ScheduleRequest(BaseModel):
    """Optimise a battery charge/discharge schedule over a horizon.

    Provide either ``prices`` (one per step, currency/kWh) or ``tariff_id``
    (prices derived from a stored tariff over ``horizon_hours`` starting at
    ``horizon_start``).
    """

    resource_ids: list[str] | None = Field(
        None, max_length=50, description="Battery resource ids (default: all online batteries)"
    )
    prices: list[float] | None = Field(None, min_length=1, max_length=MAX_SCHEDULE_STEPS)
    tariff_id: str | None = None
    horizon_start: datetime | None = Field(None, description="Tariff mode: first interval start (default: now)")
    horizon_hours: int | None = Field(None, ge=1, le=168, description="Tariff mode: horizon length (default 24)")
    nem: Literal["none", "nem2", "nem3"] = Field("nem2", description="Tariff mode: export compensation regime")
    interval_minutes: int = Field(60, ge=5, le=60)
    load_kw: list[float] | None = Field(None, max_length=MAX_SCHEDULE_STEPS)
    solar_kw: list[float] | None = Field(None, max_length=MAX_SCHEDULE_STEPS)
    degradation_aware: bool = Field(
        False, description="Add the SOH-aware battery wear cost (from each battery's persisted SOH/chemistry)"
    )
    replacement_cost_per_kwh: float = Field(250.0, gt=0, le=10_000)
    feeder_max_import_kw: float | None = Field(None, ge=0)
    feeder_max_export_kw: float | None = Field(None, ge=0)
    terminal_soc_policy: Literal["value", "hold"] = Field(
        "value",
        description="hold: end at >= initial SOC; value: free terminal SOC, stored energy credited at the mean price",
    )
    timeout_ms: int = Field(10_000, gt=0, le=60_000)
    force_fallback: bool = False

    @field_validator("interval_minutes")
    @classmethod
    def _divides_hour(cls, v: int) -> int:
        return _check_interval(v)

    @property
    def steps(self) -> int:
        if self.prices is not None:
            return len(self.prices)
        return (self.horizon_hours or 24) * 60 // self.interval_minutes

    @model_validator(mode="after")
    def _check(self) -> ScheduleRequest:
        if (self.prices is None) == (self.tariff_id is None):
            raise ValueError("provide exactly one of prices or tariff_id")
        if self.tariff_id is not None and self.steps > MAX_SCHEDULE_STEPS:
            raise ValueError(
                f"horizon_hours x (60 / interval_minutes) must be <= {MAX_SCHEDULE_STEPS} steps"
            )
        _check_series("load_kw", self.load_kw, self.steps)
        _check_series("solar_kw", self.solar_kw, self.steps)
        return self


class ScheduleResourcePlan(BaseModel):
    charge: list[float]
    discharge: list[float]
    soc: list[float] = Field(description="State of charge fraction at the end of each step")


class ScheduleResponse(BaseModel):
    """Optimised schedule. ``power = charge - discharge`` (positive = charging)."""

    run_id: str
    status: str
    method: str
    fallback_used: bool
    fallback_reason: str | None = None
    solve_time_ms: float
    objective_value: float | None = None
    energy_cost: float
    wear_cost: float = 0.0
    interval_minutes: int
    prices: list[float]
    charge: list[float]
    discharge: list[float]
    power: list[float]
    per_resource: dict[str, ScheduleResourcePlan]
    resources: list[dict[str, Any]]
    tariff_id: str | None = None
    terminal_soc_policy: str
    notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

class BatterySpec(BaseModel):
    """Inline battery description (used when no resource_id is given)."""

    capacity_kwh: float = Field(..., gt=0, le=1_000_000)
    max_power_kw: float = Field(..., gt=0, le=1_000_000)
    soc_init: float = Field(0.5, ge=0, le=1)
    soc_min: float = Field(0.05, ge=0, lt=1)
    soc_max: float = Field(0.95, gt=0, le=1)
    eta_charge: float = Field(0.95, gt=0, le=1)
    eta_discharge: float = Field(0.95, gt=0, le=1)
    state_of_health: float = Field(1.0, gt=0, le=1)

    @model_validator(mode="after")
    def _check(self) -> BatterySpec:
        if not (self.soc_min <= self.soc_init <= self.soc_max) or self.soc_min >= self.soc_max:
            raise ValueError("require soc_min <= soc_init <= soc_max and soc_min < soc_max")
        return self


class BacktestRequest(BaseModel):
    """Closed-loop receding-horizon MPC replay over a historical price series."""

    resource_id: str | None = Field(None, description="Battery to backtest (uses its persisted capacity/SOC/SOH)")
    battery: BatterySpec | None = None
    prices: list[float] = Field(..., min_length=2, max_length=MAX_BACKTEST_STEPS)
    load_kw: list[float] | None = Field(None, max_length=MAX_BACKTEST_STEPS)
    solar_kw: list[float] | None = Field(None, max_length=MAX_BACKTEST_STEPS)
    interval_minutes: int = Field(60, ge=5, le=60)
    horizon_steps: int = Field(24, ge=2, le=96)
    forecast_mode: Literal["perfect", "persistence", "noisy"] = "perfect"
    noise_sigma: float = Field(0.1, ge=0, le=2)
    seed: int | None = None
    terminal_soc_policy: Literal["value", "hold"] = "value"
    compare_offline: bool = Field(True, description="Also solve the perfect-foresight offline optimum")
    solver_timeout_ms: int = Field(2_000, gt=0, le=10_000, description="Per-tick solver time limit")
    start: datetime | None = None

    @field_validator("interval_minutes")
    @classmethod
    def _divides_hour(cls, v: int) -> int:
        return _check_interval(v)

    @model_validator(mode="after")
    def _check(self) -> BacktestRequest:
        if (self.resource_id is None) == (self.battery is None):
            raise ValueError("provide exactly one of resource_id or battery")
        n = len(self.prices)
        _check_series("load_kw", self.load_kw, n)
        _check_series("solar_kw", self.solar_kw, n)
        if n * self.horizon_steps > MAX_BACKTEST_WORK:
            raise ValueError(f"len(prices) x horizon_steps must be <= {MAX_BACKTEST_WORK}")
        return self


class BacktestResponse(BaseModel):
    run_id: str
    status: str
    ticks: int
    realized_cost: float
    realized_cost_adjusted: float
    no_action_cost: float
    rules_cost: float
    rules_cost_adjusted: float
    perfect_foresight_cost_adjusted: float | None = None
    perfect_foresight_status: str
    regret: float | None = None
    terminal_energy_value_per_kwh: float
    terminal_soc_policy: str
    final_soc: float
    fallback_count: int
    cumulative_solve_time_ms: float
    cumulative_solver_iterations: int
    wall_time_s: float
    forecast_mode: str
    interval_minutes: int
    horizon_steps: int
    charge: list[float]
    discharge: list[float]
    power: list[float]
    soc: list[float]
    notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Run history / explainer
# ---------------------------------------------------------------------------

class OptimizationRunRead(BaseModel):
    """A persisted optimization run (``DispatchRun`` in the web UI)."""

    id: str
    problem_type: str
    status: str
    created_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    objective_value: float | None = None
    solve_time_ms: float | None = None
    solver: str | None = None
    fallback_used: bool = False
    iterations: int | None = None
    gap: float | None = None
    resource_id: str | None = None
    total_cost: float | None = None
    inputs: dict[str, Any] | None = None
    solution: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class ExplainerStep(BaseModel):
    step: int
    charge: float | None = None
    discharge: float | None = None
    power: float | None = None
    price: float | None = None
    cost: float | None = None


class ExplainerRun(BaseModel):
    name: str
    total_cost: float = Field(
        description="Energy cost with the change in stored energy credited at the mean price"
    )
    energy_cost: float | None = None
    stored_energy_delta_kwh: float | None = None
    per_step: list[ExplainerStep]


class ExplainerBindingConstraint(BaseModel):
    name: str
    description: str | None = None
    step: int | None = None
    slack: float


class ExplainerResponse(BaseModel):
    run_id: str
    actual: ExplainerRun
    counterfactuals: list[ExplainerRun]
    binding_constraints: list[ExplainerBindingConstraint]
    rationale: str | None = None
