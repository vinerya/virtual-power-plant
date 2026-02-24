"""Pydantic schemas for optimization operations."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

class DispatchRequest(BaseModel):
    """Request to dispatch power across resources."""

    target_power_kw: float = Field(..., description="Target total power output in kW")
    resource_constraints: dict[str, Any] = Field(
        default_factory=dict,
        description="Per-resource constraints, keyed by resource name",
    )
    timeout_ms: int = Field(5000, gt=0, le=60_000, description="Solver timeout in ms")
    force_fallback: bool = Field(False, description="Force rule-based fallback")


class ResourceAllocation(BaseModel):
    """Power allocation for a single resource."""

    resource_id: str
    resource_name: str
    allocated_power_kw: float
    max_power_kw: float


class DispatchResponse(BaseModel):
    """Result of a dispatch operation."""

    success: bool
    target_power_kw: float
    actual_power_kw: float
    allocations: list[ResourceAllocation]
    solve_time_ms: float
    fallback_used: bool = False
    message: str = ""


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


# ---------------------------------------------------------------------------
# Stochastic optimization
# ---------------------------------------------------------------------------

class StochasticRequest(BaseModel):
    """Request for stochastic optimization."""

    num_scenarios: int = Field(50, ge=1, le=10_000, description="Number of scenarios")
    time_horizon_hours: int = Field(24, ge=1, le=168)
    risk_level: float = Field(0.05, gt=0, lt=1, description="CVaR confidence level")
    base_prices: list[float] = Field(default_factory=list)
    base_load: list[float] = Field(default_factory=list)
    volatility: float = Field(0.2, ge=0, description="Price volatility factor")
    timeout_ms: int = Field(10_000, gt=0, le=300_000)
    force_fallback: bool = False


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
