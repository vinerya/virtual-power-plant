"""Strict, JSON-Schema-exportable mirror of :class:`vpp.config.VPPConfig`.

``VPPConfig`` and its components are plain dataclasses with lenient
``from_dict`` constructors: unknown keys are silently dropped and bad types
only surface (if at all) in ``validate()``. That is fine for library use but
not for an API that accepts operator-edited YAML, where a typo such as
``time_horizn: 48`` must be rejected rather than ignored.

The pydantic models below mirror the dataclass fields one-to-one (a test
asserts the field sets stay in sync), forbid unknown keys, and encode the
same bounds ``validate()`` checks, so that

* ``GET /api/v1/config/schema`` can publish a JSON Schema the web console
  validates against client-side (Ajv), and
* ``PUT /api/v1/config`` can return precise, path-addressed errors.

Semantic checks that need the whole document (e.g. duplicate resource
names) still run through ``VPPConfig.validate()`` after this structural pass.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

__all__ = [
    "VPPConfigDocument",
    "validate_config_mapping",
    "vpp_config_json_schema",
]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ObjectiveDocument(_Strict):
    name: str = Field(..., min_length=1)
    weight: float = Field(1.0, ge=0, le=1)
    priority: int = Field(1, ge=1)
    enabled: bool = True
    parameters: dict[str, Any] = Field(default_factory=dict)


class ConstraintDocument(_Strict):
    name: str = Field(..., min_length=1)
    enabled: bool = True
    parameters: dict[str, Any] = Field(default_factory=dict)
    violation_penalty: float = Field(1000.0, ge=0)


class OptimizationDocument(_Strict):
    strategy: str = Field("linear_programming", min_length=1)
    objectives: list[ObjectiveDocument] = Field(default_factory=list)
    constraints: list[ConstraintDocument] = Field(default_factory=list)
    time_horizon: int = Field(24, gt=0, description="Hours")
    time_step: int = Field(15, gt=0, description="Minutes")
    solver_timeout: int = Field(300, gt=0, description="Seconds")
    solver_options: dict[str, Any] = Field(default_factory=dict)


class HeuristicDocument(_Strict):
    algorithm: str = Field("genetic_algorithm", min_length=1)
    parameters: dict[str, Any] = Field(default_factory=dict)
    max_iterations: int = Field(1000, gt=0)
    convergence_tolerance: float = Field(1e-6, gt=0)
    random_seed: int | None = None


class RuleDocument(_Strict):
    name: str = Field(..., min_length=1)
    enabled: bool = True
    priority: int = Field(1, ge=1)
    conditions: dict[str, Any] = Field(default_factory=dict)
    actions: dict[str, Any] = Field(default_factory=dict)


class RuleEngineDocument(_Strict):
    inference_method: Literal["forward_chaining", "backward_chaining"] = "forward_chaining"
    conflict_resolution: Literal["priority", "specificity", "recency"] = "priority"
    rules: list[RuleDocument] = Field(default_factory=list)
    max_inference_depth: int = Field(100, gt=0)
    enable_explanation: bool = True


class MonitoringDocument(_Strict):
    enabled: bool = True
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_file: str | None = None
    metrics_collection: bool = True
    performance_profiling: bool = False
    alert_thresholds: dict[str, float] = Field(default_factory=dict)
    dashboard_enabled: bool = False
    dashboard_port: int = Field(8080, ge=1, le=65535)


class SimulationDocument(_Strict):
    enabled: bool = False
    start_time: str | None = Field(None, description="ISO-8601")
    end_time: str | None = Field(None, description="ISO-8601")
    time_step_minutes: int = Field(15, gt=0)
    weather_simulation: bool = True
    market_simulation: bool = True
    random_seed: int | None = None
    monte_carlo_runs: int = Field(1, gt=0)


class SecurityDocument(_Strict):
    enable_authentication: bool = False
    api_key_required: bool = False
    rate_limiting: bool = True
    max_requests_per_minute: int = Field(100, gt=0)
    allowed_ips: list[str] = Field(default_factory=list)
    encryption_enabled: bool = False


class ResourceDocument(_Strict):
    name: str = Field(..., min_length=1)
    type: str = Field(..., min_length=1, description="battery | solar | wind | generator | load")
    enabled: bool = True
    parameters: dict[str, Any] = Field(default_factory=dict)
    constraints: dict[str, Any] = Field(default_factory=dict)


class VPPConfigDocument(_Strict):
    """Top-level VPP configuration document (YAML/JSON)."""

    model_config = ConfigDict(extra="forbid", title="VPPConfig")

    name: str = Field("Virtual Power Plant", min_length=1)
    description: str = ""
    location: str = ""
    timezone: str = Field("UTC", min_length=1)
    optimization: OptimizationDocument = Field(default_factory=OptimizationDocument)
    heuristics: HeuristicDocument = Field(default_factory=HeuristicDocument)
    rules: RuleEngineDocument = Field(default_factory=RuleEngineDocument)
    monitoring: MonitoringDocument = Field(default_factory=MonitoringDocument)
    simulation: SimulationDocument = Field(default_factory=SimulationDocument)
    security: SecurityDocument = Field(default_factory=SecurityDocument)
    resources: list[ResourceDocument] = Field(default_factory=list)
    enable_hot_reload: bool = False
    backup_config: bool = True
    config_version: str = "1.0"


def vpp_config_json_schema() -> dict[str, Any]:
    """Return the JSON Schema for a VPP config document.

    No ``$schema`` dialect URI is emitted on purpose: the console compiles
    it with Ajv's default (draft-07) build, which rejects unknown dialect
    URIs, and the generated keywords (``$defs``/``$ref``, ``anyOf``,
    ``enum``, numeric bounds, ``additionalProperties``) are understood by
    draft-07 and 2020-12 validators alike.
    """
    return VPPConfigDocument.model_json_schema()


def _format_loc(loc: tuple[Any, ...]) -> str:
    """Render a pydantic error location as a JSON Pointer (Ajv-style)."""
    if not loc:
        return "$"
    return "/" + "/".join(str(p) for p in loc)


def validate_config_mapping(
    data: Any,
) -> tuple[VPPConfigDocument | None, list[dict[str, str]]]:
    """Structurally validate a parsed config document.

    Returns ``(document, [])`` on success or ``(None, errors)`` where each
    error is ``{"path": "/optimization/time_horizon", "message": "..."}``.
    """
    if not isinstance(data, dict):
        return None, [{"path": "$", "message": "configuration must be a mapping"}]
    try:
        return VPPConfigDocument.model_validate(data), []
    except ValidationError as exc:
        return None, [
            {"path": _format_loc(tuple(err["loc"])), "message": err["msg"]}
            for err in exc.errors()
        ]
